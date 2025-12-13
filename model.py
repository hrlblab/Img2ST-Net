import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ContrastiveProjector(nn.Module):
    """
    Independent contrastive learning module.
    Computes contrastive loss using features and expressions from Img2STNet.

    Usage:
        projector = ContrastiveProjector(img_dim=512, expr_dim=300)
        optimizer_ctr = Adam(projector.parameters(), lr=1e-5)  # very small lr

        pred, feat = model(image, return_feature=True)
        ctr_loss = projector(feat, expression)
        total_loss = mse_loss + 0.01 * ctr_loss  # small weight
    """
    def __init__(self,
                 img_dim: int = 512,
                 expr_dim: int = 300,
                 proj_dim: int = 128,
                 temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim)
        )
        self.expr_proj = nn.Sequential(
            nn.Linear(expr_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, img_feat: torch.Tensor, expr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_feat: (B, num_bins, 512) from Img2STNet
            expr: (B, num_bins, expr_dim) ground truth
        Returns:
            InfoNCE loss
        """
        B, N, _ = img_feat.shape
        img_flat = img_feat.reshape(B * N, -1)
        expr_flat = expr.reshape(B * N, -1)

        img_emb = F.normalize(self.img_proj(img_flat), dim=-1)
        expr_emb = F.normalize(self.expr_proj(expr_flat), dim=-1)

        logits = (img_emb @ expr_emb.t()) / self.temperature
        targets = torch.arange(B * N, device=img_feat.device)

        return 0.5 * (F.cross_entropy(logits, targets) +
                      F.cross_entropy(logits.t(), targets))


class Img2STNet(nn.Module):
    """
    Image to Spatial Transcriptomics prediction model with lightweight UNet-style.

    Architecture:
        - Encoder: ResNet backbone
        - Decoder: Single-layer upsampling + 1 skip connection
        - Output: 14x14 or 7x7 spatial grid

    Args:
        backbone: 'resnet50', 'resnet34', 'resnet18'
        bin_num: 196 (14x14) or 49 (7x7)
        pred_dim: prediction dimension per bin (default: 300 genes)
        pretrained: use ImageNet pretrained weights

    Input:  (B, 3, 448, 448)
    Output: (B, bin_num, pred_dim)
    """
    def __init__(self,
                 backbone: str = 'resnet50',
                 bin_num: int = 196,
                 pred_dim: int = 300,
                 pretrained: bool = True):
        super().__init__()

        self.bin_num = bin_num
        if bin_num == 196:
            self.out_size = 14
        elif bin_num == 49:
            self.out_size = 7
        else:
            raise ValueError("bin_num must be 196 (14x14) or 49 (7x7)")

        # Build encoder
        if backbone == 'resnet50':
            resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            self.enc_ch = [1024, 2048]  # layer3, layer4
        elif backbone == 'resnet34':
            resnet = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            self.enc_ch = [256, 512]
        elif backbone == 'resnet18':
            resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            self.enc_ch = [256, 512]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Encoder
        self.enc_front = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2
        )  # -> 56x56
        self.enc3 = resnet.layer3  # 56->28, ch[0], for skip
        self.enc4 = resnet.layer4  # 28->14, ch[1], bottleneck

        # Decoder: upsample + skip + downsample
        self.up_conv = nn.Sequential(
            nn.Conv2d(self.enc_ch[1], 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.skip_conv = nn.Conv2d(self.enc_ch[0], 256, 1)  # compress skip channels
        self.fuse = nn.Sequential(
            nn.Conv2d(512 + 256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, pred_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.up_conv, self.skip_conv, self.fuse, self.pred_head]:
            for mod in m.modules() if hasattr(m, 'modules') else [m]:
                if isinstance(mod, nn.Conv2d):
                    nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)
                elif isinstance(mod, nn.BatchNorm2d):
                    nn.init.ones_(mod.weight)
                    nn.init.zeros_(mod.bias)

    def forward(self, x: torch.Tensor, return_feature: bool = False):
        """
        Args:
            x: (B, 3, 448, 448) input image
            return_feature: if True, also return decoder feature for contrastive learning

        Returns:
            pred: (B, bin_num, pred_dim)
            feat: (B, bin_num, 512) only if return_feature=True
        """
        # Encoder
        x = self.enc_front(x)      # 56x56
        e3 = self.enc3(x)          # 28x28, skip
        e4 = self.enc4(e3)         # 14x14, bottleneck

        # Decoder: upsample -> concat skip -> process
        d = self.up_conv(e4)       # 14x14, 512ch
        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=False)  # 28x28
        skip = self.skip_conv(e3)  # 28x28, 256ch
        d = torch.cat([d, skip], dim=1)  # 28x28, 768ch
        d = self.fuse(d)           # 28x28, 512ch

        # Adjust to target resolution
        d = F.adaptive_avg_pool2d(d, (self.out_size, self.out_size))  # 14x14 or 7x7

        # Save feature for contrastive learning (512ch)
        B, C_feat, H, W = d.shape
        feat = d.permute(0, 2, 3, 1).reshape(B, H * W, C_feat)  # (B, bin_num, 512)

        # Predict
        pred = self.pred_head(d)   # out_size x out_size, pred_dim
        B, C, H, W = pred.shape
        pred = pred.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, bin_num, pred_dim)

        if return_feature:
            return pred, feat
        return pred
