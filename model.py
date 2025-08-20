import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple

# -------------------------
# Basic U-Net style blocks
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class MiniUNet(nn.Module):
    """
    Input:  (B,1024,14,14) feature map
    Output: (B,512,14,14) enhanced feature
    Structure: encoder -> downsample -> deeper encoder -> upsample -> decoder with skip connection
    """
    def __init__(self, in_ch=1024, mid_ch=512, out_ch=512):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, mid_ch)  # 14x14
        self.pool = nn.MaxPool2d(2)           # downsample: 14->7
        self.enc2 = ConvBlock(mid_ch, mid_ch) # 7x7
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec  = ConvBlock(mid_ch + mid_ch, out_ch)  # concat skip connection
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool(x1)
        x3 = self.enc2(x2)
        x4 = self.up(x3)
        x  = torch.cat([x4, x1], dim=1)
        return self.dec(x)  # (B,512,14,14)


# -------------------------
# Main model with Image + ST branches
# -------------------------
class MultiBranchSpatialPredictorV2(nn.Module):
    """
    Returns:
      img_pred: (B, bin, pred_dim)  # prediction stream for regression tasks
      st_pred:  (B, bin, pred_dim) or None
      img_ctr:  (B, bin, ctr_dim)   # contrastive embedding stream
      st_ctr:   (B, bin, ctr_dim) or None
    """
    def __init__(self,
                 bin_num: int,
                 st_in_dim: Optional[int],
                 pred_dim: int = 300,
                 ctr_dim: int = 256,
                 densenet_weights: str = "IMAGENET1K_V1"):
        super().__init__()
        # Backbone: DenseNet121 (feature extractor only)
        try:
            densenet = models.densenet121(weights=getattr(models.DenseNet121_Weights, densenet_weights))
        except Exception:
            densenet = models.densenet121(weights=densenet_weights)
        self.backbone = densenet.features  # output: (B,1024,14,14)

        self.unet = MiniUNet(1024, 512, 512)

        # Match target bin_num (grid size of output patches)
        self.bin_num = bin_num
        if bin_num == 196:   # 14x14
            self.resize_to_grid = nn.Identity(); self.hw = (14, 14)
        elif bin_num == 49:  # 7x7
            self.resize_to_grid = nn.AdaptiveAvgPool2d((7, 7)); self.hw = (7, 7)
        elif bin_num == 1:   # 1x1
            self.resize_to_grid = nn.AdaptiveAvgPool2d((1, 1)); self.hw = (1, 1)
        else:
            raise ValueError("Unsupported bin_num. Only support 1, 49, 196.")

        # Image branch: prediction head
        self.image_pred_head = nn.Sequential(
            nn.Conv2d(512, 512, 1), nn.ReLU(inplace=True),
            nn.Conv2d(512, pred_dim, 1)
        )
        # Image branch: contrastive head
        self.image_ctr_head = nn.Sequential(
            nn.Conv2d(512, 512, 1), nn.ReLU(inplace=True),
            nn.Conv2d(512, ctr_dim, 1)
        )

        # ST branch (shared layers + dual heads)
        self.has_st = st_in_dim is not None
        if self.has_st:
            hidden = max(pred_dim * 2, 256)
            self.st_shared = nn.Sequential(
                nn.Linear(st_in_dim, hidden),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            self.st_pred_head = nn.Linear(hidden, pred_dim)
            self.st_ctr_head  = nn.Linear(hidden, ctr_dim)

        self._init_weights()

    def _init_weights(self):
        # Initialize new layers; DenseNet already has pretrained weights
        for m in [self.unet, self.image_pred_head, self.image_ctr_head]:
            for mod in m.modules():
                if isinstance(mod, nn.Conv2d):
                    nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
                    if mod.bias is not None: nn.init.zeros_(mod.bias)
                elif isinstance(mod, nn.BatchNorm2d):
                    nn.init.ones_(mod.weight); nn.init.zeros_(mod.bias)
        if self.has_st:
            for mod in self.st_shared.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.xavier_uniform_(mod.weight)
                    if mod.bias is not None: nn.init.zeros_(mod.bias)
            nn.init.xavier_uniform_(self.st_pred_head.weight); nn.init.zeros_(self.st_pred_head.bias)
            nn.init.xavier_uniform_(self.st_ctr_head.weight);  nn.init.zeros_(self.st_ctr_head.bias)

    @staticmethod
    def _to_seq(x_2d: torch.Tensor) -> torch.Tensor:
        """Convert feature map (B, C, H, W) -> patch sequence (B, H*W, C)."""
        B, C, H, W = x_2d.shape
        return x_2d.permute(0, 2, 3, 1).reshape(B, H * W, C)

    def forward(self, image: torch.Tensor, st: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        # Image stream
        feat = self.backbone(image)          # (B,1024,14,14)
        feat = self.unet(feat)               # (B,512,14,14)
        feat = self.resize_to_grid(feat)     # (B,512,H,W) where H*W=bin_num

        # Prediction output (for regression/supervised tasks)
        img_pred_map = self.image_pred_head(feat)   # (B,pred_dim,H,W)
        img_pred = self._to_seq(img_pred_map)       # (B,bin,pred_dim)

        # Contrastive embedding output
        img_ctr_map = self.image_ctr_head(feat)     # (B,ctr_dim,H,W)
        img_ctr = self._to_seq(img_ctr_map)         # (B,bin,ctr_dim)

        st_pred = st_ctr = None
        if self.has_st and st is not None:
            # st: (B,bin,st_in_dim)
            shared = self.st_shared(st)             # (B,bin,hidden)
            st_pred = self.st_pred_head(shared)     # (B,bin,pred_dim)
            st_ctr  = self.st_ctr_head(shared)      # (B,bin,ctr_dim)

        return img_pred, st_pred, img_ctr, st_ctr


# -------------------------
# Symmetric InfoNCE loss
# -------------------------
class ImageSTContrastive(nn.Module):
    """
    Contrastive loss between image and ST embeddings (InfoNCE).
    Modes:
      - patch_agg='mean': aggregate patch embeddings -> (B,D), do sample-level contrast
      - patch_agg='patch': contrast at patch level (requires patch order alignment)
    """
    def __init__(self, temperature: float = 0.07, normalize: bool = True, patch_agg: str = 'mean'):
        super().__init__()
        assert patch_agg in ['mean', 'patch']
        self.tau = temperature
        self.normalize = normalize
        self.patch_agg = patch_agg

    def _norm(self, x):
        return F.normalize(x, dim=-1) if self.normalize else x

    def forward(self, img_ctr: torch.Tensor, st_ctr: torch.Tensor) -> torch.Tensor:
        if self.patch_agg == 'mean':
            # Sample-level: mean over patches
            img_vec = self._norm(img_ctr.mean(dim=1))  # (B,D)
            st_vec  = self._norm(st_ctr.mean(dim=1))   # (B,D)
            logits_i2s = (img_vec @ st_vec.t()) / self.tau
            logits_s2i = (st_vec @ img_vec.t()) / self.tau
            targets = torch.arange(img_vec.size(0), device=img_ctr.device)
            return 0.5 * (F.cross_entropy(logits_i2s, targets) +
                          F.cross_entropy(logits_s2i, targets))
        else:
            # Patch-level: flatten all patches across batch
            B, P, D = img_ctr.shape
            img_flat = self._norm(img_ctr.reshape(B * P, D))
            st_flat  = self._norm(st_ctr.reshape(B * P, D))
            logits_i2s = (img_flat @ st_flat.t()) / self.tau
            logits_s2i = (st_flat @ img_flat.t()) / self.tau
            targets = torch.arange(B * P, device=img_ctr.device)
            return 0.5 * (F.cross_entropy(logits_i2s, targets) +
                          F.cross_entropy(logits_s2i, targets))
