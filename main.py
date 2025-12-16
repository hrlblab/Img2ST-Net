import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse
import logging
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch.nn as nn
import random
import torch
import torchvision
from skimage.metrics import structural_similarity as calc_ssim

from cross_validation import get_train_val_data
from model import Img2STNet, ContrastiveProjector
from data_loader import STDataset
from pytorch_msssim import ssim


def compute_ssim_loss(pred, target):
    """
    Compute SSIM Loss, consistent with calc_ssim in eval (win_size=7, Gaussian window)
    Args:
        pred: (B, bin_num, pred_dim) predicted values
        target: (B, bin_num, pred_dim) target values
    Returns:
        1 - SSIM as loss value
    """
    B, N, C = pred.shape
    H = int(N ** 0.5)  # 14 for bin_num=196

    # Reshape to (B, C, H, W) format
    pred = pred.reshape(B, H, H, C).permute(0, 3, 1, 2)
    target = target.reshape(B, H, H, C).permute(0, 3, 1, 2)

    # Dynamically compute data_range (consistent with eval, using joint range)
    joint_min = torch.min(pred.min(), target.min())
    joint_max = torch.max(pred.max(), target.max())
    data_range = joint_max - joint_min
    data_range = torch.clamp(data_range, min=1e-8)

    # Use pytorch_msssim, win_size=7 consistent with eval
    ssim_val = ssim(pred, target, data_range=data_range, size_average=True, win_size=7, K=(0.02, 0.05))

    return 1 - ssim_val


def save_checkpoint(payload, ckpt_path):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(payload, ckpt_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='../datasets/BC')
    parser.add_argument('--exp_name', type=str, default='166_BC_new')
    parser.add_argument('--max_iterations', type=int, default=1000000)
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--ctr_lr', type=float, default=1e-5)
    parser.add_argument('--ctr_model_lr', type=float, default=1e-6)
    parser.add_argument('--ctr_weight', type=float, default=0.01)
    parser.add_argument('--ssim_weight', type=float, default=0.5)
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--level', type=str, default='16')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--test_slide', type=str, default='D.npy')
    parser.add_argument('--bin_num', type=int, default=196)
    parser.add_argument('--pred_dim', type=int, default=300)

    parser.add_argument('--sample_ratio', type=float, default=1.0)
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    cudnn.benchmark = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
        np.random.seed(args.seed + worker_id)

    save_path = f'./weight/{args.exp_name}'
    os.makedirs(save_path, exist_ok=True)

    logging.basicConfig(filename=os.path.join(save_path, "log.txt"),
                        level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    img_transform = transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation((90, 90))]),
        torchvision.transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        torchvision.transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    mse_loss_fn = nn.MSELoss(reduction='mean')

    label_root = os.path.join(args.root_path, f'data_infor/{args.level}')
    train_data, val_data = get_train_val_data(label_root, args.test_slide,
                                               seed=args.seed)

    train_dataset = STDataset(train_data, transform=img_transform, root_path=args.root_path)
    val_dataset = STDataset(val_data, transform=test_transform, root_path=args.root_path)

    total_size = len(train_dataset)
    n_samples = int(total_size * args.sample_ratio)
    if n_samples < total_size:
        indices = np.random.choice(total_size, n_samples, replace=False)
        train_dataset = Subset(train_dataset, indices)
        logging.info(f"Using {n_samples}/{total_size} training samples")
    else:
        logging.info(f"Using all {total_size} training samples")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = Img2STNet(bin_num=args.bin_num, pred_dim=args.pred_dim).to(device)
    projector = ContrastiveProjector(img_dim=512, expr_dim=args.pred_dim).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9)
    optimizer_ctr = optim.Adam(projector.parameters(), lr=args.ctr_lr)

    writer = SummaryWriter(os.path.join(save_path, 'log'))
    iter_num = 0

    for epoch_num in range(args.epochs):
        model.train()
        projector.train()

        pbar = tqdm(total=len(train_loader), desc=f"Train Epoch {epoch_num}")
        mse_losses, ctr_losses, ssim_losses = [], [], []

        for batch_idx, (image_batch, label) in enumerate(train_loader):
            image_batch = image_batch.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            lr = args.base_lr * (1 - float(iter_num) / args.max_iterations) ** args.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            pred, feat = model(image_batch, return_feature=True)

            mse_loss = mse_loss_fn(pred, label)
            ssim_loss = compute_ssim_loss(pred, label)
            ctr_loss = projector(feat, label)

            # Main loss = MSE + SSIM
            main_loss = mse_loss + args.ssim_weight * ssim_loss

            optimizer.zero_grad(set_to_none=True)
            optimizer_ctr.zero_grad(set_to_none=True)

            main_loss.backward(retain_graph=True)
            mse_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

            optimizer.zero_grad(set_to_none=True)
            (args.ctr_weight * ctr_loss).backward()

            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in mse_grads:
                        p.data -= lr * mse_grads[n]
                    if p.grad is not None:
                        p.data -= args.ctr_model_lr * p.grad

            optimizer_ctr.step()

            iter_num += 1
            mse_losses.append(mse_loss.item())
            ssim_losses.append(ssim_loss.item())
            ctr_losses.append(args.ctr_weight * ctr_loss.item())

            writer.add_scalar('lr', lr, iter_num)
            writer.add_scalar('loss/mse', mse_loss.item(), iter_num)
            writer.add_scalar('loss/ssim', ssim_loss.item(), iter_num)
            writer.add_scalar('loss/ctr', args.ctr_weight * ctr_loss.item(), iter_num)

            pbar.set_postfix(MSE=f"{np.mean(mse_losses):.5f}", SSIM_L=f"{np.mean(ssim_losses):.5f}", CTR=f"{np.mean(ctr_losses):.5f}", LR=f"{lr:.6f}")
            pbar.update(1)

            if batch_idx % 20 == 0:
                logging.info(f"Iter {iter_num}, MSE: {mse_loss.item():.5f}, SSIM_L: {ssim_loss.item():.5f}, CTR: {args.ctr_weight * ctr_loss.item():.5f}, LR: {lr:.6f}")

        pbar.close()

        if epoch_num % 10 == 0 or epoch_num == args.epochs - 1:
            model.eval()
            mse_list, mae_list, ssim_list = [], [], []
            patch_size = int(np.sqrt(args.bin_num))

            val_pbar = tqdm(total=len(val_loader), desc=f"Eval Epoch {epoch_num}")
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(val_loader):
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    pred = model(images)
                    mse_val = nn.functional.mse_loss(pred, labels, reduction='mean').item()
                    mae_val = float(torch.mean(torch.abs(pred - labels)).item())
                    mse_list.append(mse_val)
                    mae_list.append(mae_val)

                    pred_np = pred.cpu().numpy()
                    labels_np = labels.cpu().numpy()
                    for i in range(pred_np.shape[0]):
                        p = pred_np[i].reshape(patch_size, patch_size, -1)
                        g = labels_np[i].reshape(patch_size, patch_size, -1)
                        # Compute data_range using joint range (consistent with eval)
                        joint_min = min(p.min(), g.min())
                        joint_max = max(p.max(), g.max())
                        dr = joint_max - joint_min
                        if dr > 0:
                            s = calc_ssim(g, p, data_range=dr, channel_axis=-1, K1=0.02, K2=0.05)
                            ssim_list.append(s)
                        else:
                            ssim_list.append(1.0)  # Consistent with eval

                    cur_mse = float(np.mean(mse_list))
                    cur_mae = float(np.mean(mae_list))
                    cur_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
                    val_pbar.set_postfix(MSE=f"{cur_mse:.4f}", MAE=f"{cur_mae:.4f}", SSIM=f"{cur_ssim:.4f}")
                    val_pbar.update(1)

            val_pbar.close()

            mean_mse = float(np.mean(mse_list)) if mse_list else 0.0
            mean_mae = float(np.mean(mae_list)) if mae_list else 0.0
            mean_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
            print(f"[Epoch {epoch_num}] Val MSE: {mean_mse:.4f}, MAE: {mean_mae:.4f}, SSIM: {mean_ssim:.4f}")
            logging.info(f"[Epoch {epoch_num}] Val MSE: {mean_mse:.4f}, MAE: {mean_mae:.4f}, SSIM: {mean_ssim:.4f}")

            writer.add_scalar('val/mse', mean_mse, epoch_num)
            writer.add_scalar('val/mae', mean_mae, epoch_num)
            writer.add_scalar('val/ssim', mean_ssim, epoch_num)

            csv_path = os.path.join(save_path, 'results.csv')
            header = not os.path.exists(csv_path)
            with open(csv_path, 'a') as f:
                if header:
                    f.write('epoch,mse,mae,ssim\n')
                f.write(f'{epoch_num},{mean_mse:.6f},{mean_mae:.6f},{mean_ssim:.6f}\n')

            ckpt_path = os.path.join(save_path, f"model_best_{epoch_num}.pth")
            payload = {
                'epoch': epoch_num,
                'model': model.state_dict(),
                'projector': projector.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': vars(args),
                'val': {'mse': mean_mse, 'mae': mean_mae, 'ssim': mean_ssim},
            }
            save_checkpoint(payload, ckpt_path)


if __name__ == "__main__":
    main()
