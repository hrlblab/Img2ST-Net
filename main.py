import os
import sys
import glob
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse
import logging
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
import torch.nn as nn
import time
import random
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
from cross_validation import get_train_val_paths

# === import your model definitions ===
from model import MultiBranchSpatialPredictorV2, ImageSTContrastive
from data_loader import STdata, load_data_from_folder


def find_latest_checkpoint(path):
    """Return the newest checkpoint path under `path`, or None if not found."""
    checkpoints = sorted(glob.glob(os.path.join(path, "model_best_*.pth")))
    if not checkpoints:
        return None
    return checkpoints[-1]


def save_checkpoint(payload, ckpt_path):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(payload, ckpt_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='../datasets/BC')
    parser.add_argument('--exp_name', type=str, default='166_BC_new')
    parser.add_argument('--max_iterations', type=int, default=1000000)
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--level', type=str, default='16')
    parser.add_argument('--epochs', type=int, default=252)
    parser.add_argument('--test_slide', type=str, default='D.npy')
    parser.add_argument('--bin_num', type=int, default=1)
    parser.add_argument('--setting', type=str, default='raw', choices=['new', 'raw'])

    # new args for contrastive + weighting
    parser.add_argument('--embed_dim', type=int, default=300, help='Prediction head dim')
    parser.add_argument('--ctr_dim', type=int, default=256, help='Contrastive head dim')
    parser.add_argument('--ctr_temp', type=float, default=0.07, help='InfoNCE temperature')
    parser.add_argument('--patch_agg', type=str, default='mean', choices=['mean', 'patch'],
                        help='Contrast granularity: mean over patches or per-patch')
    parser.add_argument('--pred_weight', type=float, default=1.0, help='Weight for prediction loss')
    parser.add_argument('--ctr_weight', type=float, default=1.0, help='Weight for contrastive loss')

    args = parser.parse_args()

    # DDP initialization
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    cudnn.benchmark = True

    def worker_init_fn(worker_id):
        # Ensure each worker has a different seed
        random.seed(args.seed + worker_id)
        np.random.seed(args.seed + worker_id)

    save_path = f'./weight/{args.exp_name}'
    if rank == 0 and not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    if rank == 0:
        logging.basicConfig(filename=os.path.join(save_path, "log.txt"),
                            level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s',
                            datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))

    # Data transforms
    img_transform = transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation((90, 90))]),
        torchvision.transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        torchvision.transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Losses
    mse_loss_fn = nn.MSELoss(reduction='mean')

    # Build dataset file lists
    if args.setting == 'raw':
        label_root = os.path.join(args.root_path, f'raw_setting/{args.level}')
    else:
        label_root = os.path.join(args.root_path, f'data_infor/{args.level}')

    train_labels_path, val_labels_path = get_train_val_paths(label_root, args.test_slide)

    val_labels, train_labels = [], []
    for folder_path in val_labels_path:
        val_labels.extend(load_data_from_folder(folder_path))
    for folder_path in train_labels_path:
        train_labels.extend(load_data_from_folder(folder_path))

    # Datasets
    if args.setting == 'raw':
        train_dataset = STdata(train_labels, root=args.root_path, transform=img_transform, is_raw=True)
        val_dataset = STdata(val_labels, root=args.root_path, transform=test_transform, is_raw=True)
    else:
        train_dataset = STdata(train_labels, root=args.root_path, transform=img_transform)
        val_dataset = STdata(val_labels, root=args.root_path, transform=test_transform)

    # Samplers / Loaders
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # === Model ===
    st_in_dim = args.embed_dim
    model = MultiBranchSpatialPredictorV2(
        bin_num=args.bin_num,
        st_in_dim=st_in_dim,
        pred_dim=args.embed_dim,
        ctr_dim=args.ctr_dim,
    ).to(device)

    # Contrastive criterion (only uses *_ctr streams)
    ctr_criterion = ImageSTContrastive(
        temperature=args.ctr_temp,
        normalize=True,
        patch_agg=args.patch_agg
    )

    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9)

    writer = SummaryWriter(os.path.join(save_path, f'log/{rank}')) if rank == 0 else None

    # Optional resume (supports both old pure-state_dict and new dict payload)
    start_epoch = 0
    iter_num = 0
    latest_ckpt = find_latest_checkpoint(save_path)
    if latest_ckpt:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        ckpt = torch.load(latest_ckpt, map_location=map_location)
        if isinstance(ckpt, dict) and ('model' in ckpt or 'state_dict' in ckpt):
            # New-style: full payload
            state = ckpt.get('model', ckpt.get('state_dict'))
            model.module.load_state_dict(state, strict=True)
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt.get('epoch', 0) + 1
        else:
            # Legacy: pure state_dict
            model.module.load_state_dict(ckpt, strict=True)
            # Derive epoch number from filename
            start_epoch = int(os.path.basename(latest_ckpt).split('_')[-1].split('.')[0]) + 1
        if rank == 0:
            logging.info(f"[Rank {rank}] Resumed from {latest_ckpt}, starting at epoch {start_epoch}")

    # =========================
    # Training / Validation
    # =========================
    for epoch_num in range(start_epoch, args.epochs):
        model.train()
        train_sampler.set_epoch(epoch_num)

        if rank == 0:
            pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch_num}")

        for batch_idx, (image_batch, gene) in enumerate(train_loader):
            image_batch = image_batch.to(device, non_blocking=True)
            gene = gene.to(device, non_blocking=True)  # shape: (B, bin_num, embed_dim)

            # Polynomial LR decay
            lr = args.base_lr * (1 - float(iter_num) / args.max_iterations) ** args.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Forward
            img_pred, st_pred, img_ctr, st_ctr = model(image_batch, gene)

            # Losses
            loss_pred = mse_loss_fn(img_pred, gene)
            loss_ctr = ctr_criterion(img_ctr, st_ctr)
            loss = args.pred_weight * loss_pred + args.ctr_weight * loss_ctr

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            iter_num += 1
            if rank == 0 and writer:
                writer.add_scalar('lr', lr, iter_num)
                writer.add_scalar('loss/total', loss.item(), iter_num)
                writer.add_scalar('loss/pred_mse', loss_pred.item(), iter_num)
                writer.add_scalar('loss/ctr', loss_ctr.item(), iter_num)

                if (batch_idx % 20) == 0:
                    logging.info(
                        f"[GPU {rank}] Iter {iter_num}, "
                        f"Total: {loss.item():.5f}, PredMSE: {loss_pred.item():.5f}, "
                        f"Ctr: {loss_ctr.item():.5f}, LR: {lr:.6f}"
                    )

            if rank == 0:
                pbar.update(1)

        if rank == 0:
            pbar.close()

        # Validation (rank=0 only to avoid duplication)
        # Evaluate every 50 epochs or at the final epoch
        if (epoch_num % 50 == 0 or epoch_num == args.epochs - 1) and rank == 0 and epoch_num >= 0:
            model.eval()
            mse_list, mae_list, pcc_list = [], [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.cuda(rank, non_blocking=True)
                    labels = labels.cuda(rank, non_blocking=True)
                    # Only use prediction stream for validation metrics
                    img_pred, _, _, _ = model(images, labels)
                    mse_val = nn.functional.mse_loss(img_pred, labels, reduction='mean').item()
                    mae_val = float(torch.mean(torch.abs(img_pred - labels)).item())

                    mse_list.append(mse_val)
                    mae_list.append(mae_val)

            mean_mse = float(np.mean(mse_list)) if mse_list else 0.0
            mean_mae = float(np.mean(mae_list)) if mae_list else 0.0
            mean_pcc = float(np.mean(pcc_list)) if pcc_list else 0.0
            print(f"[Epoch {epoch_num}] Val MSE: {mean_mse:.4f}, MAE: {mean_mae:.4f}, PCC: {mean_pcc:.4f}")
            logging.info(f"[Epoch {epoch_num}] Val MSE: {mean_mse:.4f}, MAE: {mean_mae:.4f}, PCC: {mean_pcc:.4f}")

            if writer:
                writer.add_scalar('val/mse', mean_mse, epoch_num)
                writer.add_scalar('val/mae', mean_mae, epoch_num)
                writer.add_scalar('val/pcc', mean_pcc, epoch_num)

            # Save checkpoint (model + optimizer + epoch)
            ckpt_path = os.path.join(save_path, f"model_best_{epoch_num}.pth")
            payload = {
                'epoch': epoch_num,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': vars(args),
                'val': {'mse': mean_mse, 'mae': mean_mae, 'pcc': mean_pcc},
            }
            save_checkpoint(payload, ckpt_path)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
