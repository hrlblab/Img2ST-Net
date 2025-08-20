import os
import sys
import glob
import argparse
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from data_loader import STdata, load_data_from_folder
from cross_validation import get_train_val_paths
import tqdm

# import your new model
from model import MultiBranchSpatialPredictorV2


def _epoch_from_name(path: str) -> int:
    """Extract epoch number from filename like model_best_150.pth"""
    base = os.path.basename(path)
    try:
        return int(base.split("_")[-1].split(".")[0])
    except Exception:
        return -1


def find_latest_checkpoint(path: str):
    ckpts = glob.glob(os.path.join(path, "model_best_*.pth"))
    if not ckpts:
        return None
    ckpts.sort(key=_epoch_from_name)
    return ckpts[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='../datasets/BC')
    parser.add_argument('--exp_name', type=str, default='08_BC_new')
    parser.add_argument('--level', type=str, default='08')
    parser.add_argument('--test_slide', type=str, default='D.npy')
    parser.add_argument('--bin_num', type=int, default=196)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_dir', type=str, default='./predictions/new_08_BC_c1')
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--ctr_dim', type=int, default=256)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---------------- Load model ----------------
    # st_in_dim=None -> only image branch is active
    model = MultiBranchSpatialPredictorV2(
        bin_num=args.bin_num,
        st_in_dim=None,
        pred_dim=args.embed_dim,
        ctr_dim=args.ctr_dim
    ).to(device)

    save_path = f'./weight/{args.exp_name}'
    latest_ckpt = find_latest_checkpoint(save_path)
    if latest_ckpt is None:
        print(f"No checkpoint found under {save_path}")
        sys.exit(1)

    print(f"Loading checkpoint from {latest_ckpt}")
    ckpt = torch.load(latest_ckpt, map_location=device)

    if isinstance(ckpt, dict) and ('model' in ckpt or 'state_dict' in ckpt):
        state = ckpt.get('model', ckpt.get('state_dict'))
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    model.eval()

    # ---------------- Load data ----------------
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    label_root = os.path.join(args.root_path, f'data_infor/{args.level}')
    _, val_labels_path = get_train_val_paths(label_root, args.test_slide)

    val_labels = []
    for folder_path in val_labels_path:
        val_labels.extend(load_data_from_folder(folder_path))

    val_dataset = STdata(val_labels, root=args.root_path, transform=test_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # ---------------- Inference ----------------
    all_preds, all_gts = [], []

    with torch.no_grad():
        for images, labels in tqdm.tqdm(val_loader, desc="Predicting"):
            images = images.to(device, non_blocking=True)

            # forward: only use image branch output
            img_pred, _, _, _ = model(images)

            all_preds.append(img_pred.cpu().numpy())
            all_gts.append(labels.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_gts   = np.concatenate(all_gts, axis=0)

    # ---------------- Save ----------------
    os.makedirs(args.save_dir, exist_ok=True)
    np.save(os.path.join(args.save_dir, "predictions.npy"), all_preds)
    np.save(os.path.join(args.save_dir, "ground_truths.npy"), all_gts)

    print(f"Saved predictions to {os.path.join(args.save_dir, 'predictions.npy')}")
    print(f"Saved ground truths to {os.path.join(args.save_dir, 'ground_truths.npy')}")


if __name__ == "__main__":
    main()
