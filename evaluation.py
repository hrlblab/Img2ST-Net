import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import pandas as pd

def _safe_genewise_pcc(gts: np.ndarray, preds: np.ndarray):
    """
    Vectorized PCC over genes with safe handling for zero-variance (constant) vectors.
    gts, preds: (N, G)
    returns: pcc_per_gene: (G,)
    """
    # Center
    gts_c = gts - gts.mean(axis=0, keepdims=True)
    preds_c = preds - preds.mean(axis=0, keepdims=True)
    # Variances
    gts_std = np.sqrt((gts_c ** 2).mean(axis=0))
    preds_std = np.sqrt((preds_c ** 2).mean(axis=0))
    denom = gts_std * preds_std

    # Dot products
    num = (gts_c * preds_c).mean(axis=0)

    pcc = np.zeros_like(num, dtype=np.float64)
    mask = denom > 0
    pcc[mask] = num[mask] / denom[mask]
    # For denom==0 (constant GT or Pred), keep pcc=0
    return pcc

def evaluate_predictions(pred_path, gt_path, save_dir='./result/16_crc', patch_size=14):
    # --- IO & basic checks ---
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"pred_path not found: {pred_path}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"gt_path not found: {gt_path}")

    os.makedirs(save_dir, exist_ok=True)
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Load arrays
    preds_raw = np.load(pred_path)
    gts_raw   = np.load(gt_path)

    # Expect (..., 300); flatten spatial dims to N x 300
    preds = preds_raw.reshape(-1, preds_raw.shape[-1])
    gts   = gts_raw.reshape(-1, gts_raw.shape[-1])

    print(f"Loaded predictions from {pred_path}, shape: {preds.shape}")
    print(f"Loaded ground truths from {gt_path}, shape: {gts.shape}")

    assert preds.shape == gts.shape, f"Shape mismatch: preds {preds.shape} vs gts {gts.shape}"
    n_samples, n_genes = preds.shape

    # Derive number of patches (assumes square grid of patch_size x patch_size)
    patch_area = patch_size * patch_size
    assert n_samples % patch_area == 0, (
        f"n_samples={n_samples} is not divisible by patch_size^2={patch_area}. "
        f"Check patch_size or input arrays."
    )
    num_patches = n_samples // patch_area

    # --- Vectorized gene-wise metrics ---
    diff = preds - gts
    mse_per_gene = (diff ** 2).mean(axis=0)             # (G,)
    mae_per_gene = np.abs(diff).mean(axis=0)            # (G,)
    pcc_per_gene = _safe_genewise_pcc(gts, preds)       # (G,)

    print("\n===== Overall Evaluation =====")
    print(f"Average MSE over genes: {mse_per_gene.mean():.6f}")
    print(f"Average MAE over genes: {mae_per_gene.mean():.6f}")
    print(f"Average PCC over genes: {np.nanmean(pcc_per_gene):.6f}")

    overall_mse = (diff ** 2).mean()
    overall_mae = np.abs(diff).mean()
    print(f"Overall MSE (sample-level): {overall_mse:.6f}")
    print(f"Overall MAE (sample-level): {overall_mae:.6f}")

    # Relative variance difference
    pred_var = preds.var(axis=0)
    gt_var   = gts.var(axis=0)
    rvd = np.mean((pred_var - gt_var) ** 2 / (gt_var ** 2 + 1e-8))
    print(f"RVD (relative variance difference): {rvd:.6f}")

    # --- Patch-wise SSIM (multi-channel over gene axis) ---
    # Expect (num_patches, H, W, C)
    try:
        pred_patches = preds.reshape(num_patches, patch_size, patch_size, n_genes)
        gt_patches   = gts.reshape(num_patches, patch_size, patch_size, n_genes)

        ssim_scores = []
        for i in range(num_patches):
            pred_patch = pred_patches[i]
            gt_patch   = gt_patches[i]

            # Use joint data range; if zero, SSIM = 1.0 (identical constant patches)
            joint_min = min(pred_patch.min(), gt_patch.min())
            joint_max = max(pred_patch.max(), gt_patch.max())
            dr = joint_max - joint_min

            if dr == 0:
                ssim_scores.append(1.0)
                continue

            s = ssim(
                gt_patch,
                pred_patch,
                data_range=dr,
                channel_axis=-1   # last axis is channel (genes)
            )
            ssim_scores.append(float(s))

        mean_ssim = float(np.mean(ssim_scores))
        print(f"Patch-wise SSIM ({patch_size}x{patch_size} patches): {mean_ssim:.6f}")

        ssim_csv = os.path.join(save_dir, "ssim_per_patch.csv")
        pd.DataFrame({"patch_idx": np.arange(num_patches), "ssim": ssim_scores}) \
          .to_csv(ssim_csv, index=False)
        print(f"Saved patch-level SSIM scores to {ssim_csv}")
    except Exception as e:
        print(f"Failed to compute patch-wise SSIM: {e}")

    # --- Save gene-wise metrics ---
    metrics_df = pd.DataFrame({
        "gene_idx": np.arange(n_genes),
        "mse": mse_per_gene,
        "mae": mae_per_gene,
        "pcc": pcc_per_gene
    })
    metrics_csv = os.path.join(save_dir, "metrics_per_gene.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved gene-level metrics to {metrics_csv}")

    # --- Plot mean expression distribution (GT vs Pred) ---
    fig, ax = plt.subplots(figsize=(8, 8))

    gt_mean   = gts.mean(axis=0)
    pred_mean = preds.mean(axis=0)

    # Shift for non-negativity in plotting only
    min_val = min(gt_mean.min(), pred_mean.min())
    if min_val <= 0:
        shift = -min_val
        gt_mean   = gt_mean + shift
        pred_mean = pred_mean + shift

    # Sort by ground-truth mean
    sorted_idx = np.argsort(gt_mean)
    gt_mean_sorted   = gt_mean[sorted_idx]
    pred_mean_sorted = pred_mean[sorted_idx]

    ax.plot(np.arange(n_genes), gt_mean_sorted, label='Ground Truth', linewidth=2)
    ax.scatter(np.arange(n_genes), pred_mean_sorted, s=5, label='Predicted')

    ax.set_title("Mean Expression per Gene", fontsize=18)
    ax.set_xlabel("Gene Index", fontsize=14)
    ax.set_ylabel("Mean Expression", fontsize=14)
    ax.legend(fontsize=12)

    plt.tight_layout()
    save_plot_path = os.path.join(plot_dir, "gene_distribution.png")
    plt.savefig(save_plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved gene distribution plot to {save_plot_path}")


if __name__ == "__main__":
    pred_npy_path = './predictions/crc_08/predictions.npy'
    gt_npy_path   = './predictions/crc_08/ground_truths.npy'
    evaluate_predictions(pred_npy_path, gt_npy_path, save_dir='./result/crc_08', patch_size=7)


