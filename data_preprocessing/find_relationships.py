import os

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from tqdm import tqdm  # Optional: progress bar

# Parameters
PATCH_SIZE = 448
PATCH_HALF = PATCH_SIZE // 2
#
datasets = ['Our']
levels = ['08', '16']

for data in datasets:
    for tmp in [i.split('.')[0] for i in os.listdir(f'./datasets/{data}/WSI')]:
        for level in levels:
            if level == '08':
                NUM_POINTS = 14 * 14
            else:
                NUM_POINTS = 7 * 7

            # --- Input paths ---
            patch_csv_path = Path(f"./datasets/{data}/cropped_img/csv_index/{tmp}.csv")
            parquet_path = Path(f"./datasets/{data}/Raw_data/{tmp}/binned_outputs/square_0{level}um/spatial/filtered_in_tissue.parquet")

            # --- Load data ---
            patch_df = pd.read_csv(patch_csv_path)
            points_df = pd.read_parquet(parquet_path)  # Only read x/y columns
            if tmp == 'KV':
                coords = points_df.iloc[:, -2:].to_numpy()  # shape (N, 2)
            else:
                tmpset = int(tmp.split('-')[-1])
                if tmpset % 2 != 0:
                    print(tmpset)
                    coords = points_df.iloc[:, -2:].to_numpy() * 16   # shape (N, 2)
                else:
                    coords = points_df.iloc[:, -2:].to_numpy()  # shape (N, 2)

            # --- Build KDTree ---
            tree = cKDTree(coords)

            # --- Process each patch ---
            valid_rows = []
            parquet_indices_list = []

            for i, row in tqdm(patch_df.iterrows(), total=len(patch_df)):
                cx, cy = row["center_x"], row["center_y"]
                patch_left, patch_top = cx - PATCH_HALF, cy - PATCH_HALF
                patch_right, patch_bottom = cx + PATCH_HALF, cy + PATCH_HALF

                # Query the nearest NUM_POINTS points
                dists, idxs = tree.query([cx, cy], k=NUM_POINTS)

                selected = coords[idxs]
                x_in = (selected[:, 0] >= patch_left) & (selected[:, 0] < patch_right)
                y_in = (selected[:, 1] >= patch_top) & (selected[:, 1] < patch_bottom)
                in_patch = x_in & y_in

                # Keep only patches with at least 1/4 of NUM_POINTS inside the patch
                if np.sum(in_patch) >= NUM_POINTS // 4:
                    valid_rows.append(i)
                    parquet_indices_list.append(idxs.tolist())

            # --- Add and save ---
            patch_df = patch_df.loc[valid_rows].copy()
            patch_df["parquet_indices"] = parquet_indices_list

            if parquet_indices_list:
                max_index_used = max(max(idxs) for idxs in parquet_indices_list)
                print(f" {tmp} | Level {level} | Max index used: {max_index_used}")
            else:
                print(f"️  {tmp} | Level {level} | No valid patch, no indices used")

            output_path = patch_csv_path.with_name(f"{tmp}_0{level}_patch_index_filtered.csv")
            patch_df.to_csv(output_path, index=False)

            print(f"Saved {len(patch_df)} valid patches to: {output_path}")
