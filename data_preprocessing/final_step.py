import os
import numpy as np
import pandas as pd
from scipy.io import mmread
from tqdm import tqdm
import ast


def extract_selected_gene_values_pandas(csv_path, mtx_path, top_gene_indices_path, ds, fi, output_path):
    # Load CSV
    df = pd.read_csv(csv_path)
    print("CSV Header:", list(df.columns))

    # Load sparse expression matrix
    mtx = mmread(mtx_path).tocsc()  # Use CSC for efficient column slicing
    print(f"Loaded matrix of shape {mtx.shape}")

    # Load column indices (i.e., gene indices)
    top_gene_indices = np.load(top_gene_indices_path)

    save_results = []

    # Avoid calling .toarray() repeatedly: preload a submatrix (all top genes × all columns)
    # This uses more memory but significantly reduces repeated operations
    print("Preloading top_gene rows from matrix...")
    top_gene_matrix = mtx[top_gene_indices, :]  # shape: [300, num_spots]

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        tmp = {}
        img_path = f'./datasets/{ds}/cropped_img/{fi}/' + row['patch_name']
        img_index = (row["i"], row["j"])

        try:
            row_indices = ast.literal_eval(row["parquet_indices"])  # e.g. "[123, 456, 789]"
        except Exception as e:
            print(f"Error parsing row: {row['parquet_indices']}")
            continue

        try:
            # Slice multiple columns from the preloaded matrix (by each r_idx), then convert to dense
            results = top_gene_matrix[:, row_indices].T.toarray()  # shape: [N_indices, 300]
        except Exception as e:
            print(f"Matrix slice error: {e}")
            continue

        tmp['img_path'] = img_path
        tmp['position'] = img_index
        tmp['label'] = results
        save_results.append(tmp)

    np.save(output_path, save_results)
    print(f"Saved result array to {output_path}, len {len(save_results)}")
    if save_results:
        print("Example:", save_results[0])


levels = ['08', '16']
datasets = ['Our', 'CRC', 'BC']
for data in datasets:
    for level in levels:
        for tmp_file in [i.split('.')[0] for i in os.listdir(f'/data/zhuj28/Neurips2025/datasets/{data}/WSI')]:
            print(tmp_file)
            csv_path = f"./datasets/{data}/cropped_img/csv_index/{tmp_file}_0{level}_patch_index_filtered.csv"
            mtx_path = f"./datasets/{data}/Raw_data/{tmp_file}/binned_outputs/square_0{level}um/filtered_feature_bc_matrix/matrix.mtx"
            top_gene_indices_path = f"./datasets/{data}/Raw_data/{level}_top_gene_indices.npy"
            save_path = f'./datasets/{data}/data_infor/{level}/{tmp_file}.npy'

            extract_selected_gene_values_pandas(csv_path, mtx_path, top_gene_indices_path, data, tmp_file, save_path)
