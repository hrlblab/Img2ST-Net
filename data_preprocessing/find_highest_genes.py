import os
from pathlib import Path
import numpy as np
import scipy.io
from tqdm import tqdm

def load_mtx_file(file_path):
    """Read a single .mtx file and convert it to a dense matrix"""
    return scipy.io.mmread(file_path).toarray()  # shape: [num_genes, num_spots]

def compute_average_expression_from_folders(folder_paths, top_k=300, output_path='top_gene_indices.npy'):
    """Read .mtx files from multiple folders and compute the top_k genes with the highest average expression"""
    all_gene_expr = []
    total_files = 0

    # Support multiple input paths
    for folder_path in folder_paths:
        folder = Path(folder_path)
        mtx_files = list(folder.rglob("*.mtx"))  # support recursive search
        total_files += len(mtx_files)

        print(f"Found {len(mtx_files)} .mtx files in {folder_path}")
        for mtx_file in tqdm(mtx_files, desc=f"Processing {folder_path}"):
            expr_matrix = load_mtx_file(mtx_file)
            gene_means = expr_matrix.mean(axis=1)  # compute mean expression per gene
            all_gene_expr.append(gene_means)

    assert all_gene_expr, "No .mtx files found in any provided folders."

    # Combine all per-gene average expression vectors by stacking along columns
    all_gene_expr = np.stack(all_gene_expr, axis=1)  # shape: [num_genes, num_files]
    print(all_gene_expr.shape)

    # Average across all files -> global mean expression per gene
    overall_gene_avg = all_gene_expr.mean(axis=1)

    # Get indices of top_k genes with highest expression
    top_indices = np.argsort(overall_gene_avg)[::-1][:top_k]

    # Save indices
    np.save(output_path, top_indices)
    print(f"\nTotal {total_files} .mtx files processed.")
    print(f"Saved top-{top_k} gene indices to {output_path}")
    return top_indices


# ✅ Example usage
if __name__ == "__main__":
    levels = ['08', '16']
    datasets = ['BC', 'CRC', 'Our']

    for level in levels:
        for data in datasets:
            folders = [
                f'./datasets/{data}/Raw_data/{i}/binned_outputs/square_0{level}um/filtered_feature_bc_matrix'
                for i in os.listdir(f'/data/zhuj28/Neurips2025/datasets/{data}/Raw_data')
            ]
            print(folders)
            out_path = f'./datasets/{data}/Raw_data/{level}_top_gene_indices.npy'
            top_gene_indices = compute_average_expression_from_folders(folders, output_path=out_path)
