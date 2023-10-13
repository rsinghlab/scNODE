'''
Description:
    Pre-processing of mouse pancreatic data.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>

Reference:
    [1] https://github.com/theislab/pancreatic-endocrinogenesis/blob/master/scRNA_seq_qc_preprocessing_clustering.ipynb
'''
import scanpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tarfile
import functools



# day_dir_name = ["GSM3852752_E12_5", "GSM3852753_E13_5", "GSM3852754_E14_5", "GSM3852755_E15_5"]

def loadAndQC():
    # Read cellranger files for all four samples
    filename = './raw/GSM3852752_E12_5_counts/mm10/matrix.mtx'
    filename_genes = './raw/GSM3852752_E12_5_counts/mm10/genes.tsv'
    filename_barcodes = './raw/GSM3852752_E12_5_counts/mm10/barcodes.tsv'
    e125 = scanpy.read(filename).transpose()
    gene_list = np.genfromtxt(filename_genes, dtype=str)[:, 1]
    cell_list = np.genfromtxt(filename_barcodes, dtype=str)
    unique_gene_list, unique_gene_idx = np.unique(gene_list, return_index=True)
    e125 = e125[:, unique_gene_idx]
    e125.var_names = unique_gene_list
    e125.obs_names = cell_list
    print(e125.shape)

    filename = './raw/GSM3852753_E13_5_counts/mm10/matrix.mtx'
    filename_genes = './raw/GSM3852753_E13_5_counts/mm10/genes.tsv'
    filename_barcodes = './raw/GSM3852753_E13_5_counts/mm10/barcodes.tsv'
    e135 = scanpy.read(filename).transpose()
    gene_list = np.genfromtxt(filename_genes, dtype=str)[:, 1]
    cell_list = np.genfromtxt(filename_barcodes, dtype=str)
    e135 = e135[:, unique_gene_idx]
    e135.var_names = unique_gene_list
    e135.obs_names = cell_list
    print(e135.shape)

    filename = './raw/GSM3852754_E14_5_counts/mm10/matrix.mtx'
    filename_genes = './raw/GSM3852754_E14_5_counts/mm10/genes.tsv'
    filename_barcodes = './raw/GSM3852754_E14_5_counts/mm10/barcodes.tsv'
    e145 = scanpy.read(filename).transpose()
    gene_list = np.genfromtxt(filename_genes, dtype=str)[:, 1]
    cell_list = np.genfromtxt(filename_barcodes, dtype=str)
    e145 = e145[:, unique_gene_idx]
    e145.var_names = unique_gene_list
    e145.obs_names = cell_list
    print(e145.shape)

    filename = './raw/GSM3852755_E15_5_counts/mm10/matrix.mtx'
    filename_genes = './raw/GSM3852755_E15_5_counts/mm10/genes.tsv'
    filename_barcodes = './raw/GSM3852755_E15_5_counts/mm10/barcodes.tsv'
    e155 = scanpy.read(filename).transpose()
    gene_list = np.genfromtxt(filename_genes, dtype=str)[:, 1]
    cell_list = np.genfromtxt(filename_barcodes, dtype=str)
    e155 = e155[:, unique_gene_idx]
    e155.var_names = unique_gene_list
    e155.obs_names = cell_list
    print(e155.shape)

    # Add dev. timepoint label for each sample
    e125.obs['day'] = '12.5'
    e135.obs['day'] = '13.5'
    e145.obs['day'] = '14.5'
    e155.obs['day'] = '15.5'
    # Create Concatenated anndata object for all timepoints
    all_ann = e125.concatenate(e135, e145, e155)
    # Deleting individual day arrays
    del e125
    del e135
    del e145
    del e155
    print("All data shape: ", all_ann.shape)
    # QC
    all_ann.obs['n_genes'] = (all_ann.X > 0).sum(axis=1)
    mt_gene_mask = [gene.startswith('mt-') for gene in all_ann.var_names]
    mt_gene_index = np.where(mt_gene_mask)[0]
    all_ann.obs['mt_frac'] = np.asarray(all_ann.X[:, mt_gene_index].sum(axis=1) / all_ann.X.sum(axis=1)).squeeze()
    all_ann = all_ann[all_ann.obs['mt_frac'] < 0.2]
    scanpy.pp.filter_cells(all_ann, min_genes=1200)
    scanpy.pp.filter_genes(all_ann, min_cells=20)
    print("Filtered data shape: ", all_ann.shape)
    # Save filtered data
    all_ann.write_h5ad("./raw/filtered_data.h5ad")


def splitDataset():
    all_ann = scanpy.read_h5ad("./raw/filtered_data.h5ad")
    print("Data shape: ", all_ann.shape)
    day_name = ["12.5", "13.5", "14.5", "15.5"]
    all_ann.obs.day = all_ann.obs.day.apply(lambda x: day_name.index(x))

    # Assign cell type label
    annotated_data = scanpy.read_h5ad("./raw/GSE132188_adata.h5ad.h5")
    common_cells = np.intersect1d(annotated_data.obs_names.values, all_ann.obs_names.values)
    cell_idx = [i for i, c in enumerate(all_ann.obs_names.values) if c in common_cells]
    cell_type = np.asarray(["NAN" for _ in range(all_ann.shape[0])])
    cell_type[cell_idx] = annotated_data.obs.loc[common_cells]["clusters_fig2_final"].values
    all_ann.obs["cell_type"] = cell_type

    # Split into train & test
    unique_days = all_ann.obs['day'].unique()
    num_cells_list = [all_ann[all_ann.obs.day == t].shape[0] for t in unique_days]
    print("Num of tps: ", len(unique_days))
    print("Num cells: ", num_cells_list)
    split_type = "one_forecasting"  # one_forecasting, one_interpolation, two_forecasting, two_interpolation
    if split_type == "one_interpolation":
        train_tps = [0, 1, 3]
        test_tps = [2]
    elif split_type == "one_forecasting":
        train_tps = [0, 1, 2]
        test_tps = [3]
    print("Train tps: ", train_tps)
    print("Test tps: ", test_tps)
    # -----
    train_adata = all_ann[np.where(all_ann.obs['day'].apply(lambda x: x in train_tps))[0], :]
    print("Train data shape: ", train_adata.shape)
    hvgs_summary = scanpy.pp.highly_variable_genes(
        scanpy.pp.log1p(train_adata, copy=True), n_top_genes=2000, inplace=False
    )
    hvgs = train_adata.var.index.values[hvgs_summary.highly_variable]
    adata = all_ann[:, hvgs]
    print("HVG data shape: ", adata.shape)
    # -----
    print("Saving data...")
    adata.to_df().to_csv("./processed/{}-count_data-hvg.csv".format(split_type))  # cell x genes
    pd.DataFrame(hvgs).to_csv("./processed/{}-var_genes_list.csv".format(split_type))
    meta_df = all_ann.obs
    meta_df.to_csv("./processed/meta_data.csv")

    # # Visualization
    # print("Visualization...")
    # vis_data = all_ann.copy()
    # scanpy.pp.neighbors(vis_data, n_neighbors=50, n_pcs=None)
    # scanpy.tl.umap(vis_data, min_dist=0.5)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 8))
    # scanpy.pl.umap(vis_data, color="day", ax=ax1, show=False)
    # scanpy.pl.umap(vis_data, color="cell_type", legend_loc="on data", ax=ax2, show=False)
    # plt.show()



if __name__ == '__main__':
    # loadAndQC()
    splitDataset()