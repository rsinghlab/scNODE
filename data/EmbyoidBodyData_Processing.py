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


def loadAndQC():
    ann_data_list = []
    for t in ["T0_1A", "T2_3B", "T4_5C", "T6_7D", "T8_9E"]:
        print("Loading data for {}...".format(t))
        filename = './raw/{}/matrix.mtx'.format(t)
        filename_genes = './raw/{}/genes.tsv'.format(t)
        filename_barcodes = './raw/{}/barcodes.tsv'.format(t)
        d_ann = scanpy.read(filename).transpose()
        gene_list = np.genfromtxt(filename_genes, dtype=str)[:, 1]
        cell_list = np.genfromtxt(filename_barcodes, dtype=str)
        unique_gene_list, unique_gene_idx = np.unique(gene_list, return_index=True)
        d_ann = d_ann[:, unique_gene_idx]
        d_ann.var_names = unique_gene_list
        d_ann.obs_names = cell_list
        d_ann.obs["day"] = t
        print(d_ann.shape)
        ann_data_list.append(d_ann)
    all_ann = ann_data_list[0].concatenate(*ann_data_list[1:])
    print("All data shape: ", all_ann.shape)
    # Filter out cells and genes
    scanpy.pp.filter_cells(all_ann, min_genes=1)
    scanpy.pp.filter_genes(all_ann, min_cells=1)
    print("Filtered data shape: ", all_ann.shape)
    # Save filtered data
    all_ann.write_h5ad("./raw/filtered_data.h5ad")


def splitDataset():
    all_ann = scanpy.read_h5ad("./raw/filtered_data.h5ad")
    print("Data shape: ", all_ann.shape)
    day_name = ["T0_1A", "T2_3B", "T4_5C", "T6_7D", "T8_9E"]
    all_ann.obs.day = all_ann.obs.day.apply(lambda x: day_name.index(x))

    # Split into train & test
    unique_days = all_ann.obs['day'].unique()
    num_cells_list = [all_ann[all_ann.obs.day == t].shape[0] for t in unique_days]
    print("Num of tps: ", len(unique_days))
    print("Num cells: ", num_cells_list)
    split_type = "one_interpolation"  # one_forecasting, one_interpolation
    if split_type == "one_interpolation":
        train_tps = [0, 1, 3, 4]
        test_tps = [2]
    elif split_type == "one_forecasting":
        train_tps = [0, 1, 2, 3]
        test_tps = [4]
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
    # scanpy.pl.umap(vis_data, color="day", show=False)
    # plt.show()



if __name__ == '__main__':
    # loadAndQC()
    splitDataset()