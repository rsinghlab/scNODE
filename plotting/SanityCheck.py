'''
Description:
    Sanity check of datasets.
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy
import natsort
import torch

from plotting.__init__ import *
from plotting.utils import linearSegmentCMap
from plotting.visualization import umapWithoutPCA, onlyPCA
from benchmark.BenchmarkUtils import preprocess2, loadZebrafishData, loadMammalianData, loadWOTData, loadDrosophilaData

# =========================================

zebrafish_data_dir = "../data/single_cell/experimental/zebrafish_embryonic/new_processed"
mammalian_data_dir = "../data/single_cell/experimental/mammalian_cerebral_cortex/new_processed"
wot_data_dir = "../data/single_cell/experimental/Schiebinger2019/processed/"
drosophila_data_dir = "../data/single_cell/experimental/drosophila_embryonic/processed/"


def loadData(data_name, split_type):
    print("[ Data={} | Split={} ] Loading data...".format(data_name, split_type))
    if data_name == "zebrafish":
        ann_data = loadZebrafishData(zebrafish_data_dir, split_type)
        print("Pre-processing...")
        processed_data = preprocess2(ann_data.copy())
        cell_types =  processed_data.obs["ZF6S-Cluster"].values
    elif data_name == "mammalian":
        ann_data = loadMammalianData(mammalian_data_dir, split_type)
        processed_data = ann_data.copy()
        cell_types = processed_data.obs.New_cellType.values
    elif data_name == "drosophila":
        ann_data = loadDrosophilaData(drosophila_data_dir, split_type)
        print("Pre-processing...")
        processed_data = preprocess2(ann_data.copy())
        cell_types = processed_data.obs.seurat_clusters.values
    elif data_name == "wot":
        ann_data = loadWOTData(wot_data_dir, split_type)
        processed_data = ann_data.copy()
        processed_data.obs["cell_type"] = ["Undefined" for _ in range(processed_data.shape[0])]
        with open("../data/single_cell/experimental/Schiebinger2019/raw/cell_sets.gmt", "r") as file:
            cell_sets = file.readlines()
        cell_sets = [each.split("\t") for each in cell_sets]
        cell_set_labels = [each[0] for each in cell_sets]
        cell_set_names = [each[2:] for each in cell_sets]
        for i in range(len(cell_set_labels)):
            processed_data.obs.loc[np.intersect1d(cell_set_names[i], processed_data.obs_names.values), "cell_type"] = cell_set_labels[i]
        cell_types = processed_data.obs.cell_type.values
    else:
        raise ValueError("Unknown data name.")
    cell_tps = ann_data.obs["tp"]
    n_tps = len(np.unique(cell_tps))
    n_genes = ann_data.shape[1]
    print("# tps={}, # genes={}".format(n_tps, n_genes))
    data = processed_data.X
    return processed_data, data, cell_tps, cell_types, n_genes, n_tps


def splitTP(data, cell_tps, cell_types, n_tps):
    traj_data_np = [data[np.where(cell_tps == t)[0], :] for t in range(1, n_tps + 1)]  # (# tps, # cells, # genes)
    traj_data_torch = [torch.FloatTensor(each) for each in traj_data_np]
    if cell_types is not None:
        traj_cell_types = [cell_types[np.where(cell_tps == t)[0]] for t in range(1, n_tps + 1)]  # (# tps, # cells, # genes)
    else:
        traj_cell_types = None
    return traj_data_np, traj_data_torch, traj_cell_types



if __name__ == '__main__':
    # Load data
    data_name = "zebrafish"  # zebrafish, mammalian, drosophila, wot
    split_type = "interpolation"
    ann_data, data, cell_tps, cell_types, n_genes, n_tps = loadData(data_name, split_type)
    # # Split by time points
    # traj_data_np, traj_data_torch, traj_cell_types = splitTP(data, cell_tps, cell_types, n_tps)
    # print("# cells={}".format([each.shape[0] for each in traj_data_np]))
    # -----
    scanpy.pp.neighbors(ann_data, n_neighbors=50, n_pcs=None)
    scanpy.tl.umap(ann_data, min_dist=0.5)
    if data_name == "zebrafish":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 8))
        scanpy.pl.umap(ann_data, color="tp", ax=ax1, show=False)
        scanpy.pl.umap(ann_data, color="ZF6S-Cluster", legend_loc="on data", ax=ax2, show=False)
        plt.show()
    elif data_name == "mammalian":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 8))
        scanpy.pl.umap(ann_data, color="tp", ax=ax1, show=False)
        scanpy.pl.umap(ann_data, color="New_cellType", legend_loc="on data", ax=ax2, show=False)
        plt.show()
    elif data_name == "drosophila":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 8))
        scanpy.pl.umap(ann_data, color="tp", ax=ax1, show=False)
        scanpy.pl.umap(ann_data, color="seurat_clusters", legend_loc="on data", ax=ax2, show=False)
        plt.show()
    elif data_name == "wot":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 8))
        scanpy.pl.umap(ann_data, color="tp", ax=ax1, show=False)
        scanpy.pl.umap(ann_data, color="cell_type", legend_loc="on data", ax=ax2, show=False)
        plt.show()

