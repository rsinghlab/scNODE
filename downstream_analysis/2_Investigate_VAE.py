'''
Description:
    Investigate the VAE structure of our model.
'''
import copy

import matplotlib.pyplot as plt
import scanpy
import scipy.interpolate
import torch
import torch.distributions as dist
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import itertools
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import seaborn as sbn
from scipy.optimize import minimize

from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, tunedOurPars
from plotting.visualization import plotUMAP, plotPredAllTime, plotPredTestTime, umapWithoutPCA, umapWithPCA
from data.preprocessing import splitBySpec
from benchmark.Compare_SingleCell_Predictions import basicStats, globalEvaluation
from optim.running import constructscNODEModel, scNODETrainWithPreTrain
from plotting import linearSegmentCMap, _removeAllBorders
from plotting.__init__ import *
import matplotlib.patheffects as pe

# ======================================================

merge_dict = {
    "zebrafish": ["Hematopoeitic", 'Hindbrain', "Notochord", "PSM", "Placode", "Somites", "Spinal", "Neural", "Endoderm"],
}

def mergeCellTypes(cell_types, merge_list):
    new_cell_types = []
    for c in cell_types:
        c_sep = c.strip(" ").split(" ")
        if c in merge_list:
            new_cell_types.append(c)
        elif c_sep[0] in merge_list:
            new_cell_types.append(c_sep[0])
        else:
            new_cell_types.append(c)
    return new_cell_types


latent_dim = 50
drift_latent_size = [50]
enc_latent_list = [50]
dec_latent_list = [50]
act_name = "relu"
def loadModel(data_name, split_type):
    dict_filename = "../res/downstream_analysis/vector_field/{}-{}-latent_ODE_OT_pretrain-state_dict.pt".format(data_name,split_type)
    latent_ode_model = constructscNODEModel(
        n_genes, latent_dim=latent_dim,
        enc_latent_list=enc_latent_list, dec_latent_list=dec_latent_list, drift_latent_size=drift_latent_size,
        latent_enc_act="none", latent_dec_act=act_name, drift_act=act_name,
        ode_method="euler"
    )
    latent_ode_model.load_state_dict(torch.load(dict_filename))
    latent_ode_model.eval()
    return latent_ode_model

# ======================================================

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import spearmanr
def visEncoderWeights(latent_ode_model):
    enc_pars = list(latent_ode_model.latent_encoder.net.parameters())
    first_layer_w = enc_pars[0].detach().numpy().T # gene x dim
    second_layer_w = enc_pars[2].detach().numpy().T # dim x dim
    product_layer_w = first_layer_w @ second_layer_w
    # -----
    linkage_matrix = linkage(product_layer_w, method="ward")
    dend_dict = dendrogram(linkage_matrix, no_plot=True, color_threshold = 1.5)
    gene_labels = dend_dict["leaves_color_list"]
    gene_idx = [int(x) for x in dend_dict["ivl"]]
    color_dict = dict(zip(np.unique(gene_labels), Vivid_10.mpl_colors))
    gene_color = [color_dict[x] for x in gene_labels]
    np.save("../res/downstream_analysis/gene_weight/VAE_weights_clustering.npy", {
        "linkage": linkage_matrix,
        "dendogram": dend_dict
    })
    # -----
    clus_map = sbn.clustermap(
        product_layer_w,
        cmap="RdBu",
        row_cluster=True, row_linkage=linkage_matrix,
        tree_kws={'colors':gene_color},
        col_cluster=False,
        figsize=(12, 8),
    )
    heatmap_axes = clus_map.ax_heatmap
    heatmap_axes.set_xticks([])
    heatmap_axes.set_yticks([])
    heatmap_axes.set_xlabel("Latent")
    heatmap_axes.set_ylabel("Gene")
    heatmap_axes.set_title("Encoder Weights")
    clus_map.figure.subplots_adjust(right=0.8)
    clus_map.figure.subplots_adjust(top=1.0)
    clus_map.ax_cbar.set_position((0.85, .5, .01, .2))
    # clus_map.figure.set_tight_layout(True)
    plt.show()
    # -----
    gene_corr = spearmanr(product_layer_w[gene_idx, :].T)[0]
    thr = 0.5
    gene_corr[np.abs(gene_corr) <= thr] = 0.0
    plt.title("Gene Correlation")
    sbn.heatmap(gene_corr, cmap="RdBu", square=True, vmin=-1.0, vmax=1.0)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()


# ======================================================
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
def compareCellClustering(traj_data, traj_cell_types, latent_ode_model, color_list):
    print("PCA...")
    concat_traj_data = np.concatenate(traj_data, axis=0)
    pca_model = PCA(n_components=latent_dim, svd_solver="arpack")
    pca_model.fit(concat_traj_data)
    pca_last_time = pca_model.transform(traj_data[-1])
    # -----
    print("VAE...")
    latent_mu, latent_std = latent_ode_model.latent_encoder(torch.FloatTensor(traj_data[-1]))
    vae_last_time = latent_ode_model._sampleGaussian(latent_mu, latent_std).detach().numpy()
    # -----
    print("UMAP...")
    pca_last_time_vis = UMAP(n_components=2, min_dist=0.0).fit_transform(pca_last_time)
    vae_last_time_vis = UMAP(n_components=2, min_dist=0.0).fit_transform(vae_last_time)
    # -----
    print("KMeans..")
    cell_type_last_time = traj_cell_types[-1]
    pca_pred = KMeans(n_clusters=len(np.unique(cell_type_last_time))).fit_predict(pca_last_time)
    vae_pred = KMeans(n_clusters=len(np.unique(cell_type_last_time))).fit_predict(vae_last_time)
    pca_ari = adjusted_rand_score(cell_type_last_time, pca_pred)
    vae_ari = adjusted_rand_score(cell_type_last_time, vae_pred)
    pca_ami = adjusted_mutual_info_score(cell_type_last_time, pca_pred)
    vae_ami = adjusted_mutual_info_score(cell_type_last_time, vae_pred)
    # -----
    print("Visualization...")
    fig, ax_list = plt.subplots(1, 2, figsize=(12, 5))
    ax_list[0].set_title("PCA \n (ARI={:.2f}, AMI={:.2f})".format(pca_ari, pca_ami))
    ax_list[1].set_title("Our VAE \n (ARI={:.2f}, AMI={:.2f})".format(vae_ari, vae_ami))
    for i, c in enumerate(np.unique(cell_type_last_time)):
        cell_idx = np.where(cell_type_last_time == c)[0]
        ax_list[0].scatter(pca_last_time_vis[cell_idx, 0], pca_last_time_vis[cell_idx, 1],  cmap="hsv", s=10, alpha=0.9)
        ax_list[1].scatter(vae_last_time_vis[cell_idx, 0], vae_last_time_vis[cell_idx, 1],  cmap="hsv", s=10, alpha=0.9, label=c)
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1, fontsize=12)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    # Load data and pre-processing
    print("=" * 70)
    data_name = "zebrafish"  # zebrafish, mammalian
    print("[ {} ]".format(data_name).center(60))
    split_type = "all"  # all
    print("Split type: {}".format(split_type))
    ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name, split_type)
    train_tps, test_tps = list(range(n_tps)), []
    data = ann_data.X
    if data_name in merge_dict:
        print("Merge cell types...")
        cell_types = np.asarray(mergeCellTypes(cell_types, merge_dict[data_name]))
    # Convert to torch project
    traj_data = [torch.FloatTensor(data[np.where(cell_tps == t)[0], :]) for t in range(1, n_tps + 1)]
    if cell_types is not None:
        traj_cell_types = [cell_types[np.where(cell_tps == t)[0]] for t in range(1, n_tps + 1)]
    else:
        traj_cell_types = None
    all_tps = list(range(n_tps))
    train_data, test_data = splitBySpec(traj_data, train_tps, test_tps)
    tps = torch.FloatTensor(all_tps)
    train_tps = torch.FloatTensor(train_tps)
    test_tps = torch.FloatTensor(test_tps)
    n_cells = [each.shape[0] for each in traj_data]
    print("# tps={}, # genes={}".format(n_tps, n_genes))
    print("# cells={}".format(n_cells))
    print("Train tps={}".format(train_tps))
    print("Test tps={}".format(test_tps))
    # =======================
    latent_ode_model = loadModel(data_name, split_type)
    data_res = np.load(
        "../res/downstream_analysis/vector_field/{}-{}-latent_ODE_OT_pretrain-aux_data.npy".format(data_name,
                                                                                                   split_type),
        allow_pickle=True
    ).item()
    latent_seq = data_res["latent_seq"]
    next_seq = data_res["next_seq"]
    drift_seq = data_res["drift_seq"]
    drift_magnitude = data_res["drift_magnitude"]
    umap_latent_data = data_res["umap_latent_data"]
    umap_next_data = data_res["umap_next_data"]
    umap_model = data_res["umap_model"]  # Note: The UMAP is for the model latent space
    latent_tp_list = data_res["latent_tp_list"]
    # -----
    # visEncoderWeights(latent_ode_model)
    # compareCellClustering(
    #     [each.detach().numpy() for each in traj_data],
    #     traj_cell_types, latent_ode_model,
    #     color_list = Bold_10.mpl_colors
    # )

    # -----
    # gene_weight = np.load("../res/downstream_analysis/gene_weight/VAE_weights_clustering.npy", allow_pickle=True).item()
    # dend_dict = gene_weight["dendogram"]
    # gene_labels = np.asarray(dend_dict["leaves_color_list"])
    # gene_idx = [int(x) for x in dend_dict["ivl"]]
    # gene_names = ann_data.var_names.values
    # with open("../res/downstream_analysis/gene_weight/gene_cluster_list.txt", "w") as file:
    #     for g_l in ["C1", "C2", "C3", "C4", "C5", "C6"]:
    #         label_gene_names = list(gene_names[gene_idx][np.where(gene_labels == g_l)[0]])
    #         print("{}: {}".format(g_l, len(label_gene_names)))
    #         file.write("-" * 50)
    #         file.write("\n")
    #         file.write("\n".join(label_gene_names))
    #         file.write("\n")
    # -----
    g_l = "C1" # "C1", "C2", "C3", "C4", "C5", "C6"
    go_table = pd.read_csv("../res/downstream_analysis/gene_weight/C1-analysis.txt", skiprows=11, header=0, index_col=0, sep="\t")
    print()
