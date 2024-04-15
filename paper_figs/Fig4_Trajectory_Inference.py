'''
Description:
    Compare model predictions in recovering cell trajectories.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import torch
import numpy as np
from plotting import linearSegmentCMap, _removeAllBorders
from plotting.__init__ import *
from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, tunedOurPars, splitBySpec
import matplotlib.patheffects as pe


def visTraj():
    n_neighbors = 50
    min_dist = 0.25
    pca_pcs = 50
    if data_name == "wot":
        n_neighbors = 50
        min_dist = 0.8
    # -----
    print("Loading embedding ...")
    res_filename = "../res/trajectory/{}-res-embedding-neighbor{}-min_dist{:.2f}-pcs{}-project_to_all-{}-comparison.npy".format(data_name, n_neighbors, min_dist, pca_pcs, split_type)
    res_dict = np.load(res_filename, allow_pickle=True).item()
    scNODE_use_tps = res_dict["use_tps"]
    pred_tps = res_dict["pred_tps"]
    all_cell_tps = res_dict["all_cell_tps"]
    use_cell_tps = res_dict["use_cell_tps"]
    scNODE_cell_tps = res_dict["scNODE_cell_tps"]
    PRESCIENT_cell_tps = res_dict["PRESCIENT_cell_tps"]
    PRESCIENT_cell_tps = PRESCIENT_cell_tps / 10.0
    MIOFlow_cell_tps = res_dict["MIOFlow_cell_tps"]
    all_umap_traj = res_dict["all_umap_traj"]
    use_umap_traj = res_dict["use_umap_traj"]
    scNODE_umap_traj = res_dict["scNODE_umap_traj"]
    PRESCIENT_umap_traj = res_dict["PRESCIENT_umap_traj"]
    MIOFlow_umap_traj = res_dict["MIOFlow_umap_traj"]
    # =====
    graph_dict = np.load("../res/trajectory/{}-PAGA_graph-{}-comparison.npy".format(data_name, split_type),allow_pickle=True).item()
    all_node_pos = graph_dict["all_node_pos"]
    all_edge = graph_dict["all_edge"]

    use_node_pos = graph_dict["use_node_pos"]
    use_edge = graph_dict["use_edge"]

    scNODE_node_pos = graph_dict["scNODE_node_pos"]
    scNODE_edge = graph_dict["scNODE_edge"]

    PRESCIENT_node_pos = graph_dict["PRESCIENT_node_pos"]
    PRESCIENT_edge = graph_dict["PRESCIENT_edge"]

    MIOFlow_node_pos = graph_dict["MIOFlow_node_pos"]
    MIOFlow_edge = graph_dict["MIOFlow_edge"]

    ################################
    bg_s_size = 10
    s_size = 1.0
    node_s_size = 5
    lw = 1.0
    title_fontsize = 17
    if data_name == "zebrafish":
        im_list = ["0.200", "0.093", "0.113", "0.107"]
    if data_name == "drosophila":
        im_list = ["0.144", "0.114", "0.152", "0.126"]
    if data_name == "wot":
        im_list = ["0.137", "0.106", "0.119", "0.150"]
    color_list = linearSegmentCMap(len(all_tps), "viridis")
    fig, ax_list = plt.subplots(1, 5, figsize=(14, 2.5))
    ax_list[0].set_title("Original \n $\\mathcal{G}_{\\mathrm{true}}$", fontsize=title_fontsize)
    for t_idx, t in enumerate(all_tps):
        cell_idx = np.where(all_cell_tps == t)[0]
        ax_list[0].scatter(
            all_umap_traj[cell_idx, 0], all_umap_traj[cell_idx, 1],
            color=color_list[t_idx], s=s_size, alpha=0.4
        )
    ax_list[0].scatter(all_node_pos[:, 0], all_node_pos[:, 1], s=node_s_size, color="k", alpha=0.8)
    for e in all_edge:
        ax_list[0].plot([all_node_pos[e[0]][0], all_node_pos[e[1]][0]], [all_node_pos[e[0]][1], all_node_pos[e[1]][1]], "k-", lw=lw)
    #####
    ax_list[1].set_title("After Removal \n $\\mathcal{G}_{\\mathrm{removal}}$ IM="+im_list[0], fontsize=title_fontsize)
    for t_idx, t in enumerate(scNODE_use_tps):
        cell_idx = np.where(use_cell_tps == t)[0]
        sc1 = ax_list[1].scatter(
            use_umap_traj[cell_idx, 0], use_umap_traj[cell_idx, 1],
            color=color_list[list(all_tps).index(t)], s=5, alpha=0.4
        )
    ax_list[1].scatter(use_node_pos[:, 0], use_node_pos[:, 1], s=node_s_size, color="k", alpha=0.8)
    for e in use_edge:
        ax_list[1].plot([use_node_pos[e[0]][0], use_node_pos[e[1]][0]], [use_node_pos[e[0]][1], use_node_pos[e[1]][1]], "k-", lw=lw)
    #####
    ax_list[2].set_title("scNODE Prediction \n $\\mathcal{G}_{\\mathrm{scNODE}}$ IM="+im_list[1], fontsize=title_fontsize)
    if data_name == "zebrafish":
        scNODE_cell_tps[(scNODE_cell_tps >= 2) & (scNODE_cell_tps < 3)] = 2
        scNODE_cell_tps[(scNODE_cell_tps >= 4) & (scNODE_cell_tps < 5)] = 4
        scNODE_cell_tps[(scNODE_cell_tps >= 6) & (scNODE_cell_tps < 7)] = 6
        scNODE_cell_tps[(scNODE_cell_tps >= 8) & (scNODE_cell_tps < 9)] = 8
    if data_name == "drosophila":
        scNODE_cell_tps[(scNODE_cell_tps >= 2) & (scNODE_cell_tps < 3)] = 2
        scNODE_cell_tps[(scNODE_cell_tps >= 4) & (scNODE_cell_tps < 5)] = 4
        scNODE_cell_tps[(scNODE_cell_tps >= 6) & (scNODE_cell_tps < 7)] = 6
        scNODE_cell_tps[(scNODE_cell_tps >= 8) & (scNODE_cell_tps < 9)] = 8
    if data_name == "wot":
        scNODE_cell_tps[(scNODE_cell_tps >= 5) & (scNODE_cell_tps < 6)] = 5
        scNODE_cell_tps[(scNODE_cell_tps >= 7) & (scNODE_cell_tps < 8)] = 7
        scNODE_cell_tps[(scNODE_cell_tps >= 9) & (scNODE_cell_tps < 10)] = 9
        scNODE_cell_tps[(scNODE_cell_tps >= 11) & (scNODE_cell_tps < 12)] = 11

    for t_idx, t in enumerate(all_tps):
        cell_idx = np.where(scNODE_cell_tps == t)[0]
        sc2 =ax_list[2].scatter(
            scNODE_umap_traj[cell_idx, 0], scNODE_umap_traj[cell_idx, 1],
            color=color_list[t_idx], s=s_size, alpha=0.4
        )
    ax_list[2].scatter(scNODE_node_pos[:, 0], scNODE_node_pos[:, 1], s=node_s_size, color="k", alpha=0.8)
    for e in scNODE_edge:
        ax_list[2].plot([scNODE_node_pos[e[0]][0], scNODE_node_pos[e[1]][0]], [scNODE_node_pos[e[0]][1], scNODE_node_pos[e[1]][1]], "k-", lw=lw)
    #####
    ax_list[3].set_title("MIOFlow Prediction \n $\\mathcal{G}_{\\mathrm{MIOFlow}}$ IM="+im_list[2], fontsize=title_fontsize)
    if data_name == "zebrafish":
        MIOFlow_cell_tps[(MIOFlow_cell_tps >= 2) & (MIOFlow_cell_tps < 3)] = 2
        MIOFlow_cell_tps[(MIOFlow_cell_tps >= 4) & (MIOFlow_cell_tps < 5)] = 4
        MIOFlow_cell_tps[(MIOFlow_cell_tps >= 6) & (MIOFlow_cell_tps < 7)] = 6
        MIOFlow_cell_tps[(MIOFlow_cell_tps >= 8) & (MIOFlow_cell_tps < 9)] = 8
    if data_name == "drosophila":
        MIOFlow_cell_tps[(MIOFlow_cell_tps >= 2) & (MIOFlow_cell_tps < 3)] = 2
        MIOFlow_cell_tps[(MIOFlow_cell_tps >= 4) & (MIOFlow_cell_tps < 5)] = 4
        MIOFlow_cell_tps[(MIOFlow_cell_tps >= 6) & (MIOFlow_cell_tps < 7)] = 6
        MIOFlow_cell_tps[(MIOFlow_cell_tps >= 8) & (MIOFlow_cell_tps < 9)] = 8
    if data_name == "wot":
        MIOFlow_cell_tps[(MIOFlow_cell_tps >= 5) & (MIOFlow_cell_tps < 6)] = 5
        MIOFlow_cell_tps[(MIOFlow_cell_tps >= 7) & (MIOFlow_cell_tps < 8)] = 7
        MIOFlow_cell_tps[(MIOFlow_cell_tps >= 9) & (MIOFlow_cell_tps < 10)] = 9
        MIOFlow_cell_tps[(MIOFlow_cell_tps >= 11) & (MIOFlow_cell_tps < 12)] = 11
    for t_idx, t in enumerate(all_tps):
        cell_idx = np.where(MIOFlow_cell_tps == t)[0]
        sc3 = ax_list[3].scatter(
            MIOFlow_umap_traj[cell_idx, 0], MIOFlow_umap_traj[cell_idx, 1],
            color=color_list[t_idx], s=s_size, alpha=0.4
        )
    ax_list[3].scatter(MIOFlow_node_pos[:, 0], MIOFlow_node_pos[:, 1], s=node_s_size, color="k", alpha=0.8)
    for e in MIOFlow_edge:
        ax_list[3].plot([MIOFlow_node_pos[e[0]][0], MIOFlow_node_pos[e[1]][0]],
                        [MIOFlow_node_pos[e[0]][1], MIOFlow_node_pos[e[1]][1]], "k-", lw=lw)
    #####
    ax_list[4].set_title("PRESCIENT Prediction \n $\\mathcal{G}_{\\mathrm{PRESCIENT}}$ IM="+im_list[3], fontsize=title_fontsize)
    if data_name == "zebrafish":
        PRESCIENT_cell_tps[(PRESCIENT_cell_tps >= 2) & (PRESCIENT_cell_tps < 3)] = 2
        PRESCIENT_cell_tps[(PRESCIENT_cell_tps >= 4) & (PRESCIENT_cell_tps < 5)] = 4
        PRESCIENT_cell_tps[(PRESCIENT_cell_tps >= 6) & (PRESCIENT_cell_tps < 7)] = 6
        PRESCIENT_cell_tps[(PRESCIENT_cell_tps >= 8) & (PRESCIENT_cell_tps < 9)] = 8
    if data_name == "drosophila":
        PRESCIENT_cell_tps[(PRESCIENT_cell_tps >= 2) & (PRESCIENT_cell_tps < 3)] = 2
        PRESCIENT_cell_tps[(PRESCIENT_cell_tps >= 4) & (PRESCIENT_cell_tps < 5)] = 4
        PRESCIENT_cell_tps[(PRESCIENT_cell_tps >= 6) & (PRESCIENT_cell_tps < 7)] = 6
        PRESCIENT_cell_tps[(PRESCIENT_cell_tps >= 8) & (PRESCIENT_cell_tps < 9)] = 8
    if data_name == "wot":
        PRESCIENT_cell_tps[(PRESCIENT_cell_tps >= 5) & (PRESCIENT_cell_tps < 6)] = 5
        PRESCIENT_cell_tps[(PRESCIENT_cell_tps >= 7) & (PRESCIENT_cell_tps < 8)] = 7
        PRESCIENT_cell_tps[(PRESCIENT_cell_tps >= 9) & (PRESCIENT_cell_tps < 10)] = 9
        PRESCIENT_cell_tps[(PRESCIENT_cell_tps >= 11) & (PRESCIENT_cell_tps < 12)] = 11
    for t_idx, t in enumerate(all_tps):
        cell_idx = np.where(PRESCIENT_cell_tps == t)[0]
        sc4 = ax_list[4].scatter(
            PRESCIENT_umap_traj[cell_idx, 0], PRESCIENT_umap_traj[cell_idx, 1],
            color=color_list[t_idx], s=s_size, alpha=0.4
        )
    ax_list[4].scatter(PRESCIENT_node_pos[:, 0], PRESCIENT_node_pos[:, 1], s=node_s_size, color="k", alpha=0.8)
    for e in PRESCIENT_edge:
        ax_list[4].plot([PRESCIENT_node_pos[e[0]][0], PRESCIENT_node_pos[e[1]][0]],
                        [PRESCIENT_node_pos[e[0]][1], PRESCIENT_node_pos[e[1]][1]], "k-", lw=lw)
    #####
    for ax in ax_list:
        ax.set_xticks([])
        ax.set_yticks([])
        _removeAllBorders(ax)
    plt.tight_layout()
    plt.show()
    # -----
    for t in range(len(all_tps)):
        plt.scatter([], [], label=t, color=color_list[t], s=30, alpha=1.0)
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="TPs", title_fontsize=14, fontsize=11, ncol=2 if data_name=="wot" else 1)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    # Load data and pre-processing
    print("=" * 70)
    data_name = "zebrafish"  # zebrafish, drosophila, wot
    print("[ {} ]".format(data_name).center(60))
    split_type = "remove_recovery"
    print("Split type: {}".format(split_type))
    # Load data and pre-processing
    print("=" * 70)
    ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name, split_type)
    train_tps, test_tps = list(range(n_tps)), []
    data = ann_data.X
    # Convert to torch project
    traj_data = [data[np.where(cell_tps == t)[0], :] for t in range(1, n_tps + 1)]
    if cell_types is not None:
        traj_cell_types = [cell_types[np.where(cell_tps == t)[0]] for t in range(1, n_tps + 1)]
    else:
        traj_cell_types = None
    all_tps = list(range(n_tps))
    tps = torch.FloatTensor(all_tps)
    n_cells = [each.shape[0] for each in traj_data]
    print("# tps={}, # genes={}".format(n_tps, n_genes))
    print("# cells={}".format(n_cells))
    # # =======================
    visTraj()



