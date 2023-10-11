'''
Description:
    Augmenting data to improve age prediction.
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
import elpigraph

from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, tunedOurPars
from plotting.visualization import plotUMAP, plotUMAPTimePoint, plotUMAPTestTime, umapWithoutPCA, umapWithPCA
from data.preprocessing import splitBySpec
from benchmark.Compare_SingleCell_Predictions import basicStats, globalEvaluation
from optim.running import constructLatentODEModel, latentODETrainWithPreTrain, latentODESimulate
from plotting.utils import linearSegmentCMap, _removeAllBorders
from plotting.__init__ import *
import matplotlib.patheffects as pe

# ======================================================

latent_dim = 50
drift_latent_size = [50]
enc_latent_list = [50]
dec_latent_list = [50]
act_name = "relu"

def augmentation(train_data, train_tps, tps, n_sim_cells):
    pretrain_iters = 200
    pretrain_lr = 1e-3
    latent_coeff = 1.0
    epochs = 10
    iters = 100
    batch_size = 32
    lr = 1e-3
    latent_ode_model = constructLatentODEModel(
        n_genes, latent_dim=latent_dim,
        enc_latent_list=enc_latent_list, dec_latent_list=dec_latent_list, drift_latent_size=drift_latent_size,
        latent_enc_act="none", latent_dec_act=act_name, drift_act=act_name,
        ode_method="euler"
    )
    latent_ode_model, loss_list, recon_obs, first_latent_dist, latent_seq = latentODETrainWithPreTrain(
        train_data, train_tps, latent_ode_model, latent_coeff=latent_coeff,
        epochs=epochs, iters=iters, batch_size=batch_size, lr=lr,
        pretrain_iters=pretrain_iters, pretrain_lr=pretrain_lr, only_train_de=False
    )
    all_recon_obs = latentODESimulate(latent_ode_model, first_latent_dist, tps, n_cells=n_sim_cells)  # (# trajs, # tps, # genes)
    return latent_ode_model, all_recon_obs


def saveModel(latent_ode_model, data_name, split_type):
    dict_filename = "../res/downstream_analysis/vector_field/{}-{}-latent_ODE_OT_pretrain-state_dict.pt".format(data_name,split_type)
    torch.save(latent_ode_model.state_dict(), dict_filename)


def loadModel(state_dict_name):
    latent_ode_model = constructLatentODEModel(
        n_genes, latent_dim=latent_dim,
        enc_latent_list=enc_latent_list, dec_latent_list=dec_latent_list, drift_latent_size=drift_latent_size,
        latent_enc_act="none", latent_dec_act=act_name, drift_act=act_name,
        ode_method="euler"
    )
    latent_ode_model.load_state_dict(torch.load(state_dict_name))
    latent_ode_model.eval()
    return latent_ode_model

# ======================================================

def recoverTraj(traj_data, traj_cell_types):
    n_neighbors = 50
    min_dist = 0.1
    pca_pcs = 50
    # -----
    # Original data
    print("All tps...")
    all_tps = list(range(len(traj_data)))
    all_cell_tps = np.concatenate([np.repeat(all_tps[idx], x.shape[0]) for idx, x in enumerate(traj_data)])
    all_umap_traj, all_umap_model, all_pca_model = umapWithPCA(
        np.concatenate(traj_data, axis=0), n_neighbors=n_neighbors, min_dist=min_dist, pca_pcs=pca_pcs
    )
    # -----
    # Remove several timepoints
    use_tps = [0, 1, 2, 3, 5, 7, 9, 10, 11]
    print("Used tps {}...".format(use_tps))
    use_traj_data = [traj_data[i] for i in use_tps]
    use_cell_tps = np.concatenate([np.repeat(use_tps[idx], x.shape[0]) for idx, x in enumerate(use_traj_data)])
    use_umap_traj, use_umap_model, use_pca_model = umapWithPCA(
        np.concatenate(use_traj_data, axis=0), n_neighbors=n_neighbors, min_dist=min_dist, pca_pcs=pca_pcs
    )
    # -----
    # Augment removed timepoints
    print("Augment...")
    train_data = [torch.FloatTensor(traj_data[i]) for i in use_tps]
    train_tps = torch.FloatTensor(use_tps)
    # pred_tps = torch.FloatTensor([i for i in all_tps if i not in use_tps])
    # pred_tps = tps
    pred_tps = torch.FloatTensor([0, 1, 2, 3, 4, 4.25, 4.5, 4.75, 5, 6, 6.25, 6.5, 6.75, 7, 8, 8.25, 8.5, 8.75, 9, 10, 11])
    latent_ode_model, all_recon_obs = augmentation(train_data, train_tps, pred_tps, n_sim_cells=300)
    # aug_traj_data = copy.deepcopy(traj_data)
    # for idx, i in enumerate([4, 6, 8]):
    #     pred_idx = list(pred_tps.detach().numpy()).index(i)
    #     aug_traj_data[int(i)] = all_recon_obs[:, pred_idx, :]
    # aug_cell_tps = np.concatenate([np.repeat(all_tps[idx], x.shape[0]) for idx, x in enumerate(aug_traj_data)])
    # aug_umap_traj, aug_umap_model, aug_pca_model = umapWithPCA(
    #     np.concatenate(aug_traj_data, axis=0), n_neighbors=n_neighbors, min_dist=min_dist, pca_pcs=pca_pcs
    # )
    # aug_umap_traj = use_umap_model.transform(use_pca_model.transform(np.concatenate(aug_traj_data, axis=0)))
    # -----
    # Predictions
    all_recon_obs = [all_recon_obs[:, t, :] for t in range(all_recon_obs.shape[1])]
    # pred_cell_tps = np.concatenate([np.repeat(all_tps[idx], x.shape[0]) for idx, x in enumerate(all_recon_obs)])
    pred_umap_traj, pred_umap_model, pred_pca_model = umapWithPCA(
        np.concatenate(all_recon_obs, axis=0), n_neighbors=n_neighbors, min_dist=min_dist, pca_pcs=pca_pcs
    )
    # -----
    res_filename = "../res/downstream_analysis/recover_traj/zebrafish-res.npy"
    state_filename = "../res/downstream_analysis/recover_traj/zebrafish-latent_ODE_OT-state_dict.pt"
    print("Saving to {}".format(res_filename))
    np.save(
        res_filename,
        {"true": traj_data,
         "pred": all_recon_obs,
         # "aug": aug_traj_data,

         "use_tps": use_tps,
         "pred_tps": list(pred_tps.detach().numpy()),

         "all_umap_traj": all_umap_traj,
         "all_umap_model": all_umap_model,
         "all_pca_model": all_pca_model,

         "use_umap_traj": use_umap_traj,
         "use_umap_model": use_umap_model,
         "use_pca_model": use_pca_model,

         # "aug_umap_traj": aug_umap_traj,
         # "aug_umap_model": aug_umap_model,
         # "aug_pca_model": aug_pca_model,

         "pred_umap_traj": pred_umap_traj,
         "pred_umap_model": pred_umap_model,
         "pred_pca_model": pred_pca_model,
         },
        allow_pickle=True
    )
    torch.save(latent_ode_model.state_dict(), state_filename)


def computeEmbedding():
    res_filename = "../res/downstream_analysis/recover_traj/zebrafish-res.npy"
    state_filename = "../res/downstream_analysis/recover_traj/zebrafish-latent_ODE_OT-state_dict.pt"
    # -----
    latent_ode_model = loadModel(state_filename)
    res = np.load(res_filename, allow_pickle=True).item()
    traj_data = res["true"]
    all_recon_obs = res["pred"]
    use_tps = res["use_tps"]
    pred_tps = res["pred_tps"]
    use_traj_data = [traj_data[i] for i in use_tps]
    aug_traj_data = []
    for i, x in enumerate(all_recon_obs):
        if pred_tps[i] not in use_tps:
            aug_traj_data.append(x)
        else:
            aug_traj_data.append(traj_data[int(pred_tps[i])])
    # all_umap_traj = res["all_umap_traj"]
    # all_umap_model = res["all_umap_model"]
    # all_pca_model = res["all_pca_model"]
    # use_umap_traj = res["use_umap_traj"]
    # use_umap_model = res["use_umap_model"]
    # use_pca_model = res["use_pca_model"]
    # pred_umap_traj = res["pred_umap_traj"]
    # pred_umap_model = res["pred_umap_model"]
    # pred_pca_model = res["pred_pca_model"]
    # -----
    all_cell_tps = np.concatenate([np.repeat(all_tps[idx], x.shape[0]) for idx, x in enumerate(traj_data)])
    use_cell_tps = np.concatenate([np.repeat(use_tps[idx], x.shape[0]) for idx, x in enumerate(use_traj_data)])
    aug_cell_tps = np.concatenate([np.repeat(pred_tps[idx], x.shape[0]) for idx, x in enumerate(aug_traj_data)])
    pred_cell_tps = np.concatenate([np.repeat(pred_tps[idx], x.shape[0]) for idx, x in enumerate(all_recon_obs)])
    # -----
    n_neighbors = 50
    min_dist = 0.25
    pca_pcs = 50
    print("All")
    all_umap_traj, all_umap_model, all_pca_model = umapWithPCA(
        np.concatenate(traj_data, axis=0), n_neighbors=n_neighbors, min_dist=min_dist, pca_pcs=pca_pcs
    )
    print("Used")
    use_umap_traj, use_umap_model, use_pca_model = umapWithPCA(
        np.concatenate(use_traj_data, axis=0), n_neighbors=n_neighbors, min_dist=min_dist, pca_pcs=pca_pcs
    )
    print("Aug")
    aug_umap_traj, aug_umap_model, aug_pca_model = umapWithPCA(
        np.concatenate(aug_traj_data, axis=0), n_neighbors=n_neighbors, min_dist=min_dist, pca_pcs=pca_pcs
    )
    print("Pred")
    pred_umap_traj, pred_umap_model, pred_pca_model = umapWithPCA(
        np.concatenate(all_recon_obs, axis=0), n_neighbors=n_neighbors, min_dist=min_dist, pca_pcs=pca_pcs
    )
    # -----
    np.save(
        "../res/downstream_analysis/recover_traj/zebrafish-res-embedding-neighbor{}-min_dist{:.2f}-pcs{}.npy".format(n_neighbors, min_dist, pca_pcs),
        {
            "use_tps": use_tps,
            "pred_tps": pred_tps,

            "all_cell_tps": all_cell_tps,
            "use_cell_tps": use_cell_tps,
            "aug_cell_tps": aug_cell_tps,
            "pred_cell_tps": pred_cell_tps,

            "all_umap_traj": all_umap_traj,
            "all_umap_model": all_umap_model,
            "all_pca_model": all_pca_model,

            "use_umap_traj": use_umap_traj,
            "use_umap_model": use_umap_model,
            "use_pca_model": use_pca_model,

            "aug_umap_traj": aug_umap_traj,
            "aug_umap_model": aug_umap_model,
            "aug_pca_model": aug_pca_model,

            "pred_umap_traj": pred_umap_traj,
            "pred_umap_model": pred_umap_model,
            "pred_pca_model": pred_pca_model,
        }
    )



def visTraj():
    print("Loading embedding...")
    n_neighbors = 50
    min_dist = 0.25
    pca_pcs = 50
    res_filename = "../res/downstream_analysis/recover_traj/zebrafish-res-embedding-neighbor{}-min_dist{:.2f}-pcs{}.npy".format(
        n_neighbors, min_dist, pca_pcs
    )
    state_filename = "../res/downstream_analysis/recover_traj/zebrafish-latent_ODE_OT-state_dict.pt"
    res_dict = np.load(res_filename, allow_pickle=True).item()
    use_tps = res_dict["use_tps"]
    pred_tps = res_dict["pred_tps"]
    all_cell_tps = res_dict["all_cell_tps"]
    use_cell_tps = res_dict["use_cell_tps"]
    aug_cell_tps = res_dict["aug_cell_tps"]
    pred_cell_tps = res_dict["pred_cell_tps"]

    all_umap_traj = res_dict["all_umap_traj"]
    all_umap_model = res_dict["all_umap_model"]
    all_pca_model = res_dict["all_pca_model"]

    use_umap_traj = res_dict["use_umap_traj"]
    use_umap_model = res_dict["use_umap_model"]
    use_pca_model = res_dict["use_pca_model"]

    aug_umap_traj = res_dict["aug_umap_traj"]
    aug_umap_model = res_dict["aug_umap_model"]
    aug_pca_model = res_dict["aug_pca_model"]

    pred_umap_traj = res_dict["pred_umap_traj"]
    pred_umap_model = res_dict["pred_umap_model"]
    pred_pca_model = res_dict["pred_pca_model"]
    # -----
    # X = all_umap_traj
    # pg_tree = elpigraph.computeElasticPrincipalTree(X, NumNodes=50)[0]
    # elpigraph.plot.PlotPG(X, pg_tree, Do_PCA=False, show_text=False)
    # plt.show()
    # -----
    all_ann_data = scanpy.AnnData(all_umap_traj)
    all_ann_data.obs["time"] = all_cell_tps
    use_ann_data = scanpy.AnnData(use_umap_traj)
    use_ann_data.obs["time"] = use_cell_tps
    aug_ann_data = scanpy.AnnData(aug_umap_traj)
    aug_ann_data.obs["time"] = aug_cell_tps
    print("Neighbors")
    scanpy.pp.neighbors(all_ann_data, n_neighbors=5)
    scanpy.pp.neighbors(use_ann_data, n_neighbors=5)
    scanpy.pp.neighbors(aug_ann_data, n_neighbors=5)
    print("Louvain")
    scanpy.tl.louvain(all_ann_data, resolution=0.5)
    scanpy.tl.louvain(use_ann_data, resolution=0.5)
    scanpy.tl.louvain(aug_ann_data, resolution=0.5)
    print("PAGA")
    scanpy.tl.paga(all_ann_data, groups='louvain')
    scanpy.tl.paga(use_ann_data, groups='louvain')
    scanpy.tl.paga(aug_ann_data, groups='louvain')

    thr = 0.1
    all_conn = all_ann_data.uns["paga"]["connectivities"].todense()
    all_conn[np.tril_indices_from(all_conn)] = 0
    all_conn[all_conn < thr] = 0
    all_cell_types = all_ann_data.obs.louvain.values
    all_node_pos = [np.mean(all_umap_traj[np.where(all_cell_types == str(c))[0], :], axis=0) for c in np.arange(len(np.unique(all_cell_types)))]
    all_node_pos = np.asarray(all_node_pos)
    all_edge = np.asarray(np.where(all_conn != 0)).T

    use_conn = use_ann_data.uns["paga"]["connectivities"].todense()
    use_conn[np.tril_indices_from(use_conn)] = 0
    use_conn[use_conn < thr] = 0
    use_cell_types = use_ann_data.obs.louvain.values
    use_node_pos = [np.mean(use_umap_traj[np.where(use_cell_types == str(c))[0], :], axis=0) for c in
                    np.arange(len(np.unique(use_cell_types)))]
    use_node_pos = np.asarray(use_node_pos)
    use_edge = np.asarray(np.where(use_conn != 0)).T

    aug_conn = aug_ann_data.uns["paga"]["connectivities"].todense()
    aug_conn[np.tril_indices_from(aug_conn)] = 0
    aug_conn[aug_conn < thr] = 0
    aug_cell_types = aug_ann_data.obs.louvain.values
    aug_node_pos = [np.mean(aug_umap_traj[np.where(aug_cell_types == str(c))[0], :], axis=0) for c in
                    np.arange(len(np.unique(aug_cell_types)))]
    aug_node_pos = np.asarray(aug_node_pos)
    aug_edge = np.asarray(np.where(aug_conn != 0)).T
    # -----
    color_list = linearSegmentCMap(len(all_tps), "viridis")
    fig, ax_list = plt.subplots(1, 3, figsize=(11, 4))
    ax_list[0].set_title("Original")
    for t_idx, t in enumerate(all_tps):
        cell_idx = np.where(all_cell_tps == t)[0]
        ax_list[0].scatter(
            all_umap_traj[cell_idx, 0], all_umap_traj[cell_idx, 1],
            color=color_list[t_idx], s=5, alpha=0.4
        )
    ax_list[0].scatter(all_node_pos[:, 0], all_node_pos[:, 1], s=30, color="k", alpha=0.8)
    for e in all_edge:
        ax_list[0].plot([all_node_pos[e[0]][0], all_node_pos[e[1]][0]], [all_node_pos[e[0]][1], all_node_pos[e[1]][1]], "k-", lw=1.5)

    ax_list[1].set_title("After Removal")
    for t_idx, t in enumerate(use_tps):
        cell_idx = np.where(use_cell_tps == t)[0]
        sc1 = ax_list[1].scatter(
            use_umap_traj[cell_idx, 0], use_umap_traj[cell_idx, 1],
            color=color_list[list(all_tps).index(t)], s=5, alpha=0.4
        )
    ax_list[1].scatter(use_node_pos[:, 0], use_node_pos[:, 1], s=30, color="k", alpha=0.8)
    for e in use_edge:
        ax_list[1].plot([use_node_pos[e[0]][0], use_node_pos[e[1]][0]], [use_node_pos[e[0]][1], use_node_pos[e[1]][1]], "k-", lw=1.5)

    ax_list[2].set_title("W/ Prediction")
    aug_cell_tps[aug_cell_tps == 4.25] = 4
    aug_cell_tps[aug_cell_tps == 4.5] = 4
    aug_cell_tps[aug_cell_tps == 4.75] = 4
    aug_cell_tps[aug_cell_tps == 6.25] = 6
    aug_cell_tps[aug_cell_tps == 6.5] = 6
    aug_cell_tps[aug_cell_tps == 6.75] = 6
    aug_cell_tps[aug_cell_tps == 8.25] = 8
    aug_cell_tps[aug_cell_tps == 8.5] = 8
    aug_cell_tps[aug_cell_tps == 8.75] = 8
    # for t_idx, t in enumerate(pred_tps):
    for t_idx, t in enumerate(all_tps):
        cell_idx = np.where(aug_cell_tps == t)[0]
        sc2 =ax_list[2].scatter(
            aug_umap_traj[cell_idx, 0], aug_umap_traj[cell_idx, 1],
            color=color_list[t_idx], s=5, alpha=0.4
        )
    ax_list[2].scatter(aug_node_pos[:, 0], aug_node_pos[:, 1], s=30, color="k", alpha=0.8)
    for e in aug_edge:
        ax_list[2].plot([aug_node_pos[e[0]][0], aug_node_pos[e[1]][0]], [aug_node_pos[e[0]][1], aug_node_pos[e[1]][1]], "k-", lw=1.5)

    for ax in ax_list:
        ax.set_xticks([])
        ax.set_yticks([])
        _removeAllBorders(ax)

    for t in range(len(all_tps)):
        ax_list[-1].scatter([], [], label=t, color=color_list[t], s=30, alpha=1.0)
    ax_list[-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="TPs", title_fontsize=14, fontsize=11)
    # cbar_ax = fig.add_axes([0.95, 0.1, 0.03, 0.8])
    # bounds = np.arange(len(all_tps))
    # mpl.colorbar.ColorbarBase(
    #     cbar_ax, cmap=plt.get_cmap('viridis', len(all_tps)), norm=mpl.colors.BoundaryNorm(bounds, len(all_tps)),
    #     spacing='proportional', ticks=bounds+0.5, boundaries=bounds, format='%1i'
    # )
    plt.tight_layout()
    plt.savefig("../res/figs/zebrafish_recover_traj.pdf")
    plt.show()
    # # -----
    # # color_list = BlueRed_12.mpl_colors
    # # color_list = linearSegmentCMap(len(pred_tps), "viridis")
    # fig, ax_list = plt.subplots(1, 4, figsize=(16, 4))
    # ax_list[0].set_title("All Data", fontsize=15)
    # ax_list[1].set_title("Partial Data", fontsize=15)
    # ax_list[2].set_title("Augmented Data", fontsize=15)
    # ax_list[3].set_title("Prediction", fontsize=15)
    # for t_idx, t in enumerate(all_tps):
    #     cell_idx = np.where(all_cell_tps == t)[0]
    #     ax_list[0].scatter(
    #         all_umap_traj[cell_idx, 0], all_umap_traj[cell_idx, 1],
    #         label=int(t), color=linearSegmentCMap(len(all_tps), "viridis")[t_idx], s=10, alpha=1.0
    #     )
    # for t_idx, t in enumerate(use_tps):
    #     cell_idx = np.where(use_cell_tps == t)[0]
    #     ax_list[1].scatter(
    #         use_umap_traj[cell_idx, 0], use_umap_traj[cell_idx, 1],
    #         label=int(t), color=linearSegmentCMap(len(use_tps), "viridis")[t_idx], s=10, alpha=1.0
    #     )
    # for t_idx, t in enumerate(pred_tps):
    #     cell_idx = np.where(aug_cell_tps == t)[0]
    #     ax_list[2].scatter(
    #         aug_umap_traj[cell_idx, 0], aug_umap_traj[cell_idx, 1],
    #         label=t, color=linearSegmentCMap(len(pred_tps), "viridis")[t_idx], s=10, alpha=1.0
    #     )
    # for t_idx, t in enumerate(pred_tps):
    #     cell_idx = np.where(pred_cell_tps == t)[0]
    #     ax_list[3].scatter(
    #         pred_umap_traj[cell_idx, 0], pred_umap_traj[cell_idx, 1],
    #         label=t, color=linearSegmentCMap(len(pred_tps), "viridis")[t_idx], s=10, alpha=1.0
    #     )
    # for ax in ax_list:
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     _removeAllBorders(ax)
    # ax_list[-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="TPs", title_fontsize=14, fontsize=13)
    # plt.tight_layout()
    # plt.show()



if __name__ == '__main__':
    # Load data and pre-processing
    print("=" * 70)
    data_name = "zebrafish"  # zebrafish, mammalian
    print("[ {} ]".format(data_name).center(60))
    split_type = "three_interpolation"  # all, three_interpolation
    print("Split type: {}".format(split_type))
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
    # =======================
    # recoverTraj(traj_data, traj_cell_types)
    # computeEmbedding()
    visTraj()



