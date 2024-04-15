'''
Description:
    Use PRESCIENT predictions in recovering cell trajectories.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import scanpy
import torch
import numpy as np
from datetime import datetime
import pickle as pkl


from benchmark.BenchmarkUtils import loadSCData, tunedPRESCIENTPars, tpSplitInd
from plotting.__init__ import *
from plotting.PlottingUtils import umapWithPCA
from baseline.prescient_model.running import prescientTrain, prescientSimulate
from baseline.prescient_model.process_data import main as prepare_data

# ======================================================

def recoverTraj(traj_data, data_dict, train_tps, sim_data):
    n_neighbors = 50
    min_dist = 0.1
    pca_pcs = 50
    # -----
    # Original data
    print("All tps...")
    all_tps = list(range(len(traj_data)))
    # -----
    # Remove several timepoints
    use_tps = train_tps
    print("Used tps {}...".format(use_tps))
    use_traj_data = [traj_data[i] for i in use_tps]
    # -----
    # Augment removed timepoints
    print("Augment...")
    if data_name == "zebrafish":
        pred_tps = torch.FloatTensor([
            0,
            1,
            2, 2.1, 2.25, 2.35, 2.5, 2.6, 2.75, 2.85,
            3,
            4, 4.1, 4.25, 4.35, 4.5, 4.6, 4.75, 4.85,
            5,
            6, 6.1, 6.25, 6.35, 6.5, 6.6, 6.75, 6.85,
            7,
            8, 8.1, 8.25, 8.35, 8.5, 8.6, 8.75, 8.85,
            9,
            10, 10.1, 10.25, 10.35, 10.5, 10.6, 10.75, 10.85,
            11, 11.1, 11.25, 11.35, 11.5, 11.6, 11.75, 11.85,
        ]) * 10
    if data_name == "drosophila":
        pred_tps = torch.FloatTensor([
            0,
            1,
            2, 2.1, 2.25, 2.35, 2.5, 2.6, 2.75, 2.85,
            3,
            4, 4.1, 4.25, 4.35, 4.5, 4.6, 4.75, 4.85,
            5,
            6, 6.1, 6.25, 6.35, 6.5, 6.6, 6.75, 6.85,
            7,
            8, 8.1, 8.25, 8.35, 8.5, 8.6, 8.75, 8.85,
            9,
            10,
            11,
        ]) * 10
    if data_name == "wot":
        pred_tps = torch.FloatTensor([
            0,
            1,
            2,
            3,
            4,
            5, 5.1, 5.25, 5.35, 5.5, 5.6, 5.75, 5.85,
            # 5, 5.25, 5.5, 5.75,
            6,
            7, 7.1, 7.25, 7.35, 7.5, 7.6, 7.75, 7.85,
            # 7, 7.25,  7.5, 7.75,
            8,
            9, 9.1, 9.25, 9.35, 9.5, 9.6, 9.75, 9.85,
            # 9, 9.25, 9.5, 9.75,
            10,
            11, 11.1, 11.25, 11.35, 11.5, 11.6, 11.75, 11.85,
            # 11, 11.25, 11.5, 11.75,
            12,
            13,
            14,
            15, 15.1, 15.25, 15.35, 15.5, 15.6, 15.75, 15.85,
            # 15, 15.25, 15.5, 15.75,
            16, 16.1, 16.25, 16.35, 16.5, 16.6, 16.75, 16.85,
            # 16, 16.25, 16.5, 16.75,
            17, 17.1, 17.25, 17.35, 17.5, 17.6, 17.75, 17.85,
            # 17, 17.25, 17.5, 17.75,
            18, 18.1, 18.25, 18.35, 18.5, 18.6, 18.75, 18.85,
            # 18, 18.25, 18.5, 18.75,
        ]) * 10
    pred_tps = [int(x) for x in pred_tps]
    sim_data = sim_data[pred_tps, :, :]
    sim_data = [scaler.inverse_transform(pca.inverse_transform(sim_data[t, :, :])) for t in range(sim_data.shape[0])]
    # -----
    # Predictions
    all_recon_obs = sim_data
    # -----
    res_filename = "./res/trajectory/{}-res-{}-PRESCIENT.npy".format(data_name, split_type)
    res_filename2 = "./res/trajectory/{}-res-{}-PRESCIENT-cell_tps.npy".format(data_name, split_type)
    np.save(res_filename2, pred_tps)
    print("Saving to {}".format(res_filename))
    np.save(
        res_filename,
        {"true": traj_data,
         "pred": all_recon_obs,

         "use_tps": use_tps,
         "pred_tps": pred_tps,
         },
        allow_pickle=True
    )

# ======================================================

def computeEmbedding():
    res_filename = "./res/trajectory/{}-res-{}-PRESCIENT.npy".format(data_name, split_type)
    # -----
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
    # -----
    all_cell_tps = np.concatenate([np.repeat(all_tps[idx], x.shape[0]) for idx, x in enumerate(traj_data)])
    use_cell_tps = np.concatenate([np.repeat(use_tps[idx], x.shape[0]) for idx, x in enumerate(use_traj_data)])
    aug_cell_tps = np.concatenate([np.repeat(pred_tps[idx], x.shape[0]) for idx, x in enumerate(aug_traj_data)])
    pred_cell_tps = np.concatenate([np.repeat(pred_tps[idx], x.shape[0]) for idx, x in enumerate(all_recon_obs)])
    # -----
    n_neighbors = 50
    min_dist = 0.25
    pca_pcs = 50
    if data_name == "wot":
        n_neighbors = 50
        min_dist = 0.8
    print("All")
    all_umap_traj, all_umap_model, all_pca_model = umapWithPCA(
        np.concatenate(traj_data, axis=0), n_neighbors=n_neighbors, min_dist=min_dist, pca_pcs=pca_pcs
    )
    print("Used")
    use_umap_traj = all_umap_model.transform(all_pca_model.transform(np.concatenate(use_traj_data, axis=0)))
    print("Aug")
    aug_umap_traj = all_umap_model.transform(all_pca_model.transform(np.concatenate(aug_traj_data, axis=0)))
    print("Pred")
    pred_umap_traj = all_umap_model.transform(all_pca_model.transform(np.concatenate(all_recon_obs, axis=0)))
    # -----
    np.save(
        "./res/trajectory/{}-res-embedding-neighbor{}-min_dist{:.2f}-pcs{}-project_to_all-{}-PRESCIENT.npy".format(data_name, n_neighbors, min_dist, pca_pcs, split_type),
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
            "aug_umap_traj": aug_umap_traj,
            "pred_umap_traj": pred_umap_traj,
        }
    )
    with open("./res/trajectory/{}-res-embedding-neighbor{}-min_dist{:.2f}-pcs{}-project_to_all-{}-PRESCIENT.pkl".format(data_name, n_neighbors, min_dist, pca_pcs, split_type), "wb") as file:
        pkl.dump({
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
            "aug_umap_traj": aug_umap_traj,
            "pred_umap_traj": pred_umap_traj,
        }, file)

# ======================================================

def visTraj():
    print("Loading embedding...")
    n_neighbors = 50
    min_dist = 0.25
    pca_pcs = 50
    if data_name == "wot":
        n_neighbors = 50
        min_dist = 0.8
    # res_filename = "../res/downstream_analysis/recover_traj/zebrafish-res-embedding-neighbor{}-min_dist{:.2f}-pcs{}-project_to_all-four_interpolation.npy".format(n_neighbors, min_dist, pca_pcs)
    # state_filename = "../res/downstream_analysis/recover_traj/zebrafish-latent_ODE_OT-state_dict-four_interpolation.pt"
    res_filename = "./res/trajectory/{}-res-embedding-neighbor{}-min_dist{:.2f}-pcs{}-project_to_all-{}-PRESCIENT.npy".format(data_name, n_neighbors, min_dist, pca_pcs, split_type)
    res_dict = np.load(res_filename, allow_pickle=True).item()
    use_tps = res_dict["use_tps"]
    all_cell_tps = res_dict["all_cell_tps"]
    use_cell_tps = res_dict["use_cell_tps"]
    aug_cell_tps = res_dict["aug_cell_tps"]

    all_umap_traj = res_dict["all_umap_traj"]
    use_umap_traj = res_dict["use_umap_traj"]
    aug_umap_traj = res_dict["aug_umap_traj"]

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

    np.save(
        "./res/trajectory/{}-PAGA_graph-{}-PRESCIENT.npy".format(data_name, split_type),
        {
            "all_conn": all_conn,
            "all_node_pos": all_node_pos,
            "all_edge": all_edge,
            "all_cell_types": all_cell_types,

            "use_conn": use_conn,
            "use_node_pos": use_node_pos,
            "use_edge": use_edge,
            "use_cell_types": use_cell_types,

            "aug_conn": aug_conn,
            "aug_node_pos": aug_node_pos,
            "aug_edge": aug_edge,
            "aug_cell_types": aug_cell_types,
        }

    )
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
    if data_name == "zebrafish":
        aug_cell_tps[(aug_cell_tps >= 2) & (aug_cell_tps < 3)] = 2
        aug_cell_tps[(aug_cell_tps >= 4) & (aug_cell_tps < 5)] = 4
        aug_cell_tps[(aug_cell_tps >= 6) & (aug_cell_tps < 7)] = 6
        aug_cell_tps[(aug_cell_tps >= 8) & (aug_cell_tps < 9)] = 8
    if data_name == "drosophila":
        aug_cell_tps[(aug_cell_tps >= 2) & (aug_cell_tps < 3)] = 2
        aug_cell_tps[(aug_cell_tps >= 4) & (aug_cell_tps < 5)] = 4
        aug_cell_tps[(aug_cell_tps >= 6) & (aug_cell_tps < 7)] = 6
        aug_cell_tps[(aug_cell_tps >= 8) & (aug_cell_tps < 9)] = 8
    if data_name == "wot":
        aug_cell_tps[(aug_cell_tps >= 5) & (aug_cell_tps < 6)] = 5
        aug_cell_tps[(aug_cell_tps >= 7) & (aug_cell_tps < 8)] = 7
        aug_cell_tps[(aug_cell_tps >= 9) & (aug_cell_tps < 10)] = 9
        aug_cell_tps[(aug_cell_tps >= 11) & (aug_cell_tps < 12)] = 11
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
    plt.tight_layout()
    plt.show()

# ======================================================

from netrd.distance import IpsenMikhailov
import networkx as ntx
def compareGrpah():
    res = np.load(
        "./res/trajectory/{}-PAGA_graph-{}-PRESCIENT.npy".format(data_name, split_type),
        allow_pickle=True
    ).item()
    all_conn = res["all_conn"]
    use_conn = res["use_conn"]
    aug_conn = res["aug_conn"]
    # -----
    all_graph = ntx.from_numpy_array(all_conn)
    use_graph = ntx.from_numpy_array(use_conn)
    aug_graph = ntx.from_numpy_array(aug_conn)
    dist_func = IpsenMikhailov()
    print("Compute for removal...")
    all_use_dist = dist_func.dist(all_graph, use_graph)
    print("Compute for prediction...")
    all_aug_dist = dist_func.dist(all_graph, aug_graph)
    print("All-Use Divergence = {}".format(all_use_dist))
    print("All-Aug Divergence = {}".format(all_aug_dist))


if __name__ == '__main__':
    # Load data
    print("=" * 70)
    data_name = "wot"  # zebrafish, drosophila, wot
    print("[ {} ]".format(data_name).center(60))
    split_type = "remove_recovery" # the trajectory inference is only tested on hard tasks
    print("Split type: {}".format(split_type))
    ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name, split_type)
    train_tps, test_tps = tpSplitInd(data_name, split_type)
    data = ann_data.X
    all_tps = list(np.arange(n_tps))
    traj_data = [data[np.where(cell_tps == t)[0], :] for t in range(1, n_tps + 1)]
    processed_data = ann_data[
        [t for t in range(ann_data.shape[0]) if (ann_data.obs["tp"].values[t] - 1.0) in train_tps]
    ]
    cell_tps = (processed_data.obs["tp"].values - 1.0).astype(int)
    cell_types = np.repeat("NAN", processed_data.shape[0])
    genes = processed_data.var.index.values

    # Parameter settings
    k_dim, layers, sd, tau, clip = tunedPRESCIENTPars(data_name, split_type)

    # PCA and construct data dict
    print("Construct data dict...")
    data_dict, scaler, pca, um = prepare_data(
        processed_data.X, cell_tps, cell_types, genes,
        num_pcs=k_dim, num_neighbors_umap=10
    )
    data_dict["y"] = list(range(train_tps[-1] + 1))
    data_dict["x"] = [data_dict["x"][train_tps.index(t)] if t in train_tps else [] for t in data_dict["y"]]
    data_dict["xp"] = [data_dict["xp"][train_tps.index(t)] if t in train_tps else [] for t in data_dict["y"]]
    data_dict["w"] = [data_dict["w"][train_tps.index(t)] if t in train_tps else [] for t in data_dict["y"]]

    print("Training...")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    train_epochs = 2000
    train_lr = 1e-3
    final_model, best_state_dict, config, loss_list = prescientTrain(
        data_dict, data_name=data_name, out_dir="", train_t=train_tps[1:], timestamp=timestamp,
        k_dim=k_dim, layers=layers, train_epochs=train_epochs, train_lr=train_lr,
        train_sd=sd, train_tau=tau, train_clip=clip
    )

    print("Simulating...")
    num_cells = 500
    num_steps = (n_tps) * 10
    sim_data = prescientSimulate(data_dict, data_name, best_state_dict, num_cells, num_steps, config)
    # =======================
    recoverTraj(traj_data, data_dict, train_tps, sim_data)
    computeEmbedding()
    visTraj()
    compareGrpah()



