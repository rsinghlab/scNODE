'''
Description:
    Compare performance with different number of training time points.
'''
import torch
import torch.distributions as dist
import numpy as np
from datetime import datetime
from tqdm import tqdm
import itertools

from model.layer import LinearNet, LinearVAENet
from plotting.visualization import plotUMAP, plotPredTestTime, umapWithoutPCA
from plotting import linearSegmentCMap, _removeAllBorders
from data.preprocessing import splitBySpec
from benchmark.Compare_SingleCell_Predictions import basicStats, globalEvaluation
from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, sampleGaussian
from optim.running import constructscNODEModel, scNODETrainWithPreTrain, scNODEPredict
from plotting.__init__ import *

# ======================================================
from umap.umap_ import nearest_neighbors as umap_nearest_neighbors


# ======================================================

def runExp():
    # Load data and pre-processing
    print("=" * 70)
    data_name = "zebrafish"  # zebrafish, mammalian, drosophila, wot, pancreatic, embryoid
    print("[ {} ]".format(data_name).center(60))
    split_type = "three_interpolation"  # three_interpolation, two_forecasting
    print("Split type: {}".format(split_type))
    ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name, split_type)
    train_tps, test_tps = tpSplitInd(data_name, split_type)
    data = ann_data.X

    # Convert to torch project
    traj_data = [torch.FloatTensor(data[np.where(cell_tps == t)[0], :]) for t in
                 range(1, n_tps + 1)]  # (# tps, # cells, # genes)
    if cell_types is not None:
        traj_cell_types = [cell_types[np.where(cell_tps == t)[0]] for t in range(1, n_tps + 1)]

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

    # -------------------------------------
    latent_coeff_list = [0.0, 0.25, 0.5, 0.75, 1.0]
    latent_dim = 50
    pretrain_iters = 200
    pretrain_lr = 1e-3
    latent_coeff = 1.0
    epochs = 10
    iters = 100
    batch_size = 32
    lr = 1e-3
    act_name = "relu"
    n_sim_cells = 2000

    ot_list = []
    dist_list = []
    for latent_t, latent_coeff in enumerate(latent_coeff_list):
        print("*" * 70)
        print("Latent coeff = {}".format(latent_coeff))
        # Construct VAE
        print("-" * 60)
        latent_enc_act = "none"
        latent_dec_act = "relu"
        enc_latent_list = [latent_dim]
        dec_latent_list = [latent_dim]
        drift_latent_size = [latent_dim]

        latent_encoder = LinearVAENet(
            input_dim=n_genes, latent_size_list=enc_latent_list, output_dim=latent_dim, act_name=latent_enc_act
        )  # encoder
        obs_decoder = LinearNet(
            input_dim=latent_dim, latent_size_list=dec_latent_list, output_dim=n_genes, act_name=latent_dec_act
        )  # decoder
        print(latent_encoder)
        print(obs_decoder)
        # Model running
        latent_ode_model = constructscNODEModel(
            n_genes, latent_dim=latent_dim,
            enc_latent_list=enc_latent_list, dec_latent_list=dec_latent_list, drift_latent_size=drift_latent_size,
            latent_enc_act="none", latent_dec_act=act_name, drift_act=act_name,
            ode_method="euler"
        )
        latent_ode_model, loss_list, recon_obs, first_latent_dist, latent_seq = scNODETrainWithPreTrain(train_data,
                                                                                                        train_tps,
                                                                                                        latent_ode_model,
                                                                                                        latent_coeff=latent_coeff,
                                                                                                        epochs=epochs,
                                                                                                        iters=iters,
                                                                                                        batch_size=batch_size,
                                                                                                        lr=lr,
                                                                                                        pretrain_iters=pretrain_iters,
                                                                                                        pretrain_lr=pretrain_lr)
        all_recon_obs = scNODEPredict(latent_ode_model, first_latent_dist, tps,
                                      n_cells=n_sim_cells)  # (# trajs, # tps, # genes)
        reorder_pred_data = [all_recon_obs[:, t, :] for t in range(all_recon_obs.shape[1])]
        # Compute metric for testing time points
        print("Compute metrics...")
        test_tps_list = [int(t) for t in test_tps]
        cur_ot_list = []
        cur_dist_list = []
        for t in test_tps_list:
            print("-" * 70)
            print("t = {}".format(t))
            # -----
            pred_global_metric = globalEvaluation(traj_data[t].detach().numpy(), all_recon_obs[:, t, :])
            print(pred_global_metric)
            cur_ot_list.append(pred_global_metric["ot"])
            cur_dist_list.append(pred_global_metric["l2"])
        ot_list.append(cur_ot_list)
        dist_list.append(cur_dist_list)
        torch.save(
            latent_ode_model.state_dict(),
            "../res/model_design/{}-{}-latent_coeff{:.2f}-state_dict.pt".format(data_name, split_type, latent_coeff)
        ) # save model for each latent coeff
    np.save(
        "../res/model_design/{}-{}-performance_vs_latent_coeff.npy".format(data_name, split_type),
        {
            "ot": ot_list,
            "l2": dist_list,
            "latent_coeff_list": latent_coeff_list
        }
    )


def evaluateExp():
    # Load data and pre-processing
    print("=" * 70)
    data_name = "zebrafish"  # zebrafish, mammalian, drosophila, wot
    print("[ {} ]".format(data_name).center(60))
    split_type = "three_interpolation"  # interpolation
    print("Split type: {}".format(split_type))
    ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name, split_type)
    data = ann_data.X
    traj_data = [data[np.where(cell_tps == t)[0], :] for t in range(1, n_tps + 1)]  # (# tps, # cells, # genes)
    cell_num_list = [each.shape[0] for each in traj_data]
    # -----
    exp_res = np.load("../res/model_design/{}-{}-performance_vs_latent_coeff.npy".format(data_name, split_type), allow_pickle=True).item()
    train_tps, test_tps = tpSplitInd(data_name, split_type)
    latent_coeff_list = exp_res["latent_coeff_list"]
    # -----
    ot_list = exp_res["ot"]
    avg_ot_list = np.mean(ot_list, axis=1)
    l2_list = exp_res["l2"]
    avg_l2_list = np.mean(l2_list, axis=1)
    print("OT", avg_ot_list)
    print("L2", avg_l2_list)
    n_tps = len(train_tps) + len(test_tps)
    color_list = Cube1_6.mpl_colors
    plt.figure(figsize=(8, 4))
    plt.subplot(2, 1, 1)
    plt.bar(np.arange(len(avg_ot_list)), avg_ot_list, color=white_color, edgecolor="k", linewidth=2.0, capsize=5.0)
    plt.xticks([], [])
    plt.ylabel("OT")
    plt.ylim(400, 450)
    plt.subplot(2, 1, 2)
    plt.bar(np.arange(len(avg_l2_list)), avg_l2_list, color=white_color, edgecolor="k", linewidth=2.0, capsize=5.0)
    plt.ylabel("L2")
    plt.ylim(30, 35)
    plt.xlabel("Latent Coefficient")
    plt.xticks(np.arange(len(latent_coeff_list)), ["{:.2f}".format(x) for x in latent_coeff_list])
    plt.tight_layout()
    plt.show()

# ======================================================

def _loadModel(data_name, split_type, latent_coeff):
    dict_filename = "../res/model_design/{}-{}-latent_coeff{:.2f}-state_dict.pt".format(data_name, split_type, latent_coeff)
    latent_ode_model = constructscNODEModel(
        n_genes=2000, latent_dim=50,
        enc_latent_list=[50], dec_latent_list=[50], drift_latent_size=[50],
        latent_enc_act="none", latent_dec_act="relu", drift_act="relu",
        ode_method="euler"
    )
    latent_ode_model.load_state_dict(torch.load(dict_filename))
    latent_ode_model.eval()
    return latent_ode_model


def _computeDrift(traj_data, latent_ode_model):
    # Compute latent and drift seq
    latent_seq, drift_seq = latent_ode_model.encodingSeq(traj_data)
    next_seq = [each + 0.1 * drift_seq[i] for i, each in enumerate(latent_seq)]
    return latent_seq, next_seq, drift_seq


def _computeEmbedding(latent_seq, next_seq, n_neighbors, min_dist):
    latent_tp_list = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(latent_seq)])
    umap_latent_data, umap_model = umapWithoutPCA(np.concatenate(latent_seq, axis=0), n_neighbors=n_neighbors, min_dist=min_dist)
    # umap_latent_data, umap_model = onlyPCA(np.concatenate(latent_seq, axis=0), pca_pcs=2)
    umap_latent_data = [umap_latent_data[np.where(latent_tp_list == t)[0], :] for t in range(len(latent_seq))]
    umap_next_data = [umap_model.transform(each) for each in next_seq]
    return umap_latent_data, umap_next_data, umap_model, latent_tp_list


def computeLatent():
    # Load data and pre-processing
    print("=" * 70)
    data_name = "zebrafish"  # zebrafish, mammalian, drosophila, wot
    print("[ {} ]".format(data_name).center(60))
    split_type = "three_interpolation"  # interpolation
    print("Split type: {}".format(split_type))
    ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name, split_type)
    data = ann_data.X
    traj_data = [torch.FloatTensor(data[np.where(cell_tps == t)[0], :]) for t in range(1, n_tps + 1)]  # (# tps, # cells, # genes)
    cell_num_list = [each.shape[0] for each in traj_data]
    # -----
    exp_res = np.load("../res/model_design/{}-{}-performance_vs_latent_coeff.npy".format(data_name, split_type), allow_pickle=True).item()
    train_tps, test_tps = tpSplitInd(data_name, split_type)
    latent_coeff_list = exp_res["latent_coeff_list"]
    # -----
    latent_umap_list = []
    next_umap_list = []
    umap_model_list = []
    tp_list = []
    for latent_coeff in latent_coeff_list:
        print("*" * 70)
        print("Latent coeff = {}".format(latent_coeff))
        latent_ode_model = _loadModel(data_name, split_type, latent_coeff)
        print("Computing latent and drift sequence...")
        latent_seq, next_seq, drift_seq = _computeDrift(traj_data, latent_ode_model)
        print("Computing embedding...")
        umap_latent_data, umap_next_data, umap_model, latent_tp_list = _computeEmbedding(
            latent_seq, next_seq, n_neighbors=50, min_dist=0.0  # 0.25
        )
        latent_umap_list.append(umap_latent_data)
        next_umap_list.append(umap_next_data)
        umap_model_list.append(umap_model)
        tp_list.append(latent_tp_list)
    np.save("../res/model_design/{}-{}-latent.npy".format(data_name, split_type), {
        "latent_umap_list": latent_umap_list,
        "next_umap_list": next_umap_list,
        "umap_model_list": umap_model_list,
        "tp_list": tp_list,
        "latent_coeff_list": latent_coeff_list,
    })
    # -----


def visLatent():
    print("=" * 70)
    data_name = "zebrafish"  # zebrafish, mammalian, drosophila, wot
    print("[ {} ]".format(data_name).center(60))
    split_type = "three_interpolation"  # interpolation
    print("Split type: {}".format(split_type))
    res = np.load("../res/model_design/{}-{}-latent.npy".format(data_name, split_type), allow_pickle=True).item()
    latent_umap_list = res["latent_umap_list"]
    next_umap_list = res["next_umap_list"]
    umap_model_list = res["umap_model_list"]
    tp_list = res["tp_list"]
    latent_coeff_list = res["latent_coeff_list"]
    n_tps = len(np.unique(tp_list[0]))
    # -----
    ot_list = [440.94169148, 436.07623297, 427.63611927, 424.45108616, 429.52243109]
    l2_list = [33.64310094, 33.46972456, 33.25594141, 33.12566671, 33.27392067]
    color_list = linearSegmentCMap(n_tps, "viridis")
    fig, ax_list = plt.subplots(1, len(latent_coeff_list), figsize=(12, 4))
    for i, l_c in enumerate(latent_coeff_list):
        ax_list[i].set_title("coeff = {:.2f} \n Wass={:.2f}".format(l_c, ot_list[i]))
        each_latent = latent_umap_list[i]
        for t, each in enumerate(each_latent):
            ax_list[i].scatter(each[:, 0], each[:, 1], color=color_list[t], s=5, alpha=0.8)
            ax_list[i].set_xticks([])
            ax_list[i].set_yticks([])
            _removeAllBorders(ax_list[i])
    for t in range(n_tps):
        ax_list[-1].scatter([], [], color=color_list[t], s=20, alpha=1.0, label=t)
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.savefig("../res/figs/diff_latent_coeff.pdf")
    plt.show()




if __name__ == '__main__':
    # runExp()
    # evaluateExp()
    # -----
    # computeLatent()
    visLatent()
    pass