'''
Description:
    Compare scNODE predictions with different hyperparameter settings..

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import torch
import numpy as np
from plotting.__init__ import *
from benchmark.BenchmarkUtils import loadSCData
from plotting.PlottingUtils import umapWithoutPCA
from plotting import _removeTopRightBorders, _removeAllBorders
from optim.running import constructscNODEModel

# ======================================================

def plotDiffLatentSize():
    # Load data and pre-processing
    print("=" * 70)
    data_name = "zebrafish"  # zebrafish, mammalian, drosophila, wot
    print("[ {} ]".format(data_name).center(60))
    split_type = "three_interpolation"  # interpolation
    print("Split type: {}".format(split_type))
    # -----
    exp_res = np.load("../../sc_Dynamic_Modelling/res/model_design/{}-{}-performance_vs_latent_size.npy".format(data_name, split_type), allow_pickle=True).item()
    latent_size_list = exp_res["latent_size"]
    # -----
    ot_list = exp_res["ot"]
    avg_ot_list = np.mean(ot_list, axis=1)
    l2_list = exp_res["l2"]
    avg_l2_list = np.mean(l2_list, axis=1)
    print("OT ", avg_ot_list)
    print("Avg={}, std={}".format(np.mean(avg_ot_list), np.std(avg_ot_list)))
    print("L2 ", avg_l2_list)
    print("Avg={}, std={}".format(np.mean(avg_l2_list), np.std(avg_l2_list)))
    plt.figure(figsize=(8, 4))
    plt.subplot(2, 1, 1)
    plt.bar(np.arange(len(avg_ot_list)), avg_ot_list, color=white_color, edgecolor="k", linewidth=2.0, capsize=5.0)
    plt.xticks([], [])
    plt.ylabel("OT")
    plt.subplot(2, 1, 2)
    plt.bar(np.arange(len(avg_l2_list)), avg_l2_list, color=white_color, edgecolor="k", linewidth=2.0, capsize=5.0)
    plt.ylabel("L2")
    plt.xlabel("Latent Size")
    plt.xticks(np.arange(len(latent_size_list)), ["{:d}".format(x) for x in latent_size_list])
    plt.tight_layout()
    plt.show()

# ======================================================

def plotDiffRegCoeff():
    # Load data and pre-processing
    print("=" * 70)
    data_name = "zebrafish"  # zebrafish, mammalian, drosophila, wot
    print("[ {} ]".format(data_name).center(60))
    split_type = "three_interpolation"  # interpolation
    print("Split type: {}".format(split_type))
    # -----
    exp_res = np.load("../../sc_Dynamic_Modelling/res/model_design/{}-{}-performance_vs_latent_coeff.npy".format(data_name, split_type), allow_pickle=True).item()
    latent_coeff_list = exp_res["latent_coeff_list"]
    # -----
    ot_list = exp_res["ot"]
    avg_ot_list = np.mean(ot_list, axis=1)
    l2_list = exp_res["l2"]
    avg_l2_list = np.mean(l2_list, axis=1)
    print("OT", avg_ot_list)
    print("L2", avg_l2_list)
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


def _loadModel(data_name, split_type, latent_coeff):
    dict_filename = "../../sc_Dynamic_Modelling/res/model_design/{}-{}-latent_coeff{:.2f}-state_dict.pt".format(data_name, split_type, latent_coeff)
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
    # -----
    exp_res = np.load("../../sc_Dynamic_Modelling/res/model_design/{}-{}-performance_vs_latent_coeff.npy".format(data_name, split_type), allow_pickle=True).item()
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
    # np.save("../../sc_Dynamic_Modelling/res/model_design/{}-{}-latent.npy".format(data_name, split_type), {
    #     "latent_umap_list": latent_umap_list,
    #     "next_umap_list": next_umap_list,
    #     "umap_model_list": umap_model_list,
    #     "tp_list": tp_list,
    #     "latent_coeff_list": latent_coeff_list,
    # })


def visLatent():
    print("=" * 70)
    data_name = "zebrafish"  # zebrafish, mammalian, drosophila, wot
    print("[ {} ]".format(data_name).center(60))
    split_type = "three_interpolation"  # interpolation
    print("Split type: {}".format(split_type))
    res = np.load("../../sc_Dynamic_Modelling/res/model_design/{}-{}-latent.npy".format(data_name, split_type), allow_pickle=True).item()
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
    plt.show()

# ======================================================

def plotDiffTrainTP():
    # Load data and pre-processing
    print("=" * 70)
    data_name = "zebrafish"  # zebrafish, mammalian, drosophila
    print("[ {} ]".format(data_name).center(60))
    split_type = "first_five"  # interpolation, forecasting
    print("Split type: {}".format(split_type))
    # -----
    exp_res = np.load("../../sc_Dynamic_Modelling/res/model_design/{}-{}-performance_vs_train_tps.npy".format(data_name, split_type), allow_pickle=True).item()
    pred_list = exp_res["pred"]
    tp_split_list = exp_res["tp"]
    n_tps = len(tp_split_list[0][0]) + len(tp_split_list[0][1])
    # -----------------------
    ot_list = exp_res["ot"]
    l2_list = exp_res["l2"]
    tp_split_list = exp_res["tp"]
    n_tps = len(tp_split_list[0][0]) + len(tp_split_list[0][1])
    max_num_test_tps = len(tp_split_list[0][1])
    min_num_train_tps = len(tp_split_list[0][0])
    for i in range(len(l2_list)):
        l2_list[i] = l2_list[i] + [np.nan for _ in range(max_num_test_tps - len(l2_list[i]))]
    for i in range(len(ot_list)):
        ot_list[i] = ot_list[i] + [np.nan for _ in range(max_num_test_tps - len(ot_list[i]))]

    ot_list = np.asarray(ot_list)
    l2_list = np.asarray(l2_list)
    tr_ot_list = [ot_list[:, i][~np.isnan(ot_list[:, i])] for i in range(ot_list.shape[1])]
    tr_l2_list = [l2_list[:, i][~np.isnan(l2_list[:, i])] for i in range(l2_list.shape[1])]

    plt.figure(figsize=(12, 3))
    plt.subplot(1, 2, 1)
    plt.boxplot(
        tr_ot_list, positions=np.arange(len(tr_ot_list)),
        showfliers=False, widths=0.7,
        medianprops=dict(linewidth=0), boxprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5)
    )
    sbn.stripplot(data=tr_ot_list, palette=[BlueRed_12.mpl_colors[0]], size=8, edgecolor="k", linewidth=1)
    plt.xlabel("Next TP")
    plt.xticks(range(len(tr_ot_list)), range(1, len(tr_ot_list) + 1))
    plt.ylabel("Wasserstein")
    _removeTopRightBorders()
    plt.subplot(1, 2, 2)
    plt.boxplot(
        tr_l2_list, positions=np.arange(len(tr_l2_list)),
        showfliers=False, widths=0.7,
        medianprops=dict(linewidth=0), boxprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5)
    )
    sbn.stripplot(data=tr_l2_list, palette=[BlueRed_12.mpl_colors[0]], size=8, edgecolor="k", linewidth=1)
    plt.ylabel("L2")
    plt.xlabel("Next TP")
    plt.xticks(range(len(tr_l2_list)), range(1, len(tr_l2_list)+1))
    _removeTopRightBorders()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    #  Different latent size
    plotDiffLatentSize()
    # -----
    # Different regularization coefficient (beta)
    plotDiffRegCoeff()
    visLatent()
    # -----
    # Different number of training timepoints
    plotDiffTrainTP()

