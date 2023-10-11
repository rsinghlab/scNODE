'''
Description:
    Compare model predictions.
'''
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy
from plotting.__init__ import *
from plotting.visualization import umapWithPCA, umapWithoutPCA
from plotting.utils import _removeAllBorders, _removeTopRightBorders
from optim.evaluation import oneDimEvaluation, twoDimEvaluation, globalEvaluation, basicStats, LISIScore


# ======================================================
def _loadVAEPrediction(filename):
    res = np.load(filename, allow_pickle=True).item()
    true_data = res["true"]
    pred_data = res["pred"]
    tps = res["tps"]["all"]
    test_tps = res["tps"]["test"]
    loss = res["loss"]
    return true_data, pred_data, tps, test_tps, (loss, )


def _loadPRESCIENTPrediction(filename):
    res = np.load(filename, allow_pickle=True).item()
    true_data = res["true"]
    pred_data = res["pred"]
    tps = res["tps"]["all"]
    test_tps = res["tps"]["test"]
    latent = res["latent_seq"]
    pred_data = np.moveaxis(np.asarray(pred_data), [0, 1, 2], [1, 0, 2])
    latent = np.moveaxis(np.asarray(latent), [0, 1, 2], [1, 0, 2])
    return true_data, pred_data, tps, test_tps, (latent, )


def _loadDummyPrediction(filename):
    res = np.load(filename, allow_pickle=True).item()
    true_data = res["true"]
    pred_data = res["pred"]
    tps = res["tps"]["all"]
    test_tps = res["tps"]["test"]
    loss = res["loss"]
    return true_data, pred_data, tps, test_tps, (loss, )


def _loadFNNPrediction(filename):
    res = np.load(filename, allow_pickle=True).item()
    true_data = res["true"]
    pred_data = res["pred"]
    tps = res["tps"]["all"]
    test_tps = res["tps"]["test"]
    loss = res["loss"]
    return true_data, pred_data, tps, test_tps, (loss, )


def _loadWOTPrediction(filename):
    res = np.load(filename, allow_pickle=True).item()
    true_data = res["true"]
    pred_data = res["pred"]
    tps = res["tps"]["all"]
    test_tps = res["tps"]["test"]
    return true_data, pred_data, tps, test_tps, None


def _loadTrjectoryNetPrediction(filename):
    res = np.load(filename, allow_pickle=True).item()
    true_data = res["true"]
    pred_data = res["pred"]
    tps = res["tps"]["all"]
    test_tps = res["tps"]["test"]
    return true_data, pred_data, tps, test_tps, None


# ======================================================

load_func_dict = {
    "latent_ODE_OT_pretrain": _loadVAEPrediction,
    "PRESCIENT": _loadPRESCIENTPrediction,
    "WOT": _loadWOTPrediction,
    "dummy": _loadDummyPrediction,
    "FNN": _loadFNNPrediction,
    "TrajectoryNet": _loadTrjectoryNetPrediction,
}


def loadModelPrediction(data_name, split_type, model_list):
    file_name = "../res/single_cell/experimental/{}/{}-{}-{}-res.npy"
    true_data, _, tps, test_tps, _ = _loadVAEPrediction(file_name.format(data_name, data_name, split_type, "latent_ODE_OT_pretrain"))
    model_true_data = true_data
    model_pred_data = []
    for m in model_list:
        m_true_data, m_pred_data, m_tps, m_test_tps, m_misc = load_func_dict[m](
            file_name.format(data_name, data_name, split_type, m)
        )
        if m == "PRESCIENT":
            m_pred_data = [m_pred_data[:, t, :] for t in range(m_pred_data.shape[1])]
        model_pred_data.append(m_pred_data)
    return model_true_data, model_pred_data, tps, test_tps


# ======================================================
import phate
from sklearn.decomposition import PCA
def _phateWithoutPCA(traj_data, knn=15):
    phate_model = phate.PHATE(n_components=2, knn=knn)
    phate_traj_data = phate_model.fit_transform(traj_data)
    return phate_traj_data, phate_model

# PHATE has internal PCA computation
# def _phateWithPCA(traj_data, pca_pcs, knn=15):
#     pca_model = PCA(n_components=pca_pcs, svd_solver="arpack")
#     phate_model = phate.PHATE(n_components=2, knn=knn)
#     phate_traj_data = phate_model.fit_transform(pca_model.fit_transform(traj_data))
#     return phate_traj_data, phate_model, pca_model

def _pca(traj_data):
    pca_model = PCA(n_components=2, svd_solver="arpack")
    pca_traj_data = pca_model.fit_transform(traj_data)
    return pca_traj_data, pca_model


def computeVisEmbedding(true_data, model_pred_data, embed_name):
    if embed_name == "umap":
        true_umap_traj, umap_model = umapWithoutPCA(np.concatenate(true_data, axis=0), n_neighbors=50, min_dist=0.2)
        model_pred_umap_traj = [umap_model.transform(np.concatenate(m_pred, axis=0)) for m_pred in model_pred_data]
    elif embed_name == "pca_umap":
        true_umap_traj, umap_model, pca_model = umapWithPCA(
            np.concatenate(true_data, axis=0), n_neighbors=50, min_dist=0.1, pca_pcs=50
        )
        model_pred_umap_traj = [
            umap_model.transform(pca_model.transform(np.concatenate(m_pred, axis=0)))
            for m_pred in model_pred_data
        ]
    elif embed_name == "pca":
        true_umap_traj, pca_model = _pca(np.concatenate(true_data, axis=0))
        model_pred_umap_traj = [pca_model.transform(np.concatenate(m_pred, axis=0)) for m_pred in model_pred_data]
    elif embed_name == "phate":
        true_umap_traj, phate_model = _phateWithoutPCA(np.concatenate(true_data, axis=0), knn=50)
        model_pred_umap_traj = [phate_model.transform(np.concatenate(m_pred, axis=0)) for m_pred in model_pred_data]
    else:
        raise ValueError("Unknown embedding type {}!".format(embed_name))
    return true_umap_traj, model_pred_umap_traj


# ======================================================

mdoel_name_dict = {
    "latent_ODE_OT_pretrain": "scNODE",
    "PRESCIENT": "PRESCIENT",
    "WOT": "WOT",
    "dummy": "Dummy",
    "FNN": "FNN",
    "TrajectoryNet": "TrajectoryNet",
}


def compareUMAPTestTime(
        true_umap_traj, model_pred_umap_traj,
        true_cell_tps, model_cell_tps, test_tps, model_list):
    n_tps = len(np.unique(true_cell_tps))
    n_models = len(model_list)
    color_list = Vivid_10.mpl_colors
    fig, ax_list = plt.subplots(1, n_models+1, figsize=(14, 4))
    # Plot true data
    ax_list[0].set_title("True Data", fontsize=15)
    ax_list[0].scatter(true_umap_traj[:, 0], true_umap_traj[:, 1], c=gray_color, s=40, alpha=0.5)
    for t_idx, t in enumerate(test_tps):
        c = color_list[t_idx]
        true_t_idx = np.where(true_cell_tps == t)[0]
        ax_list[0].scatter(
            true_umap_traj[true_t_idx, 0], true_umap_traj[true_t_idx, 1],
            label=int(t), color=c, s=20, alpha=0.9
        )
    ax_list[0].set_xticks([])
    ax_list[0].set_yticks([])
    _removeAllBorders(ax_list[0])
    # Plot pred data
    for m_idx, m in enumerate(model_list):
        X = np.concatenate([true_umap_traj, model_pred_umap_traj[m_idx]])
        d_labels = np.concatenate([np.zeros(true_umap_traj.shape[0], ), np.ones(model_pred_umap_traj[m_idx].shape[0])])
        LISI = LISIScore(X, pd.DataFrame(data=d_labels, columns=['Type']), ['Type'])
        miLISI = np.median(LISI)
        # -----
        # ax_list[m_idx+1].set_title("{} \n miLISI={:.2f}".format(mdoel_name_dict[m], miLISI), fontsize=15)
        ax_list[m_idx+1].set_title(mdoel_name_dict[m], fontsize=15)
        ax_list[m_idx+1].scatter(true_umap_traj[:, 0], true_umap_traj[:, 1], c=gray_color, s=40, alpha=0.5)
        for t_idx, t in enumerate(test_tps):
            c = color_list[t_idx]
            pred_t_idx = np.where(model_cell_tps[m_idx] == t)[0]
            ax_list[m_idx+1].scatter(
                model_pred_umap_traj[m_idx][pred_t_idx, 0], model_pred_umap_traj[m_idx][pred_t_idx, 1],
                label=int(t), color=c, s=20, alpha=0.9)
        ax_list[m_idx+1].set_xticks([])
        ax_list[m_idx+1].set_yticks([])
        _removeAllBorders(ax_list[m_idx+1])
    ax_list[-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="Test TPs", title_fontsize=14, fontsize=13)
    plt.tight_layout()
    plt.savefig("../res/figs/{}-{}-{}.pdf".format(data_name, split_type, embed_name))
    plt.savefig("../res/figs/{}-{}-{}.png".format(data_name, split_type, embed_name))
    plt.show()


# ======================================================

def computeMetric(true_data, model_pred_data, tps, test_tps, model_list, n_sim_cells):
    test_tps = [int(t) for t in test_tps]
    print("Compute basic stats...")
    n_tps = len(tps)
    true_cell_stats = [basicStats(true_data[t], axis="cell") for t in range(n_tps)]
    true_gene_stats = [basicStats(true_data[t], axis="gene") for t in range(n_tps)]
    model_cell_stats = [
        [basicStats(m_pred_data[t], axis="cell") for t in range(n_tps)]
        for m_pred_data in model_pred_data
    ]
    model_gene_stats = [
        [basicStats(m_pred_data[t], axis="gene") for t in range(n_tps)]
        for m_pred_data in model_pred_data
    ]
    basic_stats = {
        "true": {"cell": true_cell_stats, "gene": true_gene_stats}
    }
    for m_idx, m in enumerate(model_list):
        basic_stats[m] = {"cell": model_cell_stats[m_idx], "gene": model_gene_stats[m_idx]}
    # -----
    # Down-sampling predictions: Because FNN and Dummy generate more samples
    sampled_model_pred_data = copy.deepcopy(model_pred_data)
    for m_idx, m in enumerate(model_list):
        for t in test_tps:
            if sampled_model_pred_data[m_idx][t].shape[0] < n_sim_cells:
                rand_idx = np.random.choice(np.arange(sampled_model_pred_data[m_idx][t].shape[0]), n_sim_cells, replace=True)
                sampled_model_pred_data[m_idx][t] = sampled_model_pred_data[m_idx][t][rand_idx, :]
            if sampled_model_pred_data[m_idx][t].shape[0] > n_sim_cells:
                rand_idx = np.random.choice(np.arange(sampled_model_pred_data[m_idx][t].shape[0]), n_sim_cells, replace=False)
                sampled_model_pred_data[m_idx][t] = sampled_model_pred_data[m_idx][t][rand_idx, :]
    # -----
    print("Compute metrics...")
    test_eval_metric = {t: {m: None for m in model_list} for t in test_tps}
    for t in test_tps:
        print("-" * 70)
        print("t = {}".format(t))
        for m_idx, m in enumerate(model_list):
            m_cell_1D_metric = oneDimEvaluation(true_cell_stats[t], model_cell_stats[m_idx][t])
            m_gene_1D_metric = oneDimEvaluation(true_gene_stats[t], model_gene_stats[m_idx][t])
            m_cell_2D_metric = twoDimEvaluation(true_cell_stats[t], model_cell_stats[m_idx][t])
            m_gene_2D_metric = twoDimEvaluation(true_gene_stats[t], model_gene_stats[m_idx][t])
            m_global_metric = globalEvaluation(true_data[t], sampled_model_pred_data[m_idx][t])
            # -----
            test_eval_metric[t][m] = {
                "cell_1d": m_cell_1D_metric, "cell_2d": m_cell_2D_metric,
                "gene_1d": m_gene_1D_metric, "gene_2d": m_gene_2D_metric,
                "global": m_global_metric
            }
    return test_eval_metric, basic_stats

# ======================================================

inter_model_list = ["latent_ODE_OT_pretrain", "PRESCIENT", "WOT", "TrajectoryNet", "dummy"]
extra_model_list = ["latent_ODE_OT_pretrain", "PRESCIENT", "dummy"]
dataset_name_dict = {
    "embryoid": "EB",
    "pancreatic": "MP",
    "zebrafish":"ZF",
    "mammalian":"MB",
    "drosophila":"DR",
    "WOT":"SC"
} # embryoid body, mouse pancreatic, zebrafish, mouse brain, drosophila, wot
dataset_list = ["embryoid", "pancreatic", "zebrafish", "mammalian", "drosophila", "WOT"]

def plotMetricBar():
    inter_avg_ot = []
    inter_std_ot = []
    inter_avg_l2 = []
    inter_std_l2 = []
    extra_avg_ot = []
    extra_std_ot = []
    extra_avg_l2 = []
    extra_std_l2 = []
    for data_name in dataset_list:
        # interpolation
        if data_name in ["embryoid", "pancreatic"]:
            inter_split = "one_interpolation"
        if data_name in ["zebrafish", "mammalian", "drosophila", "WOT"]:
            inter_split = "three_interpolation"
        metric_res = np.load("../res/comparison/{}-{}-model_metrics.npy".format(data_name, inter_split), allow_pickle=True).item()
        tmp_ot = np.asarray([[metric_res[t][m]["global"]["ot"] if m in metric_res[t] else np.nan for m in inter_model_list] for t in metric_res])
        tmp_l2 = np.asarray([[metric_res[t][m]["global"]["l2"] if m in metric_res[t] else np.nan for m in inter_model_list] for t in metric_res])
        inter_avg_ot.append(np.nanmean(tmp_ot, axis=0))
        inter_std_ot.append(np.nanstd(tmp_ot, axis=0))
        inter_avg_l2.append(np.nanmean(tmp_l2, axis=0))
        inter_std_l2.append(np.nanstd(tmp_l2, axis=0))
        # extrapolation
        if data_name in ["embryoid", "pancreatic"]:
            extra_split = "one_forecasting"
        if data_name in ["mammalian", "drosophila", "WOT"]:
            extra_split = "three_forecasting"
        if data_name in ["zebrafish"]:
            extra_split = "two_forecasting"
        metric_res = np.load("../res/comparison/{}-{}-model_metrics.npy".format(data_name, extra_split), allow_pickle=True).item()
        tmp_ot = np.asarray([[metric_res[t][m]["global"]["ot"] if m in metric_res[t] else np.nan for m in extra_model_list] for t in metric_res])
        tmp_l2 = np.asarray([[metric_res[t][m]["global"]["l2"] if m in metric_res[t] else np.nan for m in extra_model_list] for t in metric_res])
        extra_avg_ot.append(np.nanmean(tmp_ot, axis=0))
        extra_std_ot.append(np.nanmean(tmp_ot, axis=0))
        extra_avg_l2.append(np.nanmean(tmp_l2, axis=0))
        extra_std_l2.append(np.nanstd(tmp_l2, axis=0))
    # -----
    inter_avg_ot = np.asarray(inter_avg_ot)
    extra_avg_ot = np.asarray(extra_avg_ot)
    color_list = Vivid_10.mpl_colors
    fig, ax_list = plt.subplots(2, 1, figsize=(8, 8.5))
    inter_width = 0.1
    x_base = np.arange(len(dataset_list)) - inter_width*len(inter_model_list)
    ax_list[0].set_title("Interpolation")
    for i in range(len(inter_model_list)):
        ax_list[0].bar(
            x=x_base+i*inter_width,
            height=inter_avg_ot[:, i],
            width=inter_width,
            align="edge",
            color=model_colors_dict[inter_model_list[i]],
            label=mdoel_name_dict[inter_model_list[i]],
            edgecolor="k",
            linewidth=0.2,
        )
        # ax_list[0].legend(fontsize=12)
    ax_list[0].set_ylim(0.0, 600)
    ax_list[0].set_yticks([0, 200, 400, 600])
    ax_list[0].set_yticklabels(["0", "200", "400", r"$\geq$600"])
    ax_list[0].set_xticks([])
    ax_list[0].set_ylabel("Wasserstein")
    _removeTopRightBorders(ax_list[0])
    extra_width = 0.2
    x_base = np.arange(len(dataset_list)) - extra_width * len(extra_model_list)
    ax_list[1].set_title("Extrapolation")
    for i in range(len(extra_model_list)):
        ax_list[1].bar(
            x=x_base + i * extra_width,
            height=extra_avg_ot[:, i],
            width=extra_width,
            align="edge",
            color=model_colors_dict[extra_model_list[i]],
            label=mdoel_name_dict[extra_model_list[i]],
            edgecolor="k",
            linewidth=0.2,
        )
        # ax_list[1].legend(fontsize=12)
    ax_list[1].set_ylim(0.0, 800)
    ax_list[1].set_yticks([0, 200, 400, 600, 800])
    ax_list[1].set_yticklabels(["0", "200", "400", "600", r"$\geq$800"])
    ax_list[1].set_ylabel("Wasserstein")
    _removeTopRightBorders(ax_list[1])
    plt.xticks(np.arange(len(dataset_list)) - extra_width*len(extra_model_list)/2, [dataset_name_dict[x] for x in dataset_list])
    plt.xlabel("Dataset")
    # ax_list[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.1), title_fontsize=14, fontsize=10, ncol=3)
    ax_list[0].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=11)
    plt.subplots_adjust(hspace=2.5)
    plt.tight_layout()
    plt.savefig("../res/figs/exp_metric_bar.pdf")
    plt.show()
    # -----
    inter_avg_l2 = np.asarray(inter_avg_l2)
    extra_avg_l2 = np.asarray(extra_avg_l2)
    color_list = Vivid_10.mpl_colors
    fig, ax_list = plt.subplots(1, 2, figsize=(12, 4))
    inter_width = 0.1
    x_base = np.arange(len(dataset_list)) - inter_width*len(inter_model_list)
    ax_list[0].set_title("Interpolation")
    for i in range(len(inter_model_list)):
        bar1 = ax_list[0].bar(
            x=x_base+i*inter_width,
            height=inter_avg_l2[:, i],
            width=inter_width,
            align="edge",
            color=model_colors_dict[inter_model_list[i]],
            label=mdoel_name_dict[inter_model_list[i]],
            edgecolor="k",
            linewidth=0.2,
        )
        # ax_list[0].legend(fontsize=12)
    ax_list[0].set_ylim(0.0, 50)
    ax_list[0].set_yticks([0, 20, 40, 50])
    ax_list[0].set_yticklabels(["0", "20", "40", r"$\geq$50"])
    ax_list[0].set_xticks([])
    ax_list[0].set_ylabel("L2")
    ax_list[0].set_xlabel("Dataset")
    _removeTopRightBorders(ax_list[0])
    ax_list[0].set_xticks(np.arange(len(dataset_list)) - inter_width * len(inter_model_list) / 2)
    ax_list[0].set_xticklabels([dataset_name_dict[x] for x in dataset_list])
    plt.xlabel("Dataset")
    extra_width = 0.2
    x_base = np.arange(len(dataset_list)) - extra_width * len(extra_model_list)
    ax_list[1].set_title("Extrapolation")
    for i in range(len(extra_model_list)):
        bar2 = ax_list[1].bar(
            x=x_base + i * extra_width,
            height=extra_avg_l2[:, i],
            width=extra_width,
            align="edge",
            color=model_colors_dict[extra_model_list[i]],
            label=mdoel_name_dict[extra_model_list[i]],
            edgecolor="k",
            linewidth=0.2,
        )
        # ax_list[1].legend(fontsize=12)
    # ax_list[1].set_ylim(0.0, 800)
    # ax_list[1].set_yticks([0, 200, 400, 600, 800])
    # ax_list[1].set_yticklabels(["0", "200", "400", "600", r"$\geq$800"])
    # ax_list[1].set_ylabel("Wasserstein")
    _removeTopRightBorders(ax_list[1])
    plt.xticks(np.arange(len(dataset_list)) - extra_width*len(extra_model_list)/2, [dataset_name_dict[x] for x in dataset_list])
    plt.xlabel("Dataset")
    # ax_list[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.1), title_fontsize=14, fontsize=10, ncol=3)
    # ax_list[0].legend([mdoel_name_dict[x] for x in inter_model_list], loc="center left", bbox_to_anchor=(-0.0, 0.5), fontsize=11)

    # fig.add_axes([0.0, 1.0, 1.0, 0.2])
    handles, labels = ax_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.25, 1), fontsize=12, ncol=5)

    plt.subplots_adjust(hspace=2.5)
    plt.tight_layout()
    plt.savefig("../res/figs/exp_l2_bar.pdf")
    plt.show()

    # # -----
    # fig = plt.figure(figsize=(12, 2))
    # fig.legend(handles, labels, loc=(0.25, 0.5), fontsize=12, ncol=5)
    # plt.savefig("../res/figs/exp_model_legend.pdf")
    # plt.show()


def plotMetricBarHor():
    inter_avg_ot = []
    inter_std_ot = []
    inter_avg_l2 = []
    inter_std_l2 = []
    extra_avg_ot = []
    extra_std_ot = []
    extra_avg_l2 = []
    extra_std_l2 = []
    for data_name in dataset_list:
        # interpolation
        if data_name in ["embryoid", "pancreatic"]:
            inter_split = "one_interpolation"
        if data_name in ["zebrafish", "mammalian", "drosophila", "WOT"]:
            inter_split = "three_interpolation"
        metric_res = np.load("../res/comparison/{}-{}-model_metrics.npy".format(data_name, inter_split), allow_pickle=True).item()
        tmp_ot = np.asarray([[metric_res[t][m]["global"]["ot"] if m in metric_res[t] else np.nan for m in inter_model_list] for t in metric_res])
        tmp_l2 = np.asarray([[metric_res[t][m]["global"]["l2"] if m in metric_res[t] else np.nan for m in inter_model_list] for t in metric_res])
        inter_avg_ot.append(np.nanmean(tmp_ot, axis=0))
        inter_std_ot.append(np.nanstd(tmp_ot, axis=0))
        inter_avg_l2.append(np.nanmean(tmp_l2, axis=0))
        inter_std_l2.append(np.nanstd(tmp_l2, axis=0))
        # extrapolation
        if data_name in ["embryoid", "pancreatic"]:
            extra_split = "one_forecasting"
        if data_name in ["mammalian", "drosophila", "WOT"]:
            extra_split = "three_forecasting"
        if data_name in ["zebrafish"]:
            extra_split = "two_forecasting"
        metric_res = np.load("../res/comparison/{}-{}-model_metrics.npy".format(data_name, extra_split), allow_pickle=True).item()
        tmp_ot = np.asarray([[metric_res[t][m]["global"]["ot"] if m in metric_res[t] else np.nan for m in extra_model_list] for t in metric_res])
        tmp_l2 = np.asarray([[metric_res[t][m]["global"]["l2"] if m in metric_res[t] else np.nan for m in extra_model_list] for t in metric_res])
        extra_avg_ot.append(np.nanmean(tmp_ot, axis=0))
        extra_std_ot.append(np.nanmean(tmp_ot, axis=0))
        extra_avg_l2.append(np.nanmean(tmp_l2, axis=0))
        extra_std_l2.append(np.nanstd(tmp_l2, axis=0))
    # -----
    inter_avg_ot = np.asarray(inter_avg_ot)
    extra_avg_ot = np.asarray(extra_avg_ot)
    color_list = Vivid_10.mpl_colors
    fig, ax_list = plt.subplots(1, 2, figsize=(12, 4))
    inter_width = 0.1
    x_base = np.arange(len(dataset_list)) - inter_width*len(inter_model_list)
    ax_list[0].set_title("Interpolation")
    for i in range(len(inter_model_list)):
        ax_list[0].bar(
            x=x_base+i*inter_width,
            height=inter_avg_ot[:, i],
            width=inter_width,
            align="edge",
            color=model_colors_dict[inter_model_list[i]],
            label=mdoel_name_dict[inter_model_list[i]],
            edgecolor="k",
            linewidth=0.2,
        )
        # ax_list[0].legend(fontsize=12)
    ax_list[0].set_ylim(0.0, 600)
    ax_list[0].set_yticks([0, 200, 400, 600])
    ax_list[0].set_yticklabels(["0", "200", "400", r"$\geq$600"])
    ax_list[0].set_xticks([])
    ax_list[0].set_ylabel("Wasserstein")
    ax_list[0].set_xlabel("Dataset")
    ax_list[0].set_xticks(np.arange(len(dataset_list)) - inter_width * len(inter_model_list) / 2)
    ax_list[0].set_xticklabels([dataset_name_dict[x] for x in dataset_list])
    _removeTopRightBorders(ax_list[0])
    extra_width = 0.2
    x_base = np.arange(len(dataset_list)) - extra_width * len(extra_model_list)
    ax_list[1].set_title("Extrapolation")
    for i in range(len(extra_model_list)):
        ax_list[1].bar(
            x=x_base + i * extra_width,
            height=extra_avg_ot[:, i],
            width=extra_width,
            align="edge",
            color=model_colors_dict[extra_model_list[i]],
            label=mdoel_name_dict[extra_model_list[i]],
            edgecolor="k",
            linewidth=0.2,
        )
        # ax_list[1].legend(fontsize=12)
    ax_list[1].set_ylim(0.0, 800)
    ax_list[1].set_yticks([0, 200, 400, 600, 800])
    ax_list[1].set_yticklabels(["0", "200", "400", "600", r"$\geq$800"])
    # ax_list[1].set_ylabel("Wasserstein")
    _removeTopRightBorders(ax_list[1])
    plt.xticks(np.arange(len(dataset_list)) - extra_width*len(extra_model_list)/2, [dataset_name_dict[x] for x in dataset_list])
    plt.xlabel("Dataset")
    # ax_list[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.1), title_fontsize=14, fontsize=10, ncol=3)
    # ax_list[0].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=11)
    plt.subplots_adjust(hspace=2.5)
    plt.tight_layout()
    plt.savefig("../res/figs/exp_metric_bar_hori.pdf")
    plt.show()
    # -----
    inter_avg_l2 = np.asarray(inter_avg_l2)
    extra_avg_l2 = np.asarray(extra_avg_l2)
    color_list = Vivid_10.mpl_colors
    fig, ax_list = plt.subplots(1, 2, figsize=(12, 4))
    inter_width = 0.1
    x_base = np.arange(len(dataset_list)) - inter_width*len(inter_model_list)
    ax_list[0].set_title("Interpolation")
    for i in range(len(inter_model_list)):
        bar1 = ax_list[0].bar(
            x=x_base+i*inter_width,
            height=inter_avg_l2[:, i],
            width=inter_width,
            align="edge",
            color=model_colors_dict[inter_model_list[i]],
            label=mdoel_name_dict[inter_model_list[i]],
            edgecolor="k",
            linewidth=0.2,
        )
        # ax_list[0].legend(fontsize=12)
    ax_list[0].set_ylim(0.0, 50)
    ax_list[0].set_yticks([0, 20, 40, 50])
    ax_list[0].set_yticklabels(["0", "20", "40", r"$\geq$50"])
    ax_list[0].set_xticks([])
    ax_list[0].set_ylabel("L2")
    ax_list[0].set_xlabel("Dataset")
    _removeTopRightBorders(ax_list[0])
    ax_list[0].set_xticks(np.arange(len(dataset_list)) - inter_width * len(inter_model_list) / 2)
    ax_list[0].set_xticklabels([dataset_name_dict[x] for x in dataset_list])
    plt.xlabel("Dataset")
    extra_width = 0.2
    x_base = np.arange(len(dataset_list)) - extra_width * len(extra_model_list)
    ax_list[1].set_title("Extrapolation")
    for i in range(len(extra_model_list)):
        bar2 = ax_list[1].bar(
            x=x_base + i * extra_width,
            height=extra_avg_l2[:, i],
            width=extra_width,
            align="edge",
            color=model_colors_dict[extra_model_list[i]],
            label=mdoel_name_dict[extra_model_list[i]],
            edgecolor="k",
            linewidth=0.2,
        )
        # ax_list[1].legend(fontsize=12)
    ax_list[1].set_ylim(0.0, 50)
    ax_list[1].set_yticks([0, 20, 40, 50])
    ax_list[1].set_yticklabels(["0", "20", "40", r"$\geq$50"])
    _removeTopRightBorders(ax_list[1])
    plt.xticks(np.arange(len(dataset_list)) - extra_width*len(extra_model_list)/2, [dataset_name_dict[x] for x in dataset_list])
    plt.xlabel("Dataset")
    # ax_list[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.1), title_fontsize=14, fontsize=10, ncol=3)
    # ax_list[0].legend([mdoel_name_dict[x] for x in inter_model_list], loc="center left", bbox_to_anchor=(-0.0, 0.5), fontsize=11)

    # fig.add_axes([0.0, 1.0, 1.0, 0.2])
    handles, labels = ax_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.25, 1), fontsize=12, ncol=5)

    plt.subplots_adjust(hspace=2.5)
    plt.tight_layout()
    plt.savefig("../res/figs/exp_l2_bar_hori.pdf")
    plt.show()


if __name__ == '__main__':
    data_name = "mammalian"  # zebrafish, mammalian, drosophila, WOT, embryoid, pancreatic
    split_type = "three_forecasting"  # three_interpolation, three_forecasting, one_interpolation, one_forecasting, two_forecasting
    print("[ {}-{} ] Compare Predictions".format(data_name, split_type))
    # model_list = ["latent_ODE_OT_pretrain", "PRESCIENT", "WOT", "dummy", "FNN", "TrajectoryNet"]
    if split_type == "three_interpolation":
        # model_list = ["latent_ODE_OT_pretrain", "PRESCIENT", "WOT", "FNN", "dummy"]
        model_list = ["latent_ODE_OT_pretrain", "PRESCIENT", "WOT", "dummy"]
    elif split_type == "three_forecasting":
        # model_list = ["latent_ODE_OT_pretrain", "PRESCIENT", "FNN", "dummy"]
        model_list = ["latent_ODE_OT_pretrain", "PRESCIENT", "dummy"]
    elif split_type == "one_forecasting":
        # model_list = ["latent_ODE_OT_pretrain", "PRESCIENT", "FNN", "dummy"]
        model_list = ["latent_ODE_OT_pretrain", "PRESCIENT", "dummy"]
    elif split_type == "one_interpolation":
        # model_list = ["latent_ODE_OT_pretrain", "PRESCIENT", "WOT", "TrajectoryNet", "FNN", "dummy"]
        model_list = ["latent_ODE_OT_pretrain", "PRESCIENT", "WOT", "TrajectoryNet", "dummy"]
    elif split_type == "two_forecasting":
        # model_list = ["latent_ODE_OT_pretrain", "PRESCIENT", "FNN", "dummy"]
        model_list = ["latent_ODE_OT_pretrain", "PRESCIENT", "dummy"]
    print("Loading data...")
    model_true_data, model_pred_data, tps, test_tps = loadModelPrediction(data_name, split_type, model_list)
    true_cell_tps = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(model_true_data)])
    model_cell_tps = [
        np.concatenate([np.repeat(t, m_pred[t].shape[0]) for t in range(len(m_pred))])
        for m_pred in model_pred_data
    ]
    # # ----
    # embed_name = "pca_umap" # pca, phate, umap, pca_umap
    # if data_name in ["drosophila"]:
    #     embed_name = "umap"
    # print("Computing {} embeddings...".format(embed_name))
    # true_umap_traj, model_pred_umap_traj = computeVisEmbedding(model_true_data, model_pred_data, embed_name=embed_name)
    # np.save(
    #     "../res/low_dim/{}-{}-{}.npy".format(data_name, split_type, embed_name),
    #     {
    #         "true": true_umap_traj,
    #         "pred": model_pred_umap_traj,
    #         "model": model_list,
    #         "embed_name": embed_name,
    #         "true_cell_tps": true_cell_tps,
    #         "model_cell_tps": model_cell_tps,
    #         "test_tps": test_tps,
    #     }
    # )
    # ----
    # res = np.load("../res/low_dim/zebrafish-three_interpolation-pca_umap.npy", allow_pickle=True).item()
    # true_umap_traj = res["true"]
    # model_pred_umap_traj = res["pred"]
    # model_list = res["model"]
    # embed_name = res["embed_name"]
    # true_cell_tps = res["true_cell_tps"]
    # model_cell_tps = res["model_cell_tps"]
    # test_tps = res["test_tps"]
    # print("Visualization")
    # compareUMAPTestTime(true_umap_traj, model_pred_umap_traj, true_cell_tps, model_cell_tps, test_tps, model_list)
    # # -----
    # n_sim_cells = 2000
    # test_eval_metric, basic_stats = computeMetric(model_true_data, model_pred_data, tps, test_tps, model_list, n_sim_cells)
    # np.save("../res/comparison/{}-{}-model_metrics.npy".format(data_name, split_type), test_eval_metric)
    # np.save("../res/comparison/{}-{}-model_basic_stats.npy".format(data_name, split_type), basic_stats)
    # -----
    # plotMetricBar()
    plotMetricBarHor()

