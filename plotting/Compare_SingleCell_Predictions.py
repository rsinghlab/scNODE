'''
Description:
    Compare model predictions of six scRNA-seq datasets.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy
from plotting.__init__ import *
from plotting.PlottingUtils import computeVisEmbedding
from plotting.visualization import compareUMAPTestTime, plotMetricBar, printMetric
from optim.evaluation import globalEvaluation, basicStats, LISIScore

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
            m_global_metric = globalEvaluation(true_data[t], sampled_model_pred_data[m_idx][t])
            # -----
            test_eval_metric[t][m] = {
                "global": m_global_metric
            }
    return test_eval_metric, basic_stats

# ======================================================

mdoel_name_dict = {
    "latent_ODE_OT_pretrain": "scNODE",
    "PRESCIENT": "PRESCIENT",
    "WOT": "WOT",
    "dummy": "Dummy",
    # "FNN": "FNN",
    "TrajectoryNet": "TrajectoryNet",
}

inter_model_list = ["latent_ODE_OT_pretrain", "PRESCIENT", "WOT", "TrajectoryNet", "dummy"]
extra_model_list = ["latent_ODE_OT_pretrain", "PRESCIENT", "dummy"]
dataset_name_dict = {
    "embryoid": "EB", # embryoid body
    "pancreatic": "MP", # mouse pancreatic
    "zebrafish":"ZF", # zebrafish
    "mammalian":"MB", # mouse brain
    "drosophila":"DR", # drosophila
    "WOT":"SC" # wot
}
dataset_list = ["embryoid", "pancreatic", "zebrafish", "mammalian", "drosophila", "WOT"]


if __name__ == '__main__':
    # Specify dataset and task name
    data_name = "zebrafish"  # zebrafish, mammalian, drosophila, WOT, embryoid, pancreatic
    split_type = "three_interpolation"  # three_interpolation, three_forecasting, one_interpolation, one_forecasting, two_forecasting
    embed_name = "pca_umap"
    if data_name in ["drosophila"]:
        embed_name = "umap"
    embedding_filename = "../../sc_Dynamic_Modelling/res/low_dim/{}-{}-{}.npy".format(data_name, split_type, embed_name)
    print("[ {}-{} ] Compare Predictions".format(data_name, split_type))
    if split_type == "three_interpolation":
        model_list = ["latent_ODE_OT_pretrain", "PRESCIENT", "WOT", "dummy"]
    elif split_type == "three_forecasting":
        model_list = ["latent_ODE_OT_pretrain", "PRESCIENT", "dummy"]
    elif split_type == "one_forecasting":
        model_list = ["latent_ODE_OT_pretrain", "PRESCIENT", "dummy"]
    elif split_type == "one_interpolation":
        model_list = ["latent_ODE_OT_pretrain", "PRESCIENT", "WOT", "TrajectoryNet", "dummy"]
    elif split_type == "two_forecasting":
        model_list = ["latent_ODE_OT_pretrain", "PRESCIENT", "dummy"]
    # ============================
    if not os.path.isfile(embedding_filename):
        # Load predictions and compute embedding
        print("Loading data...")
        model_true_data, model_pred_data, tps, test_tps = loadModelPrediction(data_name, split_type, model_list)
        true_cell_tps = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(model_true_data)])
        model_cell_tps = [
            np.concatenate([np.repeat(t, m_pred[t].shape[0]) for t in range(len(m_pred))])
            for m_pred in model_pred_data
        ]
        # Compute embeddings and save results
        print("Computing {} embeddings...".format(embed_name))
        true_umap_traj, model_pred_umap_traj = computeVisEmbedding(model_true_data, model_pred_data, embed_name=embed_name)
        np.save(
            embedding_filename,
            {
                "true": true_umap_traj,
                "pred": model_pred_umap_traj,
                "model": model_list,
                "embed_name": embed_name,
                "true_cell_tps": true_cell_tps,
                "model_cell_tps": model_cell_tps,
                "test_tps": test_tps,
            }
        )
    # ----
    # Compare 2D UMAP
    res = np.load(embedding_filename, allow_pickle=True).item()
    true_umap_traj = res["true"]
    model_pred_umap_traj = res["pred"]
    model_list = res["model"]
    embed_name = res["embed_name"]
    true_cell_tps = res["true_cell_tps"]
    model_cell_tps = res["model_cell_tps"]
    test_tps = res["test_tps"]
    print("Visualization")
    compareUMAPTestTime(
        true_umap_traj, model_pred_umap_traj, true_cell_tps, model_cell_tps, test_tps, model_list,
        mdoel_name_dict, data_name, split_type, embed_name, save_dir="./"
    )
    # ============================
    metric_filename = "../res/comparison/{}-{}-model_metrics.npy".format(data_name, split_type)
    n_sim_cells = 2000
    if not os.path.isfile(metric_filename):
        # Load predictions and compute evaluation metrics
        model_true_data, model_pred_data, tps, test_tps = loadModelPrediction(data_name, split_type, model_list)
        true_cell_tps = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(model_true_data)])
        model_cell_tps = [
            np.concatenate([np.repeat(t, m_pred[t].shape[0]) for t in range(len(m_pred))])
            for m_pred in model_pred_data
        ]
        test_eval_metric, basic_stats = computeMetric(model_true_data, model_pred_data, tps, test_tps, model_list, n_sim_cells)
        np.save("../res/comparison/{}-{}-model_metrics.npy".format(data_name, split_type), test_eval_metric)
    # -----
    metric_dict = np.load(metric_filename, allow_pickle=True).item()
    test_tps = list(metric_dict.keys())
    n_test_tps = len(test_tps)
    column_names = [("t", t) for t in test_tps]
    # Compare pair-wise L2 dist and Wasserstein distance (OT)
    model_l2 = [[metric_dict[t][m]["global"]["l2"] for t in test_tps] for m in model_list]
    model_ot = [[metric_dict[t][m]["global"]["ot"] for t in test_tps] for m in model_list]
    printMetric(model_l2, model_ot, model_list, column_names)
    plotMetricBar("..//res/comparison", dataset_list, inter_model_list, extra_model_list, dataset_name_dict, mdoel_name_dict, save_dir="")


