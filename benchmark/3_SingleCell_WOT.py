'''
Description:
    Run WOT on single-cell data for interpolations.
'''

import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import itertools
import scanpy

from wot_model.running import wotSimulate
from plotting.visualization import plotUMAP, plotUMAPTimePoint, plotUMAPTestTime, umapWithoutPCA, umapWithPCA
from benchmark.Compare_SingleCell_Predictions import basicStats, globalEvaluation
from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, tunedWOTPars

# ======================================================

# Load data
print("=" * 70)
data_name = "mammalian"  # zebrafish, mammalian, drosophila, wot, pancreatic, embryoid
print("[ {} ]".format(data_name).center(60))
split_type = "three_interpolation"  # three_interpolation, one_interpolation
print("Split type: {}".format(split_type))
ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name, split_type)
train_tps, test_tps = tpSplitInd(data_name, split_type)
data = ann_data.X

ann_data.obs.tp = ann_data.obs.tp - 1.0
unique_tp_list = ann_data.obs.tp.unique()
print("# of tps = {}".format(len(unique_tp_list)))
print("Train tps: ", train_tps)
print("Test tps: ", test_tps)

# ======================================================
# Run WOT model
num_sim_cells = 2000
latent_dim, epsilon, lambda1, lambda2 = tunedWOTPars(data_name, split_type)
tau=10000
growth_iters=3

train_ann_data = ann_data[np.where(ann_data.obs.tp.apply(lambda x: x in train_tps))[0], :]
traj_data = [train_ann_data[train_ann_data.obs.tp == t, :].X for t in train_tps]
all_recon_data = wotSimulate(
    traj_data, train_ann_data, n_tps, train_tps, test_tps, num_cells=num_sim_cells, tp_field="tp",
    epsilon=epsilon, lambda1=lambda1, lambda2=lambda2, tau=tau, growth_iters=growth_iters)
print("Pred cell num: ", [each.shape[0] for each in all_recon_data])

# ======================================================
# Save results
res_filename="../res/single_cell/experimental/{}/{}-{}-WOT-res.npy".format(data_name, data_name, split_type)
print("Saving to {}".format(res_filename))
res_dict = {
    "true": [ann_data.X[(ann_data.obs["tp"].values - 1.0) == t] for t in range(n_tps)],
    "pred": all_recon_data,
    "tps": {"all": list(range(n_tps)), "train": train_tps, "test": test_tps},
    }
np.save(res_filename, res_dict, allow_pickle=True)

# # ======================================================
# # Visualization
# traj_data = res_dict["true"]
# all_recon_obs = res_dict["pred"]
# print("Compare true and reconstructed data...")
# true_data = traj_data
# true_cell_tps = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(true_data)])
# pred_cell_tps = np.concatenate([np.repeat(t, all_recon_obs[t].shape[0]) for t in range(len(all_recon_obs))])
# reorder_pred_data = all_recon_obs
#
# true_umap_traj, umap_model, pca_model = umapWithPCA(np.concatenate(true_data, axis=0), n_neighbors=50, min_dist=0.1, pca_pcs=50)
# pred_umap_traj = umap_model.transform(pca_model.transform(np.concatenate(reorder_pred_data, axis=0)))
#
# plotUMAPTimePoint(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps)
# plotUMAPTestTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps, test_tps)
#
# # Compute metric for testing time points
# print("Compute metrics...")
# test_tps_list = [int(t) for t in test_tps]
# for t in test_tps_list:
#     print("-" * 70)
#     print("t = {}".format(t))
#     # -----
#     pred_global_metric = globalEvaluation(traj_data[t], reorder_pred_data[t])
#     print(pred_global_metric)