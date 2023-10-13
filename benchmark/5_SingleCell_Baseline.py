'''
Description:
    Run baseline models on the single-cell dataset.
'''
import matplotlib.pyplot as plt
import torch
import torch.distributions as dist
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import itertools

from benchmark.BenchmarkUtils import loadSCData, tpSplitInd
from plotting.visualization import plotUMAP, plotPredAllTime, plotPredTestTime, umapWithoutPCA, umapWithPCA
from data.preprocessing import splitBySpec
from benchmark.Compare_SingleCell_Predictions import globalEvaluation
from optim.running import constructDummyModel, constructFNNModel, dummySimulate, FNNTrain, FNNSimulate


# ======================================================
# Load data and pre-processing
print("=" * 70)
data_name= "zebrafish" #zebrafish, mammalian, drosophila, wot, pancreatic, embryoid
print("[ {} ]".format(data_name).center(60))
split_type = "two_forecasting"  # three_interpolation, three_forecasting, one_interpolation, one_forecasting
print("Split type: {}".format(split_type))
ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name, split_type)
train_tps, test_tps = tpSplitInd(data_name, split_type)
data = ann_data.X

# Convert to torch project
traj_data = [torch.FloatTensor(data[np.where(cell_tps == t)[0], :]) for t in range(1, n_tps + 1)]  # (# tps, # cells, # genes)
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

# ======================================================
# Model running

model_name = "FNN" # dummy, FNN
if model_name == "dummy":
    dummy_model = constructDummyModel()
    all_recon_obs = dummySimulate(dummy_model, traj_data, train_tps, know_all=False)
    loss_list = None
elif model_name == "FNN":
    fnn_model = constructFNNModel(n_genes, latent_size=[50], act_name="relu")
    fnn_model, loss_list, all_recon = FNNTrain(traj_data, train_tps, fnn_model, iters=500, lr=1e-3, batch_size=32)
    all_recon_obs = FNNSimulate(traj_data, train_tps, fnn_model)
else:
    raise ValueError("Unknown model name {}!".format(model_name))

# ======================================================
# Save results
res_filename = "../res/single_cell/experimental/{}/{}-{}-{}-res.npy".format(data_name, data_name, split_type, model_name)
state_filename = "../res/single_cell/experimental/{}/{}-{}-{}-state_dict.pt".format(data_name, data_name, split_type, model_name)
print("Saving to {}".format(res_filename))
np.save(
    res_filename,
    {"true": [each.detach().numpy() for each in traj_data],
     "pred": all_recon_obs,
     "tps": {"all": tps.detach().numpy(), "train": train_tps.detach().numpy(), "test": test_tps.detach().numpy()},
     "loss": loss_list,
     },
    allow_pickle=True
)
if model_name == "FNN":
    torch.save(fnn_model.state_dict(), state_filename)

# # ======================================================
# # Visualization
# if loss_list is not None:
#     plt.figure(figsize=(6, 4))
#     plt.plot(loss_list)
#     plt.show()
#
#
# print("Compare true and reconstructed data...")
# true_data = [each.detach().numpy() for each in traj_data]
# true_cell_tps = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(true_data)])
# pred_cell_tps = np.concatenate([np.repeat(t, all_recon_obs[t].shape[0]) for t in range(n_tps)])
# reorder_pred_data = all_recon_obs
#
# true_umap_traj, umap_model, pca_model = umapWithPCA(np.concatenate(true_data, axis=0), n_neighbors=50, min_dist=0.1, pca_pcs=50)
# pred_umap_traj = umap_model.transform(pca_model.transform(np.concatenate(reorder_pred_data, axis=0)))
#
# plotUMAPTimePoint(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps)
# plotUMAPTestTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps, test_tps.detach().numpy())
#
#
#
# # Compute metric for testing time points
# print("Compute metrics...")
# test_tps_list = [int(t) for t in test_tps]
# for t in test_tps_list:
#     print("-" * 70)
#     print("t = {}".format(t))
#     # -----
#     pred_global_metric = globalEvaluation(traj_data[t].detach().numpy(), all_recon_obs[t])
#     print(pred_global_metric)
