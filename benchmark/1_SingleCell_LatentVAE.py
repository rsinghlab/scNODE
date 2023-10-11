'''
Description:
    Run our model on the single-cell dataset.
'''
import matplotlib.pyplot as plt
import torch
import torch.distributions as dist
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import itertools

from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, tunedOurPars
from plotting.visualization import plotUMAP, plotUMAPTimePoint, plotUMAPTestTime, umapWithoutPCA, umapWithPCA
from data.preprocessing import splitBySpec
from benchmark.Compare_SingleCell_Predictions import basicStats, globalEvaluation
from optim.running import constructLatentODEModel, latentODETrainWithPreTrain, latentODESimulate


# ======================================================
# Load data and pre-processing
print("=" * 70)
data_name= "wot" #zebrafish, mammalian, drosophila, wot, pancreatic, embryoid
print("[ {} ]".format(data_name).center(60))
split_type = "three_interpolation"  # three_interpolation, three_forecasting, one_interpolation, one_forecasting
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
# Construct model
pretrain_iters = 200
pretrain_lr = 1e-3
latent_coeff = 1.0
epochs = 10
iters = 100
batch_size = 32
lr = 1e-3
act_name = "relu"
n_sim_cells = 2000

latent_dim, drift_latent_size, enc_latent_list, dec_latent_list = tunedOurPars(data_name, split_type)
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

# ======================================================
# Save results
res_filename = "../res/single_cell/experimental/{}/{}-{}-latent_ODE_OT_pretrain-res.npy".format(data_name, data_name, split_type)
state_filename = "../res/single_cell/experimental/{}/{}-{}-latent_ODE_OT_pretrain-state_dict.pt".format(data_name, data_name, split_type)
print("Saving to {}".format(res_filename))
np.save(
    res_filename,
    {"true": [each.detach().numpy() for each in traj_data],
     "pred": [all_recon_obs[:, t, :] for t in range(all_recon_obs.shape[1])],
     "first_latent_dist": first_latent_dist,
     "latent_seq": latent_seq,
     "tps": {"all": tps.detach().numpy(), "train": train_tps.detach().numpy(), "test": test_tps.detach().numpy()},
     "loss": loss_list,
     },
    allow_pickle=True
)
torch.save(latent_ode_model.state_dict(), state_filename)


# # ======================================================
# # Visualization
# plt.figure(figsize=(8, 6))
# plt.subplot(3, 1, 1)
# plt.title("Loss")
# plt.plot([each[0] for each in loss_list])
# plt.subplot(3, 1, 2)
# plt.title("OT Term")
# plt.plot([each[1] for each in loss_list])
# plt.subplot(3, 1, 3)
# plt.title("Latent Difference")
# plt.plot([each[2] for each in loss_list])
# plt.xlabel("Dynamic Learning Iter")
# plt.show()
#
# print("Compare true and reconstructed data...")
# true_data = [each.detach().numpy() for each in traj_data]
# true_cell_tps = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(true_data)])
# pred_cell_tps = np.concatenate([np.repeat(t, all_recon_obs[:, t, :].shape[0]) for t in range(all_recon_obs.shape[1])])
# reorder_pred_data = [all_recon_obs[:, t, :] for t in range(all_recon_obs.shape[1])]
#
# true_umap_traj, umap_model, pca_model = umapWithPCA(np.concatenate(true_data, axis=0), n_neighbors=50, min_dist=0.1, pca_pcs=50)
# pred_umap_traj = umap_model.transform(pca_model.transform(np.concatenate(reorder_pred_data, axis=0)))
#
# plotUMAPTimePoint(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps)
# plotUMAPTestTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps, test_tps.detach().numpy())
#
# print("Visualize latent space...")
# from downstream_analysis.Visualize_Velocity import computeDrift, computeEmbedding, plotStreamByCellType, plotStream
# from plotting.utils import linearSegmentCMap
# latent_seq, next_seq, drift_seq = computeDrift(traj_data, latent_ode_model)
# drift_magnitude = [np.linalg.norm(each, axis=1) for each in drift_seq]
# umap_latent_data, umap_next_data, umap_model, latent_tp_list = computeEmbedding(
#     latent_seq, next_seq, n_neighbors = 50, min_dist = 0.1 # 0.25
# )
# umap_scatter_data = umap_latent_data
# color_list = linearSegmentCMap(n_tps, "viridis")
# plotStream(umap_scatter_data, umap_latent_data, umap_next_data, color_list, num_sep=200, n_neighbors=20)
# if cell_types is not None:
#     plotStreamByCellType(umap_scatter_data, umap_latent_data, umap_next_data, traj_cell_types, num_sep=200, n_neighbors=5)
#
# # Compute metric for testing time points
# print("Compute metrics...")
# test_tps_list = [int(t) for t in test_tps]
# for t in test_tps_list:
#     print("-" * 70)
#     print("t = {}".format(t))
#     # -----
#     pred_global_metric = globalEvaluation(traj_data[t].detach().numpy(), all_recon_obs[:, t, :])
#     print(pred_global_metric)



