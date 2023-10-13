'''
Description:
    Run TrajectoryNet on single-cell data.
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy
import torch
import sys
sys.path.append("../TrajectoryNet_model/")

from TrajectoryNet_model.running import TrajectoryNetTrain, TrajectoryNetSimulate
from benchmark.BenchmarkUtils import loadEmbryoidData, loadPancreaticData, tunedTrjectoryNetPars
from plotting.visualization import plotUMAP, plotPredAllTime, plotPredTestTime, umapWithoutPCA, umapWithPCA

# ======================================================

# Parameter settings
split_type = "one_interpolation"
data_name = "pancreatic"  # pancreatic, embryoid

if data_name == "embryoid":
    leaveout_timepoint = 2
    train_tps = [0, 1, 3, 4]
    test_tps = [2]
    n_tps = 5
    dir_path = "../data/single_cell/experimental/embryoid_body/processed/"
elif data_name == "pancreatic":
    leaveout_timepoint = 2
    train_tps = [0, 1, 3]
    test_tps = [2]
    n_tps = 4
    dir_path = "../data/single_cell/experimental/mouse_pancreatic/processed/"
else:
    raise ValueError("Unknown data name {}!".format(data_name))

# ======================================================

n_iters = 500 # 1000
batch_size = 100  # 1000
n_pcs, top_k_reg, vecint = tunedTrjectoryNetPars(data_name, split_type)
# Model training
print("Model training...")
args, model, pca_model, scalar_model = TrajectoryNetTrain(
    data_name, split_type, dir_path, n_pcs, n_tps, leaveout_timepoint, n_iters, batch_size, top_k_reg=top_k_reg, vecint=vecint
)
# Simulation
num_sim_cells = 2000
print("Simulation...")
forward_recon_traj, forward_latent_traj = TrajectoryNetSimulate(
    args, model, n_tps, pca_model, scalar_model
)
print("Pred cell num: ", [each.shape[0] for each in forward_recon_traj])
# for t in test_tps:
#     rand_idx =  np.random.choice(np.arange(forward_recon_traj[t].shape[0]), num_sim_cells, replace=False)
#     forward_recon_traj[t] =  forward_recon_traj[t][rand_idx,:]
# print("Sampled pred cell num: ", [each.shape[0] for each in forward_recon_traj])

# ======================================================
# Save results
res_filename="../res/single_cell/experimental/{}/{}-{}-TrajectoryNet-res.npy".format(data_name, data_name, split_type)
print("Saving to {}".format(res_filename))
ann_data = args.data.true_data
true_data = [ann_data.X[(ann_data.obs["tp"].values - 1.0) == t] for t in range(n_tps)]
res_dict = {
    "true": true_data,
    "pred": forward_recon_traj,
    "latent_seq": forward_latent_traj,
    "tps": {"all": list(range(n_tps)), "train": train_tps, "test": test_tps},
    }
np.save(res_filename, res_dict, allow_pickle=True)

# save model and config
model_dir="../res/single_cell/experimental/{}/{}-{}-TrajectoryNet-state_dict.pt".format(data_name, data_name, split_type)
config_dir="../res/single_cell/experimental/{}/{}-{}-TrajectoryNet-config.pt".format(data_name, data_name, split_type)
torch.save(model.state_dict(), model_dir)

# ======================================================

# # Visualization
# pred_cell_tps = np.concatenate(
#     [np.repeat(t, forward_recon_traj[t].shape[0]) for t in range(len(forward_recon_traj))]
# )
# reorder_pred_data = forward_recon_traj
#
# true_umap_traj, umap_model, pca_model = umapWithPCA(
#     np.concatenate(true_data, axis=0),
#     n_neighbors=50,
#     min_dist=0.1,
#     pca_pcs=50
# )
# pred_umap_traj = umap_model.transform(pca_model.transform(np.concatenate(reorder_pred_data, axis=0)))
#
# plotUMAPTimePoint(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps)
# plotUMAPTestTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps, test_tps)
