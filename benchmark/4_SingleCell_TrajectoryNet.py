'''
Description:
    Run TrajectoryNet on single-cell data.
    Notice TrajectoryNet can only predict one interpolating timepoints.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import numpy as np
import torch

from baseline.TrajectoryNet_model.running import TrajectoryNetTrain, TrajectoryNetSimulate
from benchmark.BenchmarkUtils import tunedTrjectoryNetPars
from plotting.visualization import plotPredAllTime, plotPredTestTime
from plotting.PlottingUtils import umapWithPCA
from plotting.Compare_SingleCell_Predictions import globalEvaluation

# ======================================================

# Parameter settings
split_type = "one_interpolation"
data_name = "pancreatic"  # pancreatic, embryoid

if data_name == "embryoid":
    leaveout_timepoint = 2
    train_tps = [0, 1, 3, 4]
    test_tps = [2]
    n_tps = 5
    # dir_path = "../data/single_cell/experimental/embryoid_body/processed/"
    dir_path = "../../sc_Dynamic_Modelling/data/single_cell/experimental/embryoid_body/processed/"
elif data_name == "pancreatic":
    leaveout_timepoint = 2
    train_tps = [0, 1, 3]
    test_tps = [2]
    n_tps = 4
    # dir_path = "../data/single_cell/experimental/mouse_pancreatic/processed/"
    dir_path = "../../sc_Dynamic_Modelling/data/single_cell/experimental/mouse_pancreatic/processed/"
else:
    raise ValueError("Unknown data name {}!".format(data_name))

# ======================================================

n_iters = 500
batch_size = 100
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

ann_data = args.data.true_data
true_data = [ann_data.X[(ann_data.obs["tp"].values - 1.0) == t] for t in range(n_tps)]

# ======================================================
# Visualization
pred_cell_tps = np.concatenate(
    [np.repeat(t, forward_recon_traj[t].shape[0]) for t in range(len(forward_recon_traj))]
)
true_cell_tps = np.concatenate(
    [np.repeat(t, true_data[t].shape[0]) for t in range(len(true_data))]
)
reorder_pred_data = forward_recon_traj

true_umap_traj, umap_model, pca_model = umapWithPCA(np.concatenate(true_data, axis=0), n_neighbors=50, min_dist=0.1, pca_pcs=50)
pred_umap_traj = umap_model.transform(pca_model.transform(np.concatenate(reorder_pred_data, axis=0)))

plotPredAllTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps)
plotPredTestTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps, test_tps)

# Compute metric for testing time points
print("Compute metrics...")
test_tps_list = [int(t) for t in test_tps]
for t in test_tps_list:
    print("-" * 70)
    print("t = {}".format(t))
    # -----
    pred_global_metric = globalEvaluation(true_data[t], reorder_pred_data[t])
    print(pred_global_metric)

# # ======================================================
# # Save results
# save_dir = "../res/single_cell/experimental/{}".format(data_name)
# res_filename="{}/{}-{}-TrajectoryNet-res.npy".format(save_dir, data_name, split_type)
# print("Saving to {}".format(res_filename))
# res_dict = {
#     "true": true_data,
#     "pred": forward_recon_traj,
#     "latent_seq": forward_latent_traj,
#     "tps": {"all": list(range(n_tps)), "train": train_tps, "test": test_tps},
#     }
# np.save(res_filename, res_dict, allow_pickle=True)
# # save model and config
# model_dir="{}/{}-{}-TrajectoryNet-state_dict.pt".format(save_dir, data_name, split_type)
# config_dir="{}/{}-{}-TrajectoryNet-config.pt".format(save_dir, data_name, split_type)
# torch.save(model.state_dict(), model_dir)
