'''
Description:
    Run PRESCIENT on single-cell data.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import numpy as np
from datetime import datetime
import torch
import sys
sys.path.append("../")
sys.path.append("../baseline/")
sys.path.append("../baseline/prescient_model/")
sys.path.append("../baseline/prescient_model/prescient")
from plotting.visualization import plotPredAllTime, plotPredTestTime
from plotting.PlottingUtils import umapWithPCA
from plotting.Compare_SingleCell_Predictions import globalEvaluation
from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, tunedPRESCIENTPars
from baseline.prescient_model.process_data import main as prepare_data
from baseline.prescient_model.running import prescientTrain, prescientSimulate

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

# ======================================================

# Load data
print("=" * 70)
# Specify the dataset: zebrafish, drosophila, wot
# Representing ZB, DR, SC, repectively
data_name= "zebrafish" # zebrafish, drosophila, wot
print("[ {} ]".format(data_name).center(60))
# Specify the type of prediction tasks: three_interpolation, two_forecasting, three_forecasting, remove_recovery
# The tasks feasible for each dataset:
#   zebrafish (ZB): three_interpolation, two_forecasting, remove_recovery
#   drosophila (DR): three_interpolation, three_forecasting, remove_recovery
#   wot (SC): three_interpolation, three_forecasting, remove_recovery
# They denote easy, medium, and hard tasks respectively.
split_type = "three_interpolation"
print("Split type: {}".format(split_type))
ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name, split_type)
train_tps, test_tps = tpSplitInd(data_name, split_type)
data = ann_data.X
all_tps = list(np.arange(n_tps))

processed_data = ann_data[
    [t for t in range(ann_data.shape[0]) if (ann_data.obs["tp"].values[t] - 1.0) in train_tps]
]
cell_tps = (processed_data.obs["tp"].values - 1.0).astype(int)
cell_types = np.repeat("NAN", processed_data.shape[0])
genes = processed_data.var.index.values

# Use tuned hyperparameters
k_dim, layers, sd, tau, clip = tunedPRESCIENTPars(data_name, split_type)

# PCA and construct data dict
data_dict, scaler, pca, um = prepare_data(
    processed_data.X, cell_tps, cell_types, genes,
    num_pcs=k_dim, num_neighbors_umap=10
)
data_dict["y"] = list(range(train_tps[-1] + 1))
data_dict["x"] = [data_dict["x"][train_tps.index(t)] if t in train_tps else [] for t in data_dict["y"]]
data_dict["xp"] = [data_dict["xp"][train_tps.index(t)] if t in train_tps else [] for t in data_dict["y"]]
data_dict["w"] = [data_dict["w"][train_tps.index(t)] if t in train_tps else [] for t in data_dict["y"]]

# ======================================================
# Model training
train_epochs = 2000
train_lr = 1e-3
final_model, best_state_dict, config, loss_list = prescientTrain(
    data_dict, data_name=data_name, out_dir="", train_t=train_tps[1:], timestamp=timestamp,
    k_dim=k_dim, layers=layers, train_epochs=train_epochs, train_lr=train_lr,
    train_sd=sd, train_tau=tau, train_clip=clip
)

# Simulation
n_sim_cells = 2000
sim_data = prescientSimulate(
    data_dict,
    data_name=data_name,
    best_model_state=best_state_dict,
    num_cells=n_sim_cells,
    num_steps= (n_tps - 1)*10, # (#tps - 1) * 10, as dt=0.1
    config=config
)
sim_tp_latent = [sim_data[int(t * 10)] for t in range(len(all_tps))]  # dt=0.1 in all cases
sim_tp_recon = [scaler.inverse_transform(pca.inverse_transform(each)) for each in sim_tp_latent]

# ======================================================
# Visualization - 2D UMAP embeddings
traj_data = [ann_data.X[(ann_data.obs["tp"].values - 1.0) == t] for t in range(len(all_tps))]
all_recon_obs = sim_tp_recon
print("Compare true and reconstructed data...")
true_data = traj_data
true_cell_tps = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(true_data)])
pred_cell_tps = np.concatenate([np.repeat(t, all_recon_obs[t].shape[0]) for t in range(len(all_recon_obs))])
reorder_pred_data = all_recon_obs

true_umap_traj, umap_model, pca_model = umapWithPCA(np.concatenate(true_data, axis=0), n_neighbors=50, min_dist=0.1, pca_pcs=50)
pred_umap_traj = umap_model.transform(pca_model.transform(np.concatenate(reorder_pred_data, axis=0)))

plotPredAllTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps)
plotPredTestTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps, test_tps)

# Compute evaluation metrics
print("Compute metrics...")
test_tps_list = [int(t) for t in test_tps]
for t in test_tps_list:
    print("-" * 70)
    print("t = {}".format(t))
    # -----
    pred_global_metric = globalEvaluation(traj_data[t], reorder_pred_data[t])
    print(pred_global_metric)

# # ======================================================
# # Save results
# save_dir = "../res/single_cell/experimental/{}".format(data_name)
# res_filename="{}/{}-{}-PRESCIENT-res.npy".format(save_dir, data_name, split_type)
# print("Saving to {}".format(res_filename))
# res_dict = {
#     "true": [ann_data.X[(ann_data.obs["tp"].values - 1.0) == t] for t in range(len(all_tps))],
#     "pred": sim_tp_recon,
#     "latent_seq": sim_tp_latent,
#     "tps": {"all": all_tps, "train": train_tps, "test": test_tps},
#     }
# res_dict["true_pca"] = [pca.transform(scaler.transform(each)) for each in res_dict["true"]]
# np.save(res_filename, res_dict, allow_pickle=True)
#
# # save model and config
# model_dir="{}/{}-{}-PRESCIENT-state_dict.pt".format(save_dir, data_name, split_type)
# config_dir="/{}/{}-{}-PRESCIENT-config.pt".format(save_dir, data_name, split_type)
# torch.save(best_state_dict['model_state_dict'], model_dir)
# torch.save(config.__dict__, config_dir)
