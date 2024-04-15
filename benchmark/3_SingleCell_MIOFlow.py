'''
Description:
    Run MIOFlow on single-cell data.
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy
import torch
import sys

sys.path.append("MIOFlow_model_revised")

from baseline.MIOFlow_model_revised.running import trainModel, makeSimulation
from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, tunedMIOFlowPars
from plotting.visualization import plotPredAllTime, plotPredTestTime, computeDrift, plotStream, plotStreamByCellType
from plotting.PlottingUtils import umapWithPCA, computeLatentEmbedding
from sklearn.decomposition import PCA
from optim.evaluation import globalEvaluation

# ======================================================

print("=" * 70)
# Specify the dataset: zebrafish, drosophila, wot
# Representing ZB, DR, SC, repectively
data_name= "zebrafish"
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
print("# tps={}, # genes={}".format(n_tps, n_genes))
print("Train tps={}".format(train_tps))
print("Test tps={}".format(test_tps))

data = ann_data.X
true_data = [data[np.where(cell_tps == t)[0], :] for t in range(1, n_tps + 1)]  # (# tps, # cells, # genes)
true_cell_tps = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(true_data)])
train_data = [true_data[t] for t in train_tps]
test_data = [true_data[t] for t in test_tps]
print("All shape: ", [each.shape[0] for each in true_data])
print("Train shape: ", [each.shape[0] for each in train_data])
print("Test shape: ", [each.shape[0] for each in test_data])
# Convert to principal components w/ training timepoints
print("PCA...")
n_pcs = 50
pca_model = PCA(n_components=n_pcs, svd_solver="arpack")
data_pca = pca_model.fit_transform(np.concatenate(train_data, axis=0))
time_df = pd.DataFrame(
    data=np.concatenate([np.repeat(train_tps[t], x.shape[0]) for t, x in enumerate(train_data)])[:, np.newaxis],
    columns=["samples"]
)
data_df = pd.DataFrame(data=data_pca)
train_df = pd.concat([time_df, data_df], axis=1).reset_index(drop=True)
n_genes = 50
# ======================================================
# Model training
print("=" * 70)
print("Model Training...")
gae_embedded_dim, encoder_layers, layers, lambda_density = tunedMIOFlowPars(data_name, split_type)
model, gae_losses, local_losses, batch_losses, globe_losses, opts = trainModel(
    train_df, train_tps, test_tps, n_genes=n_genes, n_epochs_emb=1000, samples_size_emb=(50,),
    gae_embedded_dim=gae_embedded_dim, encoder_layers=encoder_layers,
    layers=layers, lambda_density=lambda_density,
    batch_size=100, n_local_epochs=40, n_global_epochs=40, n_post_local_epochs=0
)
# Model prediction
print("=" * 70)
print("Model Predicting...")
tps = list(range(n_tps))
generated = makeSimulation(train_df, model, tps, opts, n_sim_cells=2000, n_trajectories=100, n_bins=100)
forward_recon_traj = [pca_model.inverse_transform(generated[i, :, :]) for i in range(generated.shape[0])]
# ======================================================
# Visualization - 2D UMAP embeddings
print("Compare true and reconstructed data...")
true_cell_tps = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(true_data)])
pred_cell_tps = np.concatenate(
    [np.repeat(t, forward_recon_traj[t].shape[0]) for t in range(len(forward_recon_traj))]
)
reorder_pred_data = forward_recon_traj
true_umap_traj, umap_model, pca_model = umapWithPCA(np.concatenate(true_data, axis=0), n_neighbors=50, min_dist=0.1, pca_pcs=50)
pred_umap_traj = umap_model.transform(pca_model.transform(np.concatenate(reorder_pred_data, axis=0)))
plotPredAllTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps)
plotPredTestTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps, test_tps.detach().numpy())

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
# res_filename = "{}/{}-{}-MIOFlow-res.npy".format(save_dir, data_name, split_type)
# state_filename = "{}/{}-{}-MIOFlow-state_dict.pt".format(save_dir, data_name, split_type)
# print("Saving to {}".format(res_filename))
# np.save(
#     res_filename,
#     {"true": true_data,
#      "pred": forward_recon_traj,
#      "tps": {"all": tps, "train": train_tps, "test": test_tps},
#      "gae_losses": gae_losses,
#      "local_losses": local_losses,
#      "batch_losses": batch_losses,
#      "globe_losses": globe_losses,
#      },
#     allow_pickle=True
# )
# torch.save(model.state_dict(), state_filename)