'''
Description:
    Run MIOFlow on extrapolating multiple timepoints.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy
import torch
import sys


from baseline.MIOFlow_model_revised.running import trainModel, makeSimulation
from model_validation.extrapolation.utils import tunedMIOFlowPars, loadSCData
from sklearn.decomposition import PCA

# ======================================================

import argparse
main_parser = argparse.ArgumentParser()
main_parser.add_argument('-d', default="zebrafish", type=str, help="The data name.")
main_parser.add_argument('-i', default="2", type=str, help="The number of extrapolating timepoints (1-5)")
args = main_parser.parse_args()
forecast_num = int(args.i)
data_name = args.d

# ======================================================
# Load data and pre-processing
print("=" * 70)
# The data are available at https://doi.org/10.6084/m9.figshare.25607679s
if data_name == "zebrafish":
    data_dir = "./extrapolate_data/zebrafish_embryonic/"
elif data_name == "drosophila":
    data_dir = "./extrapolate_data/drosophila_embryonic/"
elif data_name == "wot":
    data_dir = "./extrapolate_data/Schiebinger2019/"
else:
    raise ValueError("Unknown data name {}!".format(data_name))
split_type = "forecasting_{}".format(forecast_num)
print("[ {} ]".format(data_name).center(60))
print("Split type: {}".format(split_type))
ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name, split_type, data_dir=data_dir)
train_tps = list(range(n_tps-forecast_num))
test_tps = list(range(n_tps-forecast_num, n_tps))
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
# =======================
# Model training
print("=" * 70)
print("Model Training...")
gae_embedded_dim, encoder_layers, layers, lambda_density = tunedMIOFlowPars(data_name, forecast_num)
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
# -----
res_filename = "./res/extrapolation/{}-{}-MIOFlow-res.npy".format(data_name, split_type)
state_filename = "./res/extrapolation/{}-{}-MIOFlow-state_dict.pt".format(data_name, split_type)
print("Saving to {}".format(res_filename))
np.save(
    res_filename,
    {"true": true_data,
     "pred": forward_recon_traj,
     "tps": {"all": tps, "train": train_tps, "test": test_tps},
     "gae_losses": gae_losses,
     "local_losses": local_losses,
     "batch_losses": batch_losses,
     "globe_losses": globe_losses,
     },
    allow_pickle=True
)
torch.save(model.state_dict(), state_filename)
