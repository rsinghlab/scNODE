"""
Description:
    Computing auxiliary data (VAE latent variables at each timepoint and the corresponding UMAP visualization) for perturbation analysis.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
"""
import torch
import numpy as np
import pandas as pd
from benchmark.BenchmarkUtils import loadSCData, splitBySpec
from plotting.PlottingUtils import umapWithoutPCA
from optim.running import constructscNODEModel, scNODETrainWithPreTrain

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

# ======================================================

merge_dict = {
    "zebrafish": ["Hematopoeitic", 'Hindbrain', "Notochord", "PSM", "Placode", "Somites", "Spinal", "Neural",
                  "Endoderm"],
}


def mergeCellTypes(cell_types, merge_list):
    new_cell_types = []
    for c in cell_types:
        c_sep = c.strip(" ").split(" ")
        if c in merge_list:
            new_cell_types.append(c)
        elif c_sep[0] in merge_list:
            new_cell_types.append(c_sep[0])
        else:
            new_cell_types.append(c)
    return new_cell_types


# ======================================================

latent_dim = 50
drift_latent_size = [50]
enc_latent_list = [50]
dec_latent_list = [50]
act_name = "relu"


def learnVectorField(train_data, train_tps, tps):
    '''Use scNODE to learn cell developmental vector field.'''
    pretrain_iters = 200
    pretrain_lr = 1e-3
    latent_coeff = 1.0
    epochs = 10
    iters = 100
    batch_size = 32
    lr = 1e-3
    latent_ode_model = constructscNODEModel(
        n_genes, latent_dim=latent_dim,
        enc_latent_list=enc_latent_list, dec_latent_list=dec_latent_list, drift_latent_size=drift_latent_size,
        latent_enc_act="none", latent_dec_act=act_name, drift_act=act_name,
        ode_method="euler"
    )
    latent_ode_model, loss_list, recon_obs, first_latent_dist, latent_seq = scNODETrainWithPreTrain(
        train_data, train_tps, latent_ode_model, latent_coeff=latent_coeff, epochs=epochs, iters=iters,
        batch_size=batch_size, lr=lr, pretrain_iters=pretrain_iters, pretrain_lr=pretrain_lr
    )
    return latent_ode_model

# ======================================================

# Load data and pre-processing
print("=" * 70)
data_name = "zebrafish"  # zebrafish, mammalian
print("[ {} ]".format(data_name).center(60))
split_type = "all"  # all
print("Split type: {}".format(split_type))
ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name, split_type)
train_tps, test_tps = list(range(n_tps)), []
data = ann_data.X
if data_name in merge_dict:
    print("Merge cell types...")
    cell_types = np.asarray(mergeCellTypes(cell_types, merge_dict[data_name]))
# Convert to torch project
traj_data = [torch.FloatTensor(data[np.where(cell_tps == t)[0], :]) for t in range(1, n_tps + 1)]
if cell_types is not None:
    traj_cell_types = [cell_types[np.where(cell_tps == t)[0]] for t in range(1, n_tps + 1)]
else:
    traj_cell_types = None
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

# -----

print("Model training...")
latent_ode_model = learnVectorField(train_data, train_tps, tps)
torch.save(latent_ode_model.state_dict(), "./zebrafish-all-scNODE-state_dict.pt")

print("Computing aux data...")
# Compute VAE latent for each time point
latent_seq = []
for t_data in traj_data:
    latent_mean, latent_std = latent_ode_model.latent_encoder(t_data)
    latent_sample = latent_ode_model._sampleGaussian(latent_mean, latent_std)
    latent_seq.append(latent_sample.detach().numpy())

# Compute UMAP visualization of VAE latent variables
latent_tp_list = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(latent_seq)])
umap_latent_data, umap_model = umapWithoutPCA(np.concatenate(latent_seq, axis=0), n_neighbors=50, min_dist=0.1)
umap_latent_data = [umap_latent_data[np.where(latent_tp_list == t)[0], :] for t in range(len(latent_seq))]

# -----

np.save(
    "./zebrafish-all-scNODE-aux_data.npy",
    {
        "latent_seq": latent_seq,
        "umap_latent_data": umap_latent_data,
        "umap_model": umap_model,
    }
)
