'''
Description:
    Run scNODE on extrapolating multiple timepoints.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import torch
import numpy as np

from model_validation.extrapolation.utils import loadSCData, tunedOurPars
from benchmark.BenchmarkUtils import splitBySpec
from optim.running import constructscNODEModel, scNODETrainWithPreTrain, scNODEPredict


# ======================================================

import argparse
main_parser = argparse.ArgumentParser()
main_parser.add_argument('-d', default="zebrafish", type=str, help="The data name.")
main_parser.add_argument('-i', default="1", type=str, help="The number of extrapolating timepoints (1-5)")
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
latent_coeff = 0.1 # 1.0
epochs = 10
iters = 100
batch_size = 32
lr = 1e-3
act_name = "relu"
n_sim_cells = 2000

latent_dim, drift_latent_size, enc_latent_list, dec_latent_list = tunedOurPars(data_name, forecast_num)
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
all_recon_obs = scNODEPredict(latent_ode_model, traj_data[0], tps, n_cells=n_sim_cells)  # (# cells, # tps, # genes)

# ======================================================
# Save results
res_filename = "./res/extrapolation/{}-{}-scNODE-res.npy".format(data_name, split_type)
state_filename = "./res/extrapolation/{}-{}-scNODE-state_dict.pt".format(data_name, split_type)
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



