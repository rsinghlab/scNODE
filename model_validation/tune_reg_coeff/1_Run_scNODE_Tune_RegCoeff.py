'''
Description:
    Run our scNODE with different regularization coefficient (beta).

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import torch
import numpy as np

from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, tunedOurPars, splitBySpec
from optim.running import constructscNODEModel, scNODEPredict
from optim.evaluation import globalEvaluation
from model_validation.tune_reg_coeff.running import scNODETrain4Adjustment


# ======================================================

def runWithDiffRegCoeff():
    latent_coeff_list = [0.0, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
    metric_list = []
    for latent_coeff in latent_coeff_list:
        print("*" * 70)
        print("[ Regularization Coefficient = {:.1f} ]".format(latent_coeff).center(70))
        # Construct model
        latent_ode_model = constructscNODEModel(
            n_genes, latent_dim=latent_dim,
            enc_latent_list=enc_latent_list, dec_latent_list=dec_latent_list, drift_latent_size=drift_latent_size,
            latent_enc_act="none", latent_dec_act=act_name, drift_act=act_name,
            ode_method="euler"
        )
        # Model training
        latent_ode_model, loss_list, recon_obs, first_latent_dist, latent_seq, pretrained_model, scNODE_model_list = scNODETrain4Adjustment(
            train_data, train_tps, latent_ode_model, latent_coeff=latent_coeff,
            epochs=epochs, iters=iters, batch_size=batch_size, lr=lr,
            pretrain_iters=pretrain_iters, pretrain_lr=pretrain_lr, only_train_de=False,
        )
        # Model prediction
        all_recon_obs = scNODEPredict(latent_ode_model, traj_data[0], tps, n_cells=n_sim_cells)
        # -----
        print("Compute metrics...")
        test_tps_list = [int(t) for t in test_tps]
        test_metric = {}
        for t in test_tps_list:
            print("-" * 70)
            print("t = {}".format(t))
            # -----
            pred_global_metric = globalEvaluation(traj_data[t].detach().numpy(), all_recon_obs[:, t, :])
            test_metric[t] = pred_global_metric
        metric_list.append(test_metric)
    # -----
    return pretrained_model, scNODE_model_list, metric_list, latent_coeff_list


# ======================================================

import argparse

main_parser = argparse.ArgumentParser()
main_parser.add_argument('-d', default="wot", type=str, help="The data name.")
main_parser.add_argument('-s', default="remove_recovery", type=str, help="Split type.")
main_parser.add_argument('-i', default="0", type=str, help="Trial id.")
args = main_parser.parse_args()
trial = args.i
data_name = args.d
split_type = args.s
print("Trial id = {}".format(trial))

# ======================================================
# Load data and pre-processing
print("=" * 70)
print("[ {} ]".format(data_name).center(60))
print("Split type: {}".format(split_type))
ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name, split_type)
data = ann_data.X
traj_data = [torch.FloatTensor(data[np.where(cell_tps == t)[0], :]) for t in range(1, n_tps + 1)]  # (# tps, # cells, # genes)
if cell_types is not None:
    traj_cell_types = [cell_types[np.where(cell_tps == t)[0]] for t in range(1, n_tps + 1)]

# Split training/testing sets
all_tps = list(range(n_tps))
train_tps, test_tps = tpSplitInd(data_name, split_type)
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
# Configurations
pretrain_iters = 200
pretrain_lr = 1e-3
epochs = 10
iters = 100
batch_size = 32
lr = 1e-3
act_name = "relu"
n_sim_cells = 2000
latent_dim, drift_latent_size, enc_latent_list, dec_latent_list = tunedOurPars(data_name, split_type)

# Run scNODE and compute metrics
pretrained_model, scNODE_model_list, metric_list, latent_coeff_list = runWithDiffRegCoeff()
# ======================================================
# Save results

model_filename = "../../res/tuned_reg_coeff/{}-{}-scNODE-diff_reg_coeff-models-trial{}.pt".format(data_name, split_type, trial)
metric_filename = "../../res/tuned_reg_coeff/{}-{}-scNODE-diff_reg_coeff-metric-trial{}.npy".format(data_name, split_type, trial)
print("Saving to {}".format(model_filename))
torch.save(
    {
        "pretrained_model": pretrained_model,
        "scNODE_model_list": scNODE_model_list,
        "latent_coeff_list": latent_coeff_list,
    }, model_filename
)
np.save(
    metric_filename,
    {
        "metric_list": metric_list,
        "latent_coeff_list": latent_coeff_list,
    },
    allow_pickle=True
)
