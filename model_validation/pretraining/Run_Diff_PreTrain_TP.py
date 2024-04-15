'''
Description:
    Test scNODE when using different number of pre-training timepoints.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import os.path
import sys

import matplotlib.pyplot as plt
import torch
import torch.distributions as dist
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import itertools
import copy
import pickle as pkl
import traceback

from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, tunedOurPars, splitBySpec
from optim.evaluation import globalEvaluation
from optim.running import constructscNODEModel, scNODEPredict
from model_validation.pretraining.running import scNODETrainWithDiffPreTrainTP
from model_validation.pretraining.utils import selectTPs
from plotting.PlottingUtils import umapWithPCA


# ======================================================

def runWithDiffPretrainTP(n_tps_list, strategy_list):
    pred_dict = {t: {s: {} for s in strategy_list} for t in n_tps_list}
    model_dict = {t: {s: None for s in strategy_list} for t in n_tps_list}
    pretrain_model_dict = {t: {s: None for s in strategy_list} for t in n_tps_list}
    for n_pretrain_tps in n_tps_list:
        for strategy in strategy_list:
            pretrain_idx = selectTPs(train_tps, n_pretrain_tps, strategy)
            print("*" * 70)
            print("[ n_pretrain_tps={} | strategy={} ] tps_idx = {}".format(n_pretrain_tps, strategy, pretrain_idx))
            # Construct model
            latent_ode_model = constructscNODEModel(
                n_genes, latent_dim=latent_dim,
                enc_latent_list=enc_latent_list, dec_latent_list=dec_latent_list, drift_latent_size=drift_latent_size,
                latent_enc_act="none", latent_dec_act=act_name, drift_act=act_name,
                ode_method="euler"
            )
            try:
                # Model training
                latent_ode_model, loss_list, recon_obs, first_latent_dist, latent_seq, pretrained_model = scNODETrainWithDiffPreTrainTP(
                    train_data, train_tps, pretrain_idx, latent_ode_model, latent_coeff=latent_coeff,
                    epochs=epochs, iters=iters, batch_size=batch_size, lr=lr,
                    pretrain_iters=pretrain_iters, pretrain_lr=pretrain_lr, only_train_de=False,
                )
                all_recon_obs = scNODEPredict(latent_ode_model, traj_data[0], tps, n_cells=n_sim_cells)  # (# cells, # tps, # genes)
                # -----
                # Compute UMAP embeddings for predictions
                print("Compute embeddings for preds...")
                reorder_pred_data = [all_recon_obs[:, t, :] for t in range(all_recon_obs.shape[1])]
                pred_umap_traj = [
                    true_umap_model.transform(true_pca_model.transform(reorder_pred_data[t]))
                    for t in range(len(reorder_pred_data))
                ]
                # Compute metrics
                print("Compute metrics...")
                test_tps_list = [int(t) for t in test_tps]
                test_metric_dict = {}
                for t in test_tps_list:
                    pred_global_metric = globalEvaluation(traj_data[t].detach().numpy(), all_recon_obs[:, t, :])
                    test_metric_dict[t] = pred_global_metric
                # -----
                pred_dict[n_pretrain_tps][strategy] = {
                    "pred_umap": pred_umap_traj,
                    "pred_metric": test_metric_dict,
                    "pretrain_idx": pretrain_idx,
                }
                model_dict[n_pretrain_tps][strategy] = copy.deepcopy(latent_ode_model.state_dict())
                pretrain_model_dict[n_pretrain_tps][strategy] = pretrained_model
            except Exception as err:
                print("Error: ", err)
                pred_dict[n_pretrain_tps][strategy] = None
                model_dict[n_pretrain_tps][strategy] = None
                pretrain_model_dict[n_pretrain_tps][strategy] = None
    return pred_dict, model_dict, pretrain_model_dict


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
traj_data = [torch.FloatTensor(data[np.where(cell_tps == t)[0], :]) for t in
             range(1, n_tps + 1)]  # (# tps, # cells, # genes)
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

# Compute embedding for true data
print("Compute UMAP embeddings for true data...")
embed_model_filename = "../../res/pretraining/{}-{}-scNODE-diff_pretrain-embed_model.npy".format(data_name, split_type)
true_data = [each.detach().numpy() for each in traj_data]
# true_data = [traj_data[0].detach().numpy()]

if os.path.isfile(embed_model_filename):
    print("Load embed model...")
    with open(embed_model_filename, "rb") as file:
        res_dict = pkl.load(file)
        true_umap_model = res_dict["true_umap_model"]
        true_pca_model = res_dict["true_pca_model"]
else:
    true_umap_traj, true_umap_model, true_pca_model = umapWithPCA(np.concatenate(true_data, axis=0), n_neighbors=50, min_dist=0.1, pca_pcs=50)
    print("Save to embed model file...")
    with open(embed_model_filename, "wb") as file:
        pkl.dump({"true_umap_model": true_umap_model, "true_pca_model": true_pca_model}, file)

# ======================================================
# Configurations
pretrain_iters = 200
pretrain_lr = 1e-3
latent_coeff = 1.0
epochs = 5
iters = 100
batch_size = 32
lr = 1e-3
act_name = "relu"
n_sim_cells = 2000
latent_dim, drift_latent_size, enc_latent_list, dec_latent_list = tunedOurPars(data_name, split_type)

n_tp_list = list(range(len(train_tps) + 1))
strategy_list = ["random", "first"]
pred_dict, model_dict, pretrain_model_dict = runWithDiffPretrainTP(n_tp_list, strategy_list)
# ======================================================
# Save results
# trial = 0
res_filename = "../../res/pretraining/{}-{}-scNODE-diff_pretrain-res-trial{}.npy".format(data_name, split_type, trial)
state_filename = "../../res/pretraining/{}-{}-scNODE-diff_pretrain-final_models-trial{}.pt".format(data_name, split_type, trial)
pretrain_state_filename = "../../res/pretraining/{}-{}-scNODE-diff_pretrain-pretrain_models-trial{}.pt".format(data_name, split_type, trial)
print("Saving to {}".format(res_filename))
np.save(res_filename, pred_dict, allow_pickle=True)
torch.save(model_dict, state_filename)
torch.save(pretrain_model_dict, pretrain_state_filename)
