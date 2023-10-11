'''
Description:
    Compare performance with different number of training time points.
'''
import matplotlib.pyplot as plt
import scipy.spatial.distance
import torch
import torch.distributions as dist
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import itertools
from sklearn.decomposition import PCA


from model.layer import LinearNet, LinearVAENet
from model.diff_solver import ODE
from model.dynamic_model import scNODE
from optim.loss_func import MSELoss, SinkhornLoss, umapLoss
from plotting.visualization import plotUMAP, plotUMAPTimePoint, plotUMAPTestTime, umapWithoutPCA, umapWithPCA
from plotting.utils import linearSegmentCMap
from data.preprocessing import splitBySpec
from benchmark.Compare_SingleCell_Predictions import basicStats, globalEvaluation
from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, tunedOurPars, sampleGaussian
from optim.running import constructLatentODEModel, latentODETrainWithPreTrain, latentODESimulate

from plotting.__init__ import *

# ======================================================

def runExp():
    # Load data and pre-processing
    print("=" * 70)
    data_name = "zebrafish"  # zebrafish, mammalian, drosophila, wot, pancreatic, embryoid
    print("[ {} ]".format(data_name).center(60))
    split_type = "three_interpolation"  # three_interpolation, two_forecasting
    print("Split type: {}".format(split_type))
    ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name, split_type)
    train_tps, test_tps = tpSplitInd(data_name, split_type)
    data = ann_data.X

    # Convert to torch project
    traj_data = [torch.FloatTensor(data[np.where(cell_tps == t)[0], :]) for t in
                 range(1, n_tps + 1)]  # (# tps, # cells, # genes)
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

    # -------------------------------------
    latent_size_list = [25, 50, 75, 100, 125, 150, 175, 200]
    pretrain_iters = 200
    pretrain_lr = 1e-3
    latent_coeff = 1.0
    epochs = 10
    iters = 100
    batch_size = 32
    lr = 1e-3
    act_name = "relu"
    n_sim_cells = 2000

    ot_list = []
    dist_list = []
    data_list = []
    pred_list = []
    for latent_t, latent_size in enumerate(latent_size_list):
        print("*" * 70)
        print("Latent size = {}".format(latent_size))
        # Construct VAE
        print("-" * 60)
        latent_dim = latent_size
        latent_enc_act = "none"
        latent_dec_act = "relu"
        enc_latent_list = None
        dec_latent_list = None
        drift_latent_size = None

        latent_encoder = LinearVAENet(
            input_dim=n_genes, latent_size_list=enc_latent_list, output_dim=latent_dim, act_name=latent_enc_act
        )  # encoder
        obs_decoder = LinearNet(
            input_dim=latent_dim, latent_size_list=dec_latent_list, output_dim=n_genes, act_name=latent_dec_act
        )  # decoder
        print(latent_encoder)
        print(obs_decoder)
        # Model running
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
        reorder_pred_data = [all_recon_obs[:, t, :] for t in range(all_recon_obs.shape[1])]
        # Compute metric for testing time points
        print("Compute metrics...")
        test_tps_list = [int(t) for t in test_tps]
        cur_ot_list = []
        cur_dist_list = []
        for t in test_tps_list:
            print("-" * 70)
            print("t = {}".format(t))
            # -----
            pred_global_metric = globalEvaluation(traj_data[t].detach().numpy(), all_recon_obs[:, t, :])
            print(pred_global_metric)
            cur_ot_list.append(pred_global_metric["ot"])
            cur_dist_list.append(pred_global_metric["l2"])
        ot_list.append(cur_ot_list)
        dist_list.append(cur_dist_list)
        pred_list.append(reorder_pred_data)
    np.save(
        "../res/model_design/{}-{}-performance_vs_latent_size.npy".format(data_name, split_type),
        {
            "ot": ot_list,
            "l2": dist_list,
            "pred": pred_list,
            "latent_size": latent_size_list
        }
    )


def evaluateExp():
    # Load data and pre-processing
    print("=" * 70)
    data_name = "zebrafish"  # zebrafish, mammalian, drosophila, wot
    print("[ {} ]".format(data_name).center(60))
    split_type = "three_interpolation"  # interpolation
    print("Split type: {}".format(split_type))
    ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name, split_type)
    data = ann_data.X
    traj_data = [data[np.where(cell_tps == t)[0], :] for t in range(1, n_tps + 1)]  # (# tps, # cells, # genes)
    cell_num_list = [each.shape[0] for each in traj_data]
    # -----
    exp_res = np.load("../res/model_design/{}-{}-performance_vs_latent_size.npy".format(data_name, split_type), allow_pickle=True).item()
    train_tps, test_tps = tpSplitInd(data_name, split_type)
    latent_size_list = exp_res["latent_size"]
    # -----
    ot_list = exp_res["ot"]
    avg_ot_list = np.mean(ot_list, axis=1)
    l2_list = exp_res["l2"]
    avg_l2_list = np.mean(l2_list, axis=1)
    print("OT ", avg_ot_list)
    print("Avg={}, std={}".format(np.mean(avg_ot_list), np.std(avg_ot_list)))
    print("L2 ", avg_l2_list)
    print("Avg={}, std={}".format(np.mean(avg_l2_list), np.std(avg_l2_list)))
    n_tps = len(train_tps) + len(test_tps)
    color_list = Cube1_6.mpl_colors
    plt.figure(figsize=(8, 4))
    plt.subplot(2, 1, 1)
    plt.bar(np.arange(len(avg_ot_list)), avg_ot_list, color=white_color, edgecolor="k", linewidth=2.0, capsize=5.0)
    plt.xticks([], [])
    plt.ylabel("OT")
    plt.subplot(2, 1, 2)
    plt.bar(np.arange(len(avg_l2_list)), avg_l2_list, color=white_color, edgecolor="k", linewidth=2.0, capsize=5.0)
    plt.ylabel("L2")
    plt.xlabel("Latent Size")
    plt.xticks(np.arange(len(latent_size_list)), ["{:d}".format(x) for x in latent_size_list])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # runExp()
    evaluateExp()
    pass