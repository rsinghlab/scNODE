'''
Description:
    Test performance with different regularization coefficient beta.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import torch
import numpy as np

from model.layer import LinearNet, LinearVAENet
from plotting.Compare_SingleCell_Predictions import globalEvaluation
from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, splitBySpec
from optim.running import constructscNODEModel, scNODETrainWithPreTrain, scNODEPredict

# ======================================================

def runExp():
    # Load data and pre-processing
    print("=" * 70)
    data_name = "zebrafish"
    print("[ {} ]".format(data_name).center(60))
    split_type = "three_interpolation"
    print("Split type: {}".format(split_type))
    ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name, split_type)
    train_tps, test_tps = tpSplitInd(data_name, split_type)
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

    # -------------------------------------
    latent_coeff_list = [0.0, 0.25, 0.5, 0.75, 1.0]
    latent_dim = 50
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
    for latent_t, latent_coeff in enumerate(latent_coeff_list):
        print("*" * 70)
        print("Latent coeff = {}".format(latent_coeff))
        # Construct VAE
        print("-" * 60)
        latent_enc_act = "none"
        latent_dec_act = "relu"
        enc_latent_list = [latent_dim]
        dec_latent_list = [latent_dim]
        drift_latent_size = [latent_dim]

        latent_encoder = LinearVAENet(
            input_dim=n_genes, latent_size_list=enc_latent_list, output_dim=latent_dim, act_name=latent_enc_act
        )  # encoder
        obs_decoder = LinearNet(
            input_dim=latent_dim, latent_size_list=dec_latent_list, output_dim=n_genes, act_name=latent_dec_act
        )  # decoder
        print(latent_encoder)
        print(obs_decoder)
        # Model running
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
        all_recon_obs = scNODEPredict(latent_ode_model, traj_data[0], tps, n_cells=n_sim_cells)
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
        torch.save(
            latent_ode_model.state_dict(),
            "../res/model_design/{}-{}-latent_coeff{:.2f}-state_dict.pt".format(data_name, split_type, latent_coeff)
        ) # save model for each latent coeff
    np.save(
        "../res/model_design/{}-{}-performance_vs_latent_coeff.npy".format(data_name, split_type),
        {
            "ot": ot_list,
            "l2": dist_list,
            "latent_coeff_list": latent_coeff_list
        }
    )


if __name__ == '__main__':
    runExp()