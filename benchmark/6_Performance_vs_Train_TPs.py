'''
Description:
    Test performance with different number of training timepoints.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import torch
import numpy as np

from benchmark.BenchmarkUtils import loadSCData, splitBySpec
from optim.running import constructscNODEModel, scNODETrainWithPreTrain, scNODEPredict
from optim.evaluation import globalEvaluation

# ======================================================

def runExp():
    # Load data and pre-processing
    print("=" * 70)
    data_name = "zebrafish"  # zebrafish, mammalian, drosophila, wot
    print("[ {} ]".format(data_name).center(60))
    split_type = "first_five"
    print("Split type: {}".format(split_type))
    ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name, split_type)

    if data_name == "zebrafish":
        tp_split_list = [
            ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 10, 11]),
            ([0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]),
            ([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11]),
            ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11]),
            ([0, 1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11]),
            ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11]),
        ]
    elif data_name == "drosophila":
        tp_split_list = [
            ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 10]),
            ([0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10]),
            ([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10]),
            ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10]),
            ([0, 1, 2, 3, 4, 5, 6, 7, 8], [9, 10]),
            # ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10]),
        ]
    elif data_name == "mammalian":
        tp_split_list = [
            ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 10, 11, 12]),
            ([0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11, 12]),
            ([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]),
            ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12]),
            ([0, 1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12]),
            ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12]),
            ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12]),
        ]
    else:
        NotImplementedError("Not implemented for {}!".format(data_name))
    # -------------------------------------
    ot_list = []
    dist_list = []
    data_list = []
    pred_list = []
    for tp_t, tp_list in enumerate(tp_split_list):
        print("*" * 70)
        data = ann_data.X
        train_tps, test_tps = tp_list
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
        # ======================================================
        # Model training
        print("-" * 60)
        latent_dim = 50
        enc_latent_list = [50]
        dec_latent_list = [50]
        drift_latent_size = [50]
        pretrain_iters = 200
        pretrain_lr = 1e-3
        latent_coeff = 1.0  # regularization coefficient: beta
        epochs = 10
        iters = 100
        batch_size = 32
        lr = 1e-3
        act_name = "relu"
        n_sim_cells = 2000
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
        pred_list.append(reorder_pred_data)
    np.save(
        "../res/model_design/{}-{}-performance_vs_train_tps.npy".format(data_name, split_type),
        {
            "ot": ot_list,
            "l2": dist_list,
            "pred": pred_list,
            "tp": tp_split_list
        }
    )


if __name__ == '__main__':
    runExp()