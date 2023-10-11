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
from benchmark.BenchmarkUtils import loadZebrafishData, preprocess, preprocess2, sampleGaussian, loadDrosophilaData, loadWOTData, loadMammalianData
from optim.loss_func import MSELoss, SinkhornLoss, umapLoss
from plotting.visualization import plotUMAP, plotUMAPTimePoint, plotUMAPTestTime, umapWithoutPCA, umapWithPCA
from plotting.utils import linearSegmentCMap
from data.preprocessing import splitBySpec
from benchmark.Compare_SingleCell_Predictions import basicStats, globalEvaluation
from plotting.__init__ import *
from plotting.utils import _removeTopRightBorders, _removeAllBorders

# ======================================================
from umap.umap_ import nearest_neighbors as umap_nearest_neighbors
from sklearn.utils import check_random_state


# ======================================================
zebrafish_data_dir = "../data/single_cell/experimental/zebrafish_embryonic/new_processed"
mammalian_data_dir = "../data/single_cell/experimental/mammalian_cerebral_cortex/new_processed"
wot_data_dir = "../data/single_cell/experimental/Schiebinger2019/processed/"
drosophila_data_dir = "../data/single_cell/experimental/drosophila_embryonic/processed/"


def loadSCData(data_name, split_type):
    print("[ Data={} | Split={} ] Loading data...".format(data_name, split_type))
    if data_name == "zebrafish":
        ann_data = loadZebrafishData(zebrafish_data_dir, split_type)
        print("Pre-processing...")
        processed_data = preprocess2(ann_data.copy())
        cell_types =  processed_data.obs["ZF6S-Cluster"].apply(lambda x: "NAN" if pd.isna(x) else x).values
    elif data_name == "mammalian":
        ann_data = loadMammalianData(mammalian_data_dir, split_type)
        processed_data = ann_data.copy()
        cell_types = processed_data.obs.New_cellType.values
    elif data_name == "drosophila":
        ann_data = loadDrosophilaData(drosophila_data_dir, split_type)
        print("Pre-processing...")
        processed_data = preprocess2(ann_data.copy())
        cell_types = processed_data.obs.seurat_clusters.values
    elif data_name == "wot":
        ann_data = loadWOTData(wot_data_dir, split_type)
        processed_data = ann_data.copy()
        cell_types = None
    else:
        raise ValueError("Unknown data name.")
    cell_tps = ann_data.obs["tp"]
    n_tps = len(np.unique(cell_tps))
    n_genes = ann_data.shape[1]
    return processed_data, cell_tps, cell_types, n_genes, n_tps


# ======================================================

def runExp():
    # Load data and pre-processing
    print("=" * 70)
    data_name = "mammalian"  # zebrafish, mammalian, drosophila, wot
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
        # Construct VAE
        print("-" * 60)
        latent_dim = 50
        latent_enc_act = "none"
        enc_latent_list = [50]
        latent_dec_act = "relu"
        dec_latent_list = [50]

        latent_encoder = LinearVAENet(input_dim=n_genes, latent_size_list=enc_latent_list, output_dim=latent_dim,
                                      act_name=latent_enc_act)  # encoder
        obs_decoder = LinearNet(input_dim=latent_dim, latent_size_list=dec_latent_list, output_dim=n_genes,
                                act_name=latent_dec_act)  # decoder
        print(latent_encoder)
        print(obs_decoder)

        # Parametric dimensionality reduction training with all training data
        print("Start dimensionality reduction training...")
        all_train_data = torch.concatenate(train_data, dim=0)
        all_train_tps = np.concatenate([np.repeat(t, train_data[i].shape[0]) for i, t in enumerate(train_tps)])
        dim_reduction_iters = 200  # 200
        if dim_reduction_iters > 0:
            dim_reduction_lr = 1e-3
            ce_coeff = 1.0  # coefficient for cross-entropy loss
            dim_reduction_params = itertools.chain(*[latent_encoder.parameters(), obs_decoder.parameters()])
            dim_reduction_optimizer = torch.optim.Adam(params=dim_reduction_params, lr=dim_reduction_lr,
                                                       betas=(0.95, 0.99))
            dim_reduction_pbar = tqdm(range(dim_reduction_iters), desc="[ Dimensionality Reduction ]")
            latent_encoder.train()
            obs_decoder.train()
            dim_reduction_loss_list = []
            for t in dim_reduction_pbar:
                dim_reduction_optimizer.zero_grad()
                latent_mu, latent_std = latent_encoder(all_train_data)
                latent_sample = sampleGaussian(latent_mu, latent_std)
                recon_obs = obs_decoder(latent_sample)
                dim_reduction_loss = MSELoss(all_train_data, recon_obs)
                dim_reduction_pbar.set_postfix({"Loss": "{:.3f}".format(dim_reduction_loss)})
                dim_reduction_loss_list.append(dim_reduction_loss.item())
                dim_reduction_loss.backward()
                dim_reduction_optimizer.step()

            # Dimensionality reduction prediction
            latent_encoder.eval()
            obs_decoder.eval()
            latent_mu, latent_std = latent_encoder(all_train_data)
            latent_sample = sampleGaussian(latent_mu, latent_std)
            recon_obs = obs_decoder(latent_sample)

            # Visualization
            # plt.plot(dim_reduction_loss_list)
            # plt.xlabel("Dim Reduction Iter")
            # plt.show()
            #
            # true_umap, umap_model, pca_model = umapWithPCA(all_train_data.detach().numpy(), n_neighbors=50, min_dist=0.1,
            #                                                pca_pcs=50)
            # pred_umap = umap_model.transform(pca_model.transform(recon_obs.detach().numpy()))
            # plotUMAP(true_umap, pred_umap)
            # plotUMAPTimePoint(true_umap, pred_umap, all_train_tps, all_train_tps)
            #
            # latent_umap, _ = umapWithoutPCA(latent_sample.detach().numpy(), n_neighbors=50, min_dist=0.1)
            # unique_tps = np.unique(all_train_tps).astype(int).tolist()
            # color_list = linearSegmentCMap(len(unique_tps), "viridis")
            # plt.figure(figsize=(6, 4))
            # plt.title("VAE Latent", fontsize=15)
            # for i, t in enumerate(unique_tps):
            #     t_idx = np.where(all_train_tps == t)[0]
            #     plt.scatter(latent_umap[t_idx, 0], latent_umap[t_idx, 1], label=t, color=color_list[i], s=20, alpha=1.0)
            # plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
            # plt.show()
        # ======================================================
        # Construct latent_ODE model
        print("-" * 60)
        drift_latent_size = [50]
        drift_act = "relu"
        ode_method = "euler"
        diffeq_drift_net = LinearNet(input_dim=latent_dim, latent_size_list=drift_latent_size, output_dim=latent_dim,
                                     act_name=drift_act)  # drift network
        diffeq_decoder = ODE(input_dim=latent_dim, drift_net=diffeq_drift_net,
                             ode_method=ode_method)  # differential equation
        latent_ode_model = scNODE(
            input_dim=n_genes,
            latent_dim=latent_dim,
            output_dim=n_genes,
            latent_encoder=latent_encoder,
            diffeq_decoder=diffeq_decoder,
            obs_decoder=obs_decoder
        )
        print(latent_ode_model)

        # Dynamic learning
        print("Start dimensionality reduction training...")
        epochs = 10
        iters = 100
        batch_size = 32
        num_IWAE_sample = 1
        latent_coeff = 1.0  # 1.0
        blur = 0.05
        scaling = 0.5
        lr = 1e-3
        loss_list = []
        # optimizer = torch.optim.Adam(params=latent_ode_model.diffeq_decoder.net.parameters(), lr=lr, betas=(0.95, 0.99))
        optimizer = torch.optim.Adam(params=latent_ode_model.parameters(), lr=lr, betas=(0.95, 0.99))
        latent_ode_model.train()
        for e in range(epochs):
            epoch_pbar = tqdm(range(iters), desc="[ Epoch {} ]".format(e + 1))
            for t in epoch_pbar:
                optimizer.zero_grad()
                recon_obs, first_latent_dist, first_time_true_batch, latent_seq = latent_ode_model(train_data,
                                                                                                   train_tps,
                                                                                                   num_IWAE_sample,
                                                                                                   batch_size=batch_size)
                encoder_latent_seq = [
                    latent_ode_model.singleReconstruct(each[np.random.choice(np.arange(each.shape[0]), size=batch_size,
                                                                             replace=(each.shape[0] < batch_size)), :])[
                        1]
                    for each in train_data
                ]
                # -----
                # OT loss between true and reconstructed cell sets at each time point
                ot_loss = SinkhornLoss(train_data, recon_obs, blur=blur, scaling=scaling, batch_size=200)
                # Difference between encoder latent and DE latent
                latent_diff = SinkhornLoss(encoder_latent_seq, latent_seq, blur=blur, scaling=scaling, batch_size=None)
                loss = ot_loss + latent_coeff * latent_diff
                epoch_pbar.set_postfix(
                    {"Loss": "{:.3f} | OT={:.3f}, Latent_Diff={:.3f}".format(loss, ot_loss, latent_diff)})
                loss.backward()
                optimizer.step()
                loss_list.append([loss.item(), ot_loss.item(), latent_diff.item()])

        # latent_ODE model prediction
        latent_ode_model.eval()
        recon_obs, first_latent_dist, _, latent_seq = latent_ode_model(train_data, train_tps, num_IWAE_sample,
                                                                       batch_size=None)
        _, all_recon_obs = latent_ode_model.predict(first_latent_dist, tps, n_cells=1000)
        all_recon_obs = all_recon_obs.detach().numpy()  # (# trajs, # tps, # genes)
        reorder_pred_data = [all_recon_obs[:, t, :] for t in range(all_recon_obs.shape[1])]

        # # Visualization
        # plt.figure(figsize=(8, 6))
        # plt.subplot(3, 1, 1)
        # plt.title("Loss")
        # plt.plot([each[0] for each in loss_list])
        # plt.subplot(3, 1, 2)
        # plt.title("OT Term")
        # plt.plot([each[1] for each in loss_list])
        # plt.subplot(3, 1, 3)
        # plt.title("Latent Difference")
        # plt.plot([each[2] for each in loss_list])
        # plt.xlabel("Dynamic Learning Iter")
        # plt.show()
        #
        # print("Compare true and reconstructed data...")
        # true_data = [each.detach().numpy() for each in traj_data]
        # true_cell_tps = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(true_data)])
        # pred_cell_tps = np.concatenate(
        #     [np.repeat(t, all_recon_obs[:, t, :].shape[0]) for t in range(all_recon_obs.shape[1])])
        #
        # true_umap_traj, umap_model, pca_model = umapWithPCA(np.concatenate(true_data, axis=0), n_neighbors=50, min_dist=0.1,
        #                                                     pca_pcs=50)
        # pred_umap_traj = umap_model.transform(pca_model.transform(np.concatenate(reorder_pred_data, axis=0)))
        #
        # plotUMAPTimePoint(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps)
        # plotUMAPTestTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps, test_tps.detach().numpy())
        #
        # print("Visualize latent space...")
        # from downstream_analysis.Visualize_Velocity import computeDrift, computeEmbedding, plotStreamByCellType, plotStream
        # from plotting.utils import linearSegmentCMap
        #
        # latent_seq, next_seq, drift_seq = computeDrift(traj_data, latent_ode_model)
        # drift_magnitude = [np.linalg.norm(each, axis=1) for each in drift_seq]
        # umap_latent_data, umap_next_data, umap_model, latent_tp_list = computeEmbedding(
        #     latent_seq, next_seq, n_neighbors=50, min_dist=0.1  # 0.25
        # )
        # umap_scatter_data = umap_latent_data
        # color_list = linearSegmentCMap(n_tps, "viridis")
        # plotStream(umap_scatter_data, umap_latent_data, umap_next_data, color_list, num_sep=200, n_neighbors=20)
        # if cell_types is not None:
        #     plotStreamByCellType(umap_scatter_data, umap_latent_data, umap_next_data, traj_cell_types, num_sep=200,
        #                          n_neighbors=5)

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


def evaluateExp():
    # Load data and pre-processing
    print("=" * 70)
    data_name = "drosophila"  # zebrafish, mammalian, drosophila
    print("[ {} ]".format(data_name).center(60))
    split_type = "first_five"  # interpolation, forecasting
    print("Split type: {}".format(split_type))
    ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name, split_type)
    data = ann_data.X
    traj_data = [data[np.where(cell_tps == t)[0], :] for t in range(1, n_tps + 1)]  # (# tps, # cells, # genes)
    cell_num_list = [each.shape[0] for each in traj_data]
    # -----
    exp_res = np.load("../res/model_design/{}-{}-performance_vs_train_tps.npy".format(data_name, split_type), allow_pickle=True).item()
    pred_list = exp_res["pred"]
    tp_split_list = exp_res["tp"]
    # -----
    # append nan to the start of metrics
    ot_list = exp_res["ot"]
    l2_list = exp_res["l2"]
    n_tps = len(tp_split_list[0][0]) + len(tp_split_list[0][1])
    max_num_test_tps = len(tp_split_list[0][1])
    min_num_train_tps = len(tp_split_list[0][0])
    for i in range(len(l2_list)):
        l2_list[i] = [np.nan for _ in range(max_num_test_tps - len(l2_list[i]))] + l2_list[i]
    for i in range(len(ot_list)):
        ot_list[i] = [np.nan for _ in range(max_num_test_tps - len(ot_list[i]))] + ot_list[i]
    # color_list = Cube1_6.mpl_colors
    color_list = Tableau_20.mpl_colors
    plt.figure(figsize=(13, 6))
    plt.subplot(2, 1, 1)
    for i, each in enumerate(ot_list):
        plt.plot(np.log10(each), "o-", lw=5, ms=8, color=color_list[i], label=len(tp_split_list[i][0]))
    plt.xticks([], [])
    plt.ylabel("log(OT)")
    plt.legend(title="# Train TPs", title_fontsize=13, fontsize=13)
    plt.subplot(2, 1, 2)
    for i, each in enumerate(l2_list):
        plt.plot(np.log10(each), "o-", lw=5, ms=8, color=color_list[i], label=len(tp_split_list[i][0]))
    plt.ylabel("log(L2)")
    plt.xlabel("Future TP")
    plt.xticks(range(len(ot_list[0])), range(min_num_train_tps+1, n_tps+1))
    plt.legend(title="TP", title_fontsize=13, fontsize=13)
    plt.tight_layout()
    plt.show()
    # -----
    # append nan to the end of metrics
    ot_list = exp_res["ot"]
    l2_list = exp_res["l2"]
    tp_split_list = exp_res["tp"]
    n_tps = len(tp_split_list[0][0]) + len(tp_split_list[0][1])
    max_num_test_tps = len(tp_split_list[0][1])
    min_num_train_tps = len(tp_split_list[0][0])
    for i in range(len(l2_list)):
        l2_list[i] = l2_list[i] + [np.nan for _ in range(max_num_test_tps - len(l2_list[i]))]
    for i in range(len(ot_list)):
        ot_list[i] = ot_list[i] + [np.nan for _ in range(max_num_test_tps - len(ot_list[i]))]

    ot_list = np.log10(ot_list)
    l2_list = np.log10(l2_list)

    # color_list = Cube1_6.mpl_colors
    plt.figure(figsize=(8, 4))
    plt.subplot(2, 1, 1)
    avg_ot = np.nanmean(ot_list, axis=0)
    std_ot = np.nanstd(ot_list, axis=0)
    plt.bar(np.arange(len(avg_ot)), avg_ot, yerr=std_ot, color=white_color, edgecolor="k", linewidth=2.0, capsize=5.0)
    plt.xticks([], [])
    plt.ylabel("log(OT)")
    # plt.ylim(2.0, 3.5)
    plt.subplot(2, 1, 2)
    avg_l2 = np.nanmean(l2_list, axis=0)
    std_l2 = np.nanstd(l2_list, axis=0)
    plt.bar(np.arange(len(avg_l2)), avg_l2, yerr=std_l2, color=white_color, edgecolor="k", linewidth=2.0, capsize=5.0)
    plt.ylabel("log(L2)")
    # plt.ylim(1.5, 1.8)
    plt.xlabel("Next TP")
    plt.xticks(range(len(avg_l2)), range(1, len(avg_l2)+1))
    plt.tight_layout()
    plt.show()


def evaluateExp2():
    # Load data and pre-processing
    print("=" * 70)
    data_name = "mammalian"  # zebrafish, mammalian, drosophila
    print("[ {} ]".format(data_name).center(60))
    split_type = "first_five"  # interpolation, forecasting
    print("Split type: {}".format(split_type))
    ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name, split_type)
    data = ann_data.X
    traj_data = [data[np.where(cell_tps == t)[0], :] for t in range(1, n_tps + 1)]  # (# tps, # cells, # genes)
    cell_num_list = [each.shape[0] for each in traj_data]
    # -----
    exp_res = np.load("../res/model_design/{}-{}-performance_vs_train_tps.npy".format(data_name, split_type), allow_pickle=True).item()
    pred_list = exp_res["pred"]
    tp_split_list = exp_res["tp"]
    n_tps = len(tp_split_list[0][0]) + len(tp_split_list[0][1])
    # -----------------------
    ot_list = exp_res["ot"]
    l2_list = exp_res["l2"]
    if data_name == "mammalian":
        # ot_list[0] = ot_list[0][:-1]
        # l2_list[0] = l2_list[0][:-1]
        # ot_list = ot_list[1:]
        # l2_list = l2_list[1:]
        pass

    tp_split_list = exp_res["tp"]
    n_tps = len(tp_split_list[0][0]) + len(tp_split_list[0][1])
    max_num_test_tps = len(tp_split_list[0][1])
    min_num_train_tps = len(tp_split_list[0][0])
    for i in range(len(l2_list)):
        l2_list[i] = l2_list[i] + [np.nan for _ in range(max_num_test_tps - len(l2_list[i]))]
    for i in range(len(ot_list)):
        ot_list[i] = ot_list[i] + [np.nan for _ in range(max_num_test_tps - len(ot_list[i]))]

    ot_list = np.asarray(ot_list)
    l2_list = np.asarray(l2_list)
    # ot_list = np.log10(ot_list)
    # l2_list = np.log10(l2_list)

    tr_ot_list = [ot_list[:, i][~np.isnan(ot_list[:, i])] for i in range(ot_list.shape[1])]
    tr_l2_list = [l2_list[:, i][~np.isnan(l2_list[:, i])] for i in range(l2_list.shape[1])]

    # tr_ot_df = []
    # for i in range(len(tr_ot_list)):
    #     for j in range(len(tr_ot_list[i])):
    #         tr_ot_df.append([tr_ot_list[i][j], j+1, n_tps-len(tr_ot_list[i])])
    # tr_ot_df = pd.DataFrame(data=tr_ot_df, columns=["Wasserstein", "next_tp", "num_tp"])

    plt.figure(figsize=(12, 3))
    plt.subplot(1, 2, 1)
    plt.boxplot(
        tr_ot_list, positions=np.arange(len(tr_ot_list)),
        showfliers=False, widths=0.7,
        medianprops=dict(linewidth=0), boxprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5)
    )
    sbn.stripplot(data=tr_ot_list, palette=[BlueRed_12.mpl_colors[0]], size=8, edgecolor="k", linewidth=1)
    plt.xlabel("Next TP")
    plt.xticks(range(len(tr_ot_list)), range(1, len(tr_ot_list) + 1))
    plt.ylabel("Wasserstein")
    _removeTopRightBorders()
    plt.subplot(1, 2, 2)
    plt.boxplot(
        tr_l2_list, positions=np.arange(len(tr_l2_list)),
        showfliers=False, widths=0.7,
        medianprops=dict(linewidth=0), boxprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5)
    )
    sbn.stripplot(data=tr_l2_list, palette=[BlueRed_12.mpl_colors[0]], size=8, edgecolor="k", linewidth=1)
    plt.ylabel("L2")
    plt.xlabel("Next TP")
    plt.xticks(range(len(tr_l2_list)), range(1, len(tr_l2_list)+1))
    _removeTopRightBorders()
    plt.tight_layout()
    plt.savefig("../res/figs/{}-next_tp-metric.pdf".format(data_name))
    plt.show()

# ======================================================

def _avgPairDist(data):
    l2_dist_mat = scipy.spatial.distance.cdist(data, data, metric="euclidean")
    cos_dist_mat = scipy.spatial.distance.cdist(data, data, metric="cosine")
    corr_dist_mat = scipy.spatial.distance.cdist(data, data, metric="correlation")
    triu_idx = np.triu_indices(data.shape[0], k=1)
    avg_l2 = np.mean(l2_dist_mat[triu_idx])
    avg_cos = np.mean(cos_dist_mat[triu_idx])
    avg_corr = np.mean(corr_dist_mat[triu_idx])
    return avg_l2, avg_cos, avg_corr



def evaluateData():
    # Load data and pre-processing
    print("=" * 70)
    data_name = "zebrafish"  # zebrafish, mammalian, drosophila, wot
    print("[ {} ]".format(data_name).center(60))
    split_type = "first_five"  # interpolation, forecasting
    print("Split type: {}".format(split_type))
    ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name, split_type)
    data = ann_data.X
    traj_data = [data[np.where(cell_tps == t)[0], :] for t in range(1, n_tps + 1)]  # (# tps, # cells, # genes)
    cell_num_list = [each.shape[0] for each in traj_data]
    # -----
    exp_res = np.load("../res/model_design/{}-{}-performance_vs_train_tps.npy".format(data_name, split_type), allow_pickle=True).item()
    ot_list = exp_res["ot"]
    l2_list = exp_res["l2"]
    pred_list = exp_res["pred"]
    tp_split_list = exp_res["tp"]
    # append nan to the end of metrics
    n_tps = len(tp_split_list[0][0]) + len(tp_split_list[0][1])
    max_num_test_tps = len(tp_split_list[0][1])
    min_num_train_tps = len(tp_split_list[0][0])
    for i in range(len(l2_list)):
        l2_list[i] = [np.nan for _ in range(max_num_test_tps - len(l2_list[i]))] + l2_list[i]
    for i in range(len(ot_list)):
        ot_list[i] = [np.nan for _ in range(max_num_test_tps - len(ot_list[i]))] + ot_list[i]
    # -----
    # Basic stats in the input space
    cell_mean = [np.mean(each, axis=1) for each in traj_data]
    cell_var = [np.var(each, axis=1) for each in traj_data]
    gene_mean = [np.mean(each, axis=0) for each in traj_data]
    gene_var = [np.var(each, axis=0) for each in traj_data]
    plt.figure(figsize=(15, 8))
    plt.subplot(4, 1, 1)
    plt.title("Basic Stats (Input Space)")
    plt.boxplot(cell_mean)
    plt.ylabel("Cell Avg.")
    plt.xticks([], [])
    plt.subplot(4, 1, 2)
    plt.boxplot(cell_var)
    plt.ylabel("Cell Var.")
    plt.xticks([], [])
    plt.subplot(4, 1, 3)
    plt.boxplot(gene_mean)
    plt.ylabel("Gene Avg.")
    plt.xticks([], [])
    plt.subplot(4, 1, 4)
    plt.boxplot(gene_var)
    plt.ylabel("Gene Var.")
    plt.xticks(range(1, len(cell_mean)+1), range(1, len(cell_mean)+1))
    plt.xlabel("TP")
    plt.tight_layout()
    plt.show()

    print("Compute pair distance...")
    avg_dist_list = [_avgPairDist(each) for each in traj_data]
    plt.figure(figsize=(13, 8))
    plt.subplot(3, 1, 1)
    plt.title("Avg. Pair Distant (Input Space)")
    plt.plot([each[0] for each in avg_dist_list], "o-", lw=5, ms=8)
    plt.ylabel("L2")
    plt.xticks([], [])
    plt.subplot(3, 1, 2)
    plt.plot([each[1] for each in avg_dist_list], "o-", lw=5, ms=8)
    plt.ylabel("Cos")
    plt.xticks([], [])
    plt.subplot(3, 1, 3)
    plt.plot([each[2] for each in avg_dist_list], "o-", lw=5, ms=8)
    plt.ylabel("Corr")
    plt.xticks(range(0, len(avg_dist_list)), range(1, len(avg_dist_list)+1))
    plt.tight_layout()
    plt.show()

    # -----
    # Basic stats in the latent space
    print("PCA")
    cell_tps = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(traj_data)])
    latent_data = PCA(n_components=50, svd_solver="arpack").fit_transform(np.concatenate(traj_data, axis=0))
    latent_traj_data = [latent_data[np.where(cell_tps == t)[0], :] for t in np.unique(cell_tps)]
    cell_mean = [np.mean(each, axis=1) for each in latent_traj_data]
    cell_var = [np.var(each, axis=1) for each in latent_traj_data]
    gene_mean = [np.mean(each, axis=0) for each in latent_traj_data]
    gene_var = [np.var(each, axis=0) for each in latent_traj_data]
    plt.figure(figsize=(15, 8))
    plt.subplot(4, 1, 1)
    plt.title("Basic Stats (Latent Space)")
    plt.boxplot(cell_mean)
    plt.ylabel("Cell Avg.")
    plt.xticks([], [])
    plt.subplot(4, 1, 2)
    plt.boxplot(cell_var)
    plt.ylabel("Cell Var.")
    plt.xticks([], [])
    plt.subplot(4, 1, 3)
    plt.boxplot(gene_mean)
    plt.ylabel("Gene Avg.")
    plt.xticks([], [])
    plt.subplot(4, 1, 4)
    plt.boxplot(gene_var)
    plt.ylabel("Gene Var.")
    plt.xticks(range(1, len(cell_mean)+1), range(1, len(cell_mean)+1))
    plt.xlabel("TP")
    plt.tight_layout()
    plt.show()

    print("Compute pair distance...")
    avg_dist_list = [_avgPairDist(each) for each in latent_traj_data]
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.title("Avg. Pair Distant (Latent Space)")
    plt.plot([each[0] for each in avg_dist_list], "o-", lw=5, ms=8)
    plt.ylabel("L2")
    plt.xticks([], [])
    plt.subplot(3, 1, 2)
    plt.plot([each[1] for each in avg_dist_list], "o-", lw=5, ms=8)
    plt.ylabel("Cos")
    plt.xticks([], [])
    plt.subplot(3, 1, 3)
    plt.plot([each[2] for each in avg_dist_list], "o-", lw=5, ms=8)
    plt.ylabel("Corr")
    plt.xticks(range(len(avg_dist_list)), range(1, len(avg_dist_list)+1))
    plt.tight_layout()
    plt.show()
    # -----
    print("PCA + UMAP")
    umap_traj, umap_model, pca_model = umapWithPCA(np.concatenate(traj_data, axis=0), n_neighbors=50, min_dist=0.1, pca_pcs=50)
    color_list = Tableau_20.mpl_colors
    plt.figure(figsize=(6, 4))
    for i, t in enumerate(range(n_tps)):
        t_idx = np.where(cell_tps == t)[0]
        plt.scatter(umap_traj[t_idx, 0], umap_traj[t_idx, 1], label=t, color=color_list[i], s=20, alpha=1.0)
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.show()



if __name__ == '__main__':
    # runExp()
    # evaluateExp()
    evaluateExp2()
    # evaluateData()
    pass