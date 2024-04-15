import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy.stats
import pickle as pkl
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os
import sys
sys.path.append("../../")
from plotting import *
from plotting.visualization import umapWithPCA, umapWithoutPCA
from plotting.utils import _removeTopRightBorders, _removeAllBorders, linearSegmentCMap
from optim.running import constructLatentODEModel
from model_validation.pretraining.utils import selectTPs, miLISI4TP, dbScore, chScore


# ======================================================

def loadModel(data_name, split_type, trial_id):
    filename = "./model/{}-{}-scNODE-diff_reg_coeff-models-trial{}.pt".format(data_name, split_type, trial_id)
    res_dict = torch.load(filename)
    model_dict = res_dict["scNODE_model_list"]
    latent_coeff_list = res_dict["latent_coeff_list"]
    return model_dict, latent_coeff_list


def loadscNODELatent(data_name, split_type, trial_id):
    filename = "./auxilary/{}-{}-scNODE-diff_reg_coeff-latent.npy".format(data_name, split_type)
    latent_res = np.load(filename, allow_pickle=True).item()
    if trial_id is None:
        struc_latent_dict = latent_res["struc_latent_dict"]
        dyn_latent_dict = latent_res["dyn_latent_dict"]
    else:
        struc_latent_dict = latent_res["struc_latent_dict"][trial_id]
        dyn_latent_dict = latent_res["dyn_latent_dict"][trial_id]
    return struc_latent_dict, dyn_latent_dict


def loadscNODEEmbed(data_name, split_type, trial_id):
    filename = "./auxilary/{}-{}-scNODE-diff_reg_coeff-umap_embed.npy".format(data_name, split_type)
    latent_res = np.load(filename, allow_pickle=True).item()
    if trial_id is None:
        struc_emb_dict = latent_res["struc_emb_dict"]
        dyn_emb_dict = latent_res["dyn_emb_dict"]
    else:
        struc_emb_dict = latent_res["struc_emb_dict"][trial_id]
        dyn_emb_dict = latent_res["dyn_emb_dict"][trial_id]
    return struc_emb_dict, dyn_emb_dict


def loadMetric(data_name, split_type, trial_id):
    filename = "./res/{}-{}-scNODE-diff_reg_coeff-metric-trial{}.npy".format(data_name, split_type, trial_id)
    metric_res = np.load(filename, allow_pickle=True).item()
    metric_list = metric_res["metric_list"]
    latent_coeff_list = metric_res["latent_coeff_list"]
    return metric_list, latent_coeff_list

# ======================================================

def _constructscNODE(model_weight):
    act_name = "relu"
    latent_dim, drift_latent_size, enc_latent_list, dec_latent_list = tunedOurPars(data_name, split_type)
    latent_ode_model = constructLatentODEModel(
        n_genes, latent_dim=latent_dim,
        enc_latent_list=enc_latent_list, dec_latent_list=dec_latent_list, drift_latent_size=drift_latent_size,
        latent_enc_act="none", latent_dec_act=act_name, drift_act=act_name,
        ode_method="euler"
    )
    latent_ode_model.load_state_dict(model_weight)
    latent_ode_model.eval()
    return latent_ode_model


def computeLatent(model_dict, traj_data, latent_coeff_list):
    n_coeff_list = latent_coeff_list
    struc_latent_list = []
    dyn_latent_list = []
    for t_idx in range(len(n_coeff_list)):
        print("*" * 50)
        print("coeff idx={}".format(t_idx))
        # structure latent
        s_latent = []
        scNODE_model = _constructscNODE(model_dict[t_idx])
        encoder, decoder = scNODE_model.latent_encoder, scNODE_model.obs_decoder
        for t_data in traj_data:
            latent_mu, latent_std = encoder(t_data)
            latent_sample = sampleGaussian(latent_mu, latent_std)
            s_latent.append(latent_sample.detach().numpy())
        # dynamic latent
        n_cells = 2000
        tps = torch.FloatTensor(all_tps)
        first_tp_data = traj_data[0]
        first_latent_mean, first_latent_std = scNODE_model.latent_encoder(first_tp_data)
        repeat_times = (n_cells // first_latent_mean.shape[0]) + 1
        repeat_mean = torch.repeat_interleave(first_latent_mean, repeat_times, dim=0)[:n_cells, :]
        repeat_std = torch.repeat_interleave(first_latent_std, repeat_times, dim=0)[:n_cells, :]
        first_latent_sample = scNODE_model._sampleGaussian(repeat_mean, repeat_std)
        latent_seq = scNODE_model.diffeq_decoder(first_latent_sample, tps)
        d_latent = [latent_seq[:, j, :].detach().numpy() for j in range(latent_seq.shape[1])]
        # -----
        struc_latent_list.append(s_latent)
        dyn_latent_list.append(d_latent)
    return struc_latent_list, dyn_latent_list

# ======================================================

def latentSilhouetteScore(latent):
    time_label = np.concatenate([np.repeat(t, latent[t].shape[0]) for t in range(n_tps)])
    sil_score = silhouette_score(np.concatenate(latent, axis=0), time_label)
    return sil_score


def latentmiLISIScore(latent):
    time_label = np.concatenate([np.repeat(t, latent[t].shape[0]) for t in range(n_tps)])
    miLISI = miLISI4TP(np.concatenate(latent, axis=0), time_label)
    return miLISI


def latentDBScore(latent):
    time_label = np.concatenate([np.repeat(t, latent[t].shape[0]) for t in range(n_tps)])
    db_score = dbScore(np.concatenate(latent, axis=0), time_label)
    return db_score


def latentCHScore(latent):
    time_label = np.concatenate([np.repeat(t, latent[t].shape[0]) for t in range(n_tps)])
    ch_score = chScore(np.concatenate(latent, axis=0), time_label)
    return ch_score

# ======================================================

def computeEmbed(traj):
    _, umap_model = umapWithoutPCA(
        np.concatenate(traj, axis=0), n_neighbors=50, min_dist=0.01 #, pca_pcs=50
    )
    traj_emb = [
        umap_model.transform(each)
        for each in traj
    ]
    return traj_emb

# ======================================================

def computescNODELatentScore():
    print("-" * 70)
    print("Load latent vars...")
    struc_latent_dict, dyn_latent_dict = loadscNODELatent(data_name, split_type, trial_id=None)
    print("Total num of latent models = {}".format(len(struc_latent_dict[0])))
    dynamic_struc_sil_dict = {}
    dynamic_struc_db_dict = {}
    dynamic_struc_ch_dict = {}
    dynamic_struc_milisi_dict = {}
    dynamic_dyn_sil_dict = {}
    dynamic_dyn_db_dict = {}
    dynamic_dyn_ch_dict = {}
    dynamic_dyn_milisi_dict = {}
    # -----
    for trial in struc_latent_dict.keys():
        print("*" * 70)
        print("Trial {}".format(trial))
        dynamic_struc_latent_list = struc_latent_dict[trial]
        dynamic_dyn_latent_list = dyn_latent_dict[trial]
        print("-" * 70)
        print("Compute scores for dynamic latent (structure)...")
        dynamic_struc_sil = []
        dynamic_struc_db = []
        dynamic_struc_ch = []
        dynamic_struc_milisi = []
        for i, i_latent in enumerate(dynamic_struc_latent_list):
            print("Model {}...".format(i))
            i_sil = latentSilhouetteScore(i_latent)
            i_db = latentDBScore(i_latent)
            i_ch = latentCHScore(i_latent)
            i_milisi = latentmiLISIScore(i_latent)
            dynamic_struc_sil.append(i_sil)
            dynamic_struc_db.append(i_db)
            dynamic_struc_ch.append(i_ch)
            dynamic_struc_milisi.append(i_milisi)
        dynamic_struc_sil_dict[trial] = dynamic_struc_sil
        dynamic_struc_db_dict[trial] = dynamic_struc_db
        dynamic_struc_ch_dict[trial] = dynamic_struc_ch
        dynamic_struc_milisi_dict[trial] = dynamic_struc_milisi
        # -----
        print("-" * 70)
        print("Compute scores for dynamic latent (dynamic)...")
        dynamic_dyn_sil = []
        dynamic_dyn_db = []
        dynamic_dyn_ch = []
        dynamic_dyn_milisi = []
        for i, i_latent in enumerate(dynamic_dyn_latent_list):
            print("Model {}...".format(i))
            i_sil = latentSilhouetteScore(i_latent)
            i_db = latentDBScore(i_latent)
            i_ch = latentCHScore(i_latent)
            i_milisi = latentmiLISIScore(i_latent)
            dynamic_dyn_sil.append(i_sil)
            dynamic_dyn_db.append(i_db)
            dynamic_dyn_ch.append(i_ch)
            dynamic_dyn_milisi.append(i_milisi)
        dynamic_dyn_sil_dict[trial] = dynamic_dyn_sil
        dynamic_dyn_db_dict[trial] = dynamic_dyn_db
        dynamic_dyn_ch_dict[trial] = dynamic_dyn_ch
        dynamic_dyn_milisi_dict[trial] = dynamic_dyn_milisi
    # -----
    print("-" * 70)
    print("Save embedding to file...")
    save_filename = "./auxilary/{}-{}-scNODE-diff_reg_coeff-score.npy".format(data_name, split_type)
    np.save(save_filename, {
        "dynamic_struc_sil_dict": dynamic_struc_sil_dict,
        "dynamic_struc_db_dict": dynamic_struc_db_dict,
        "dynamic_struc_ch_dict": dynamic_struc_ch_dict,
        "dynamic_struc_milisi_dict": dynamic_struc_milisi_dict,

        "dynamic_dyn_sil_dict": dynamic_dyn_sil_dict,
        "dynamic_dyn_db_dict": dynamic_dyn_db_dict,
        "dynamic_dyn_ch_dict": dynamic_dyn_ch_dict,
        "dynamic_dyn_milisi_dict": dynamic_dyn_milisi_dict,
    })


def computeLatentEmbed():
    # -----
    print("Loading trained models and compute latent vars...")
    struc_latent_dict = {}
    dyn_latent_dict = {}
    struc_emb_dict = {}
    dyn_emb_dict = {}
    for trial_id in range(5):
    # for trial_id in [0]:
        print("-" * 50)
        print("Trial:{}".format(trial_id).center(50))
        model_dict, latent_coeff_list = loadModel(data_name, split_type, trial_id)  # tp:[random, first]:[enc, dec]
        (
            struc_latent_list, dyn_latent_list
        ) = computeLatent(model_dict, traj_data, latent_coeff_list)
        struc_latent_dict[trial_id] = struc_latent_list
        dyn_latent_dict[trial_id] = dyn_latent_list
        # -----
        # compute embedding
        print("*" * 70)
        print("Compute UMAP for dynamic latent (structure)...")
        dynamic_struc_latent_emb = []
        for i, i_latent in enumerate(struc_latent_list):
            print("Model {}...".format(i))
            i_emb = computeEmbed(i_latent)
            dynamic_struc_latent_emb.append(i_emb)
        # -----
        print("*" * 70)
        print("Compute UMAP for dynamic latent (dynamic)...")
        dynamic_dyn_latent_emb = []
        for i, i_latent in enumerate(dyn_latent_list):
            print("Model {}...".format(i))
            i_emb = computeEmbed(i_latent)
            dynamic_dyn_latent_emb.append(i_emb)
        # -----
        struc_emb_dict[trial_id] = dynamic_struc_latent_emb
        dyn_emb_dict[trial_id] = dynamic_dyn_latent_emb
    # -----
    print("-" * 70)
    print("Save embedding to file...")
    emb_filename = "./auxilary/{}-{}-scNODE-diff_reg_coeff-umap_embed.npy".format(data_name, split_type)
    latent_filename = "./auxilary/{}-{}-scNODE-diff_reg_coeff-latent.npy".format(data_name, split_type)
    np.save(emb_filename, {
        "struc_emb_dict": struc_emb_dict,
        "dyn_emb_dict": dyn_emb_dict,
    })
    np.save(latent_filename, {
        "struc_latent_dict": struc_latent_dict,
        "dyn_latent_dict": dyn_latent_dict,
    })

# ======================================================

def _fillColor(bplt, colors):
    for patch, color in zip(bplt['boxes'], colors):
        patch.set_facecolor(color)
    for patch, color in zip(bplt['medians'], colors):
        patch.set_color("k")
        patch.set_linewidth(1.0)
        patch.set_linestyle("--")
    for patch, color in zip(bplt['whiskers'], colors):
        patch.set_color("k")
        patch.set_linewidth(2)
    for patch, color in zip(bplt['caps'], colors):
        patch.set_color("k")
        patch.set_linewidth(2)


def mainComparescNODELatentSeparation():
    print("Load latent embedding and scores...")
    score_filename = "./auxilary/{}-{}-scNODE-diff_reg_coeff-score.npy".format(data_name, split_type)
    emb_filename = "./auxilary/{}-{}-scNODE-diff_reg_coeff-umap_embed.npy".format(data_name, split_type)
    emb_res_dict = np.load(emb_filename, allow_pickle=True).item()
    score_res_dict = np.load(score_filename, allow_pickle=True).item()

    struc_emb_dict = emb_res_dict["struc_emb_dict"]
    dyn_emb_dict = emb_res_dict["dyn_emb_dict"]

    dynamic_struc_sil_dict = score_res_dict["dynamic_struc_sil_dict"]
    dynamic_struc_db_dict = score_res_dict["dynamic_struc_db_dict"]
    dynamic_struc_ch_dict = score_res_dict["dynamic_struc_ch_dict"]
    dynamic_struc_milisi_dict = score_res_dict["dynamic_struc_milisi_dict"]
    dynamic_dyn_sil_dict = score_res_dict["dynamic_dyn_sil_dict"]
    dynamic_dyn_db_dict = score_res_dict["dynamic_dyn_db_dict"]
    dynamic_dyn_ch_dict = score_res_dict["dynamic_dyn_ch_dict"]
    dynamic_dyn_milisi_dict = score_res_dict["dynamic_dyn_milisi_dict"]

    trial_list = list(dynamic_struc_sil_dict.keys())
    latent_coeff_list = [0.0, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
    dynamic_struc_sil_mat = np.asarray([dynamic_struc_sil_dict[i] for i in trial_list])
    dynamic_struc_db_mat = np.asarray([dynamic_struc_db_dict[i] for i in trial_list])
    dynamic_struc_ch_mat = np.asarray([dynamic_struc_ch_dict[i] for i in trial_list])
    dynamic_struc_milisi_mat = np.asarray([dynamic_struc_milisi_dict[i] for i in trial_list])
    dynamic_dyn_sil_mat = np.asarray([dynamic_dyn_sil_dict[i] for i in trial_list])
    dynamic_dyn_db_mat = np.asarray([dynamic_dyn_db_dict[i] for i in trial_list])
    dynamic_dyn_ch_mat = np.asarray([dynamic_dyn_ch_dict[i] for i in trial_list])
    dynamic_dyn_milisi_mat = np.asarray([dynamic_dyn_milisi_dict[i] for i in trial_list])
    # -----
    trial_id = 0
    trial_struc_emb = struc_emb_dict[trial_id]
    trial_dyn_emb = dyn_emb_dict[trial_id]
    trial_struc_milisi = dynamic_struc_milisi_mat[trial_id, :]
    trial_struc_db = dynamic_struc_db_mat[trial_id, :]
    trial_dyn_milisi = dynamic_dyn_milisi_mat[trial_id, :]
    trial_dyn_db = dynamic_dyn_db_mat[trial_id, :]
    # -----
    color_list = linearSegmentCMap(n_tps, "viridis")
    # Compare structural latent
    fig, ax_list = plt.subplots(1, len(latent_coeff_list), figsize=(17, 4))
    for i, m_idx in enumerate(latent_coeff_list):
        model_emb = trial_struc_emb[i]
        ax_list[i].set_title(
            "Structural ({:.2f}) \n DB={:.2f}, LISI={:.2f}".format(m_idx, trial_struc_db[i], trial_struc_milisi[i])
            , fontsize=12.5
        )
        for t in range(n_tps):
            c = color_list[t]
            ax_list[i].scatter(
                model_emb[t][:, 0], model_emb[t][:, 1],
                label=int(t), color=c, s=5, alpha=1.0
            )
    for ax in ax_list:
        ax.set_xticks([])
        ax.set_yticks([])
        _removeAllBorders(ax)
    ax_list[-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="TPs", title_fontsize=14, fontsize=13)
    plt.tight_layout()
    plt.tight_layout()
    plt.show()
    # -----
    # Compare dynamic latent
    fig, ax_list = plt.subplots(1, len(latent_coeff_list), figsize=(17, 4))
    for i, m_idx in enumerate(latent_coeff_list):
        model_emb = trial_dyn_emb[i]
        ax_list[i].set_title(
            "Dynamic ({:.2f}) \n DB={:.2f}, LISI={:.2f}".format(m_idx, trial_dyn_db[i], trial_dyn_milisi[i])
            , fontsize=12.5
        )
        for t in range(n_tps):
            c = color_list[t]
            ax_list[i].scatter(
                model_emb[t][:, 0], model_emb[t][:, 1],
                label=int(t), color=c, s=5, alpha=1.0
            )
    for ax in ax_list:
        ax.set_xticks([])
        ax.set_yticks([])
        _removeAllBorders(ax)
    ax_list[-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="TPs", title_fontsize=14, fontsize=13)
    plt.tight_layout()
    plt.tight_layout()
    plt.show()



def mainComapreMetric():
    ot_mat = []
    # for trial_id in range(5):
    # for trial_id in range(1):
    filename_list = [x for x in os.listdir("./res/") if data_name in x and split_type in x]
    trial_id_list = [int(x.split("-")[-1][5]) for x in filename_list]
    for trial_id in trial_id_list:
        metric_list, latent_coeff_list = loadMetric(data_name, split_type, trial_id)
        ot_list = [[m[t]["ot"] for t in m]for m in metric_list]
        avg_ot = [np.mean(x) for x in ot_list]
        ot_mat.append(avg_ot)
    ot_mat = np.asarray(ot_mat)
    print("Avg. OT: {}".format(np.mean(ot_mat, axis=0)))
    # ot_mean = np.mean(ot_mat, axis=0)
    # ot_std = np.var(ot_mat, axis=0)
    # # -----
    # fig = plt.figure(figsize=(8, 4))
    # plt.plot(ot_mean, "o-", lw=2, ms=5, color=Bold_10.mpl_colors[0])
    # plt.fill_between(
    #     x=np.arange(len(latent_coeff_list)),
    #     y1=ot_mean-ot_std,
    #     y2=ot_mean+ot_std,
    #     color=Bold_10.mpl_colors[0], alpha=0.8
    # )
    # plt.xticks(np.arange(len(latent_coeff_list)), latent_coeff_list)
    # plt.ylabel("Wasserstein Dist.")
    # plt.xlabel("Reg. Coeff.")
    # _removeTopRightBorders()
    # plt.tight_layout()
    # plt.show()
    # -----
    fig = plt.figure(figsize=(10, 4))
    width = 0.35
    bplt1 = plt.boxplot(x=ot_mat, positions=np.arange(len(latent_coeff_list)), patch_artist=True, widths=width)
    colors1 = [Bold_10.mpl_colors[0] for _ in range(len(latent_coeff_list))]
    _fillColor(bplt1, colors1)
    plt.xlim(-0.5, len(latent_coeff_list) - 0.5)
    min_val = np.min(ot_mat)
    max_val = np.max(ot_mat)
    plt.ylim(min_val - 2, max_val + 2)
    plt.xticks(np.arange(len(latent_coeff_list)), latent_coeff_list)
    plt.xlabel("Reg. Coeff.")
    plt.ylabel("Wasserstein Dist.")
    # plt.legend(frameon=False, ncol=3, loc="upper left")
    _removeTopRightBorders()
    plt.tight_layout()
    plt.show()

# ======================================================

if __name__ == '__main__':
    # Configuration
    print("=" * 70)
    data_name = "drosophila"  # zebrafish, mammalian, drosophila, wot, pancreatic, embryoid
    print("[ {} ]".format(data_name).center(60))
    split_type = "three_forecasting"  # three_interpolation, three_forecasting, one_interpolation, one_forecasting, two_forecasting, remove_recovery
    print("Split type: {}".format(split_type))
    trial_id = 0
    # -----
    if data_name == "wot":
        from tune_reduced_WOT.utils import loadSCData, tpSplitInd, tunedOurPars, sampleGaussian
    else:
        from benchmark.utils import loadSCData, tpSplitInd, tunedOurPars, sampleGaussian
    # -----
    if data_name == "zebrafish":
        # data_dir = "/oscar/data/rsingh47/jzhan322/sc_Dynamic_Modelling/data/single_cell/experimental/zebrafish_embryonic/new_processed"
        data_dir = "../../data/single_cell/experimental/zebrafish_embryonic/new_processed"
    elif data_name == "mammalian":
        # data_dir = "/oscar/data/rsingh47/jzhan322/sc_Dynamic_Modelling/data/single_cell/experimental/mammalian_cerebral_cortex/new_processed"
        data_dir = "../../data/single_cell/experimental/mammalian_cerebral_cortex/new_processed"
    elif data_name == "drosophila":
        # data_dir = "/oscar/data/rsingh47/jzhan322/sc_Dynamic_Modelling/data/single_cell/experimental/drosophila_embryonic/processed"
        data_dir = "../../data/single_cell/experimental/drosophila_embryonic/processed"
    elif data_name == "wot":
        # data_dir = "/oscar/data/rsingh47/jzhan322/sc_Dynamic_Modelling/data/single_cell/experimental/Schiebinger2019/reduced_processed/"
        data_dir = "../../data/single_cell/experimental/Schiebinger2019/reduced_processed/"
    else:
        raise ValueError("Unknown data name {} for pre-training test!".format(data_name))
    ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name, split_type, data_dir=data_dir)
    data = ann_data.X
    traj_data = [torch.FloatTensor(data[np.where(cell_tps == t)[0], :]) for t in range(1, n_tps + 1)]  # (# tps, # cells, # genes)
    if cell_types is not None:
        traj_cell_types = [cell_types[np.where(cell_tps == t)[0]] for t in range(1, n_tps + 1)]
    all_tps = list(range(n_tps))
    train_tps, test_tps = tpSplitInd(data_name, split_type)
    n_cells = [each.shape[0] for each in traj_data]
    print("# tps={}, # genes={}".format(n_tps, n_genes))
    print("# cells={}".format(n_cells))
    print("Train tps={}".format(train_tps))
    print("Test tps={}".format(test_tps))
    # =========================================================
    # computeLatentEmbed()
    # computescNODELatentScore()
    # =========================================================
    mainComapreMetric()
    # mainComparescNODELatentSeparation()