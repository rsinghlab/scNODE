'''
Description:
    Conduct in silico perturbation in scNODE latent space.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import copy
import scipy.interpolate
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from scipy.optimize import minimize
import scanpy
import os

from benchmark.BenchmarkUtils import loadSCData, splitBySpec
from plotting.PlottingUtils import umapWithoutPCA
from optim.running import constructscNODEModel, scNODETrainWithPreTrain
from plotting import _removeTopRightBorders, _removeAllBorders
from plotting.__init__ import *

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)


# ======================================================

merge_dict = {
    "zebrafish": ["Hematopoeitic", 'Hindbrain', "Notochord", "PSM", "Placode", "Somites", "Spinal", "Neural", "Endoderm"],
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


def saveModel(latent_ode_model, data_name, split_type):
    '''Save trained scNODE.'''
    dict_filename = "../res/perturbation/{}-{}-scNODE-state_dict.pt".format(data_name,split_type)
    torch.save(latent_ode_model.state_dict(), dict_filename)


def loadModel(data_name, split_type):
    '''Load scNODE model.'''
    dict_filename = "../res/perturbation/{}-{}-scNODE-state_dict.pt".format(data_name,split_type)
    latent_ode_model = constructscNODEModel(
        n_genes, latent_dim=latent_dim,
        enc_latent_list=enc_latent_list, dec_latent_list=dec_latent_list, drift_latent_size=drift_latent_size,
        latent_enc_act="none", latent_dec_act=act_name, drift_act=act_name,
        ode_method="euler"
    )
    latent_ode_model.load_state_dict(torch.load(dict_filename))
    latent_ode_model.eval()
    return latent_ode_model

# ======================================================
#          Least Action Path (LAP) Algorithm
# ======================================================

def _discreteVelo(cur_x, last_x, dt):
    # compute discretized tangential velocity
    velo = (cur_x - last_x) / dt
    return velo


def _netVelo(cur_x, last_x, vec_field):
    # compute vector field velocity
    mid_point = (cur_x + last_x) / 2
    velo = vec_field(torch.FloatTensor(mid_point)).detach().numpy()
    return velo


def _action(P, dt, vec_field, D, latent_dim):
    if len(P.shape) == 1:
        P = P.reshape(-1, latent_dim)
    cur_x = P[1:, :]
    last_x = P[:-1, :]
    v = _discreteVelo(cur_x, last_x, dt)
    f = _netVelo(cur_x, last_x, vec_field)
    s = 0.5 * np.square(np.linalg.norm(v-f, ord="fro")) * dt / D
    return s.item()


def leastActionPath(x_0, x_T, path_length, vec_field, D, iters):
    dt = 1
    P = np.linspace(x_0, x_T, num=path_length, endpoint=True, axis=0)
    iter_pbar = tqdm(range(iters), desc="[ LAP ]")
    K = P.shape[1]
    action_list = [_action(P, dt, vec_field, D, K)]
    dt_list = [dt]
    P_list = [P]
    best_dt = dt
    best_P = P
    best_s = action_list[-1]
    for _ in iter_pbar:
        # Step 1: minimize step dt
        dt_res = minimize(
            lambda t: _action(P, dt=t, vec_field=vec_field, D=D, latent_dim=K),
            dt,
            bounds=((1e-5, None), )
        )
        dt = dt_res["x"].item()
        dt_list.append(dt)
        # Step 2: minimize path
        path_res = minimize(
            lambda p: _action(P=p, dt=dt, vec_field=vec_field, D=D, latent_dim=K),
            P[1:-1, :].reshape(-1),
            method="SLSQP",
            tol=1e-5, options={'disp': False ,'eps' : 1e-2},
        )
        inter_P = path_res["x"].reshape(-1, K)
        P = np.concatenate([x_0[np.newaxis, :], inter_P, x_T[np.newaxis, :]], axis=0)
        P_list.append(P)
        # Compute action
        s = _action(P, dt, vec_field, D, K)
        action_list.append(s)
        iter_pbar.set_postfix({"Action": "{:.3f}".format(s)})
        if s < best_s:
            best_dt = dt
            best_P = P
    return best_dt, best_P, action_list, dt_list, P_list


def _interpSpline(x, y):
    x_idx = np.argsort(x)
    sort_x = x[x_idx]
    sort_y = y[x_idx]
    cs = scipy.interpolate.CubicSpline(sort_x, sort_y)
    new_x = np.linspace(sort_x[0], sort_x[-1], 100)
    new_y = cs(new_x)
    return new_x, new_y


def plotZebrafishLAP(
        umap_latent_data, umap_PSM_path, umap_hindbrain_path,
        PSM_idx, hindbrain_idx, start_idx
):
    concat_umap_latent = np.concatenate(umap_latent_data)
    color_list = Bold_10.mpl_colors
    plt.figure(figsize=(6.5, 4))
    plt.scatter(concat_umap_latent[:, 0], concat_umap_latent[:, 1], color=gray_color, s=5, alpha=0.4)
    plt.scatter(
        concat_umap_latent[PSM_idx, 0], concat_umap_latent[PSM_idx, 1],
        color=color_list[0], s=10, alpha=0.4
    )
    plt.scatter(
        concat_umap_latent[hindbrain_idx, 0], concat_umap_latent[hindbrain_idx, 1],
        color=color_list[1], s=10, alpha=0.4
    )
    plt.scatter(
        concat_umap_latent[start_idx, 0], concat_umap_latent[start_idx, 1],
        color=color_list[2], s=10, alpha=0.4
    )

    spline_PSM_x, spline_PSM_y = _interpSpline(umap_PSM_path[:, 0], umap_PSM_path[:, 1])
    spline_optic_x, spline_optic_y = _interpSpline(umap_hindbrain_path[:, 0], umap_hindbrain_path[:, 1])
    plt.plot(
        spline_PSM_x, spline_PSM_y, "--", lw=3,
        color=color_list[0],
        # path_effects=[pe.Stroke(linewidth=5, foreground="k"), pe.Normal()]
    )
    plt.plot(
        spline_optic_x, spline_optic_y, "--", lw=3,
        color=color_list[1],
        # path_effects=[pe.Stroke(linewidth=5, foreground="k"), pe.Normal()]
    )

    plt.scatter(umap_PSM_path[: ,0], umap_PSM_path[:, 1], c=color_list[0], s=200, marker="o", edgecolors= "black")
    plt.scatter(umap_hindbrain_path[: ,0], umap_hindbrain_path[:, 1], c=color_list[1], s=200, marker="o", edgecolors= "black")

    plt.scatter([], [], color=color_list[0], s=50, alpha=1.0, label="PSM")
    plt.scatter([], [], color=color_list[1], s=50, alpha=1.0, label="Hindbrain")
    plt.scatter([], [], color=color_list[2], s=50, alpha=1.0, label="t=0")

    plt.xticks([], [])
    plt.yticks([], [])
    _removeAllBorders()
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=11)
    plt.tight_layout()
    # plt.savefig("../res/figs/zebrafish_lap.pdf")
    plt.show()


# ======================================================

def pathKNNGenes(path_data, latent_data, K):
    nn = NearestNeighbors(n_neighbors=K, n_jobs=-1)
    nn.fit(latent_data)
    dists, neighs = nn.kneighbors(path_data)
    return neighs

# ======================================================

def plotGeneExpression(umap_latent_data, gene_expr, gene_list):
    concat_umap_latent = np.concatenate(umap_latent_data)
    n_genes = len(gene_list)
    # fig, ax_list = plt.subplots(1, n_genes, figsize=(16, 5))
    fig, ax_list = plt.subplots(1, n_genes, figsize=(10, 5))
    for i, g in enumerate(gene_list):
        g_expr = gene_expr[:, i]
        # g_expr = g_expr / np.linalg.norm(g_expr)
        ax_list[i].set_title(g)
        ax_list[i].scatter(concat_umap_latent[:, 0], concat_umap_latent[:, 1], color=gray_color, s=10, alpha=0.5)
        sc = ax_list[i].scatter(concat_umap_latent[:, 0], concat_umap_latent[:, 1], c=g_expr, s=5, cmap="Reds", alpha=0.8)
        ax_list[i].set_xticks([])
        ax_list[i].set_yticks([])
        _removeAllBorders(ax_list[i])
    plt.colorbar(sc)
    plt.tight_layout()
    # plt.savefig("../res/figs/PSM_key_genes.png", dpi=600)
    plt.show()

# ======================================================

def _trainClassifier(X, Y):
    clf_model = RandomForestClassifier(n_estimators=100)
    clf_model.fit(X, Y)
    score = clf_model.score(X, Y)
    print("Train accuracy = {}".format(score))
    return clf_model


def _perturbedPredict(latent_ode_model, start_data, tps, n_cells):
    latent_ode_model.eval()
    start_data_tensor = torch.FloatTensor(start_data)
    # latent_seq, drift_seq, recon_obs = latent_ode_model.computeDiffLatentDrift([start_data_tensor], tps, n_cells)
    _, _, recon_obs = latent_ode_model.predict(start_data_tensor, tps, n_cells)
    recon_obs = [recon_obs[:, t, :].detach().numpy() for t in range(recon_obs.shape[1])]
    return recon_obs


def perturbationAnalysis(traj_data, traj_cell_types, perturb_type, perturb_level = [-3, -2, -1, 0, 1, 2, 3]):
    gene_list = np.load("../res/perturbation/path_DE_genes_wilcoxon.npy", allow_pickle=True).item()
    n_trials = 10
    PSM_gene_list = gene_list["PSM"][:n_trials]
    Hindbrain_gene_list = gene_list["Hindbrain"][:n_trials]
    # Train the classifier
    X = traj_data[-1].detach().numpy()
    Y = traj_cell_types[-1]
    clf_model = _trainClassifier(X, Y)
    perturb_level_str = ["1e{}".format(x) for x in perturb_level]
    PSM_df_list = []
    Hindbrain_df_list = []
    for trial in range(n_trials):
        print("=" * 70)
        print("DE gene trial: {}/{}".format(trial+1, n_trials))
        TBX16_Y_list = []
        TBX16_X_list = []
        SOX3_Y_list = []
        SOX3_X_list = []
        SOX3_idx = ann_data.var_names.values.tolist().index("SOX3")  # for Hindbrain
        TBX16_idx = ann_data.var_names.values.tolist().index("TBX16")  # for PSM
        start_data = traj_data[0].detach().numpy()
        n_cells = traj_data[-1].shape[0]
        SOX3_mean = start_data[:, SOX3_idx].mean()
        TBX16_mean = start_data[:, TBX16_idx].mean()
        print("-" * 70)
        print("perturb_type={} | SOX3_mean={}, TBX16_mean={}".format(perturb_type, SOX3_mean, TBX16_mean))
        for p_level in perturb_level:
            # Gene expression perturbation
            m = 10 ** p_level
            TBX16_m = TBX16_mean * (10 ** p_level)
            SOX3_m = SOX3_mean * (10 ** p_level)
            print("p_level = {} | m={}".format(p_level, m))
            SOX3_perturbed_start = copy.deepcopy(start_data)
            TBX16_perturbed_start = copy.deepcopy(start_data)
            if perturb_type == "multi":
                SOX3_perturbed_start[:, SOX3_idx] *= m
                TBX16_perturbed_start[:, TBX16_idx] *= m
            elif perturb_type == "replace":
                SOX3_perturbed_start[:, SOX3_idx] = SOX3_m
                TBX16_perturbed_start[:, TBX16_idx] = TBX16_m
            # scNODE prediction based on perturbed data
            SOX3_recon_obs = _perturbedPredict(latent_ode_model, SOX3_perturbed_start, train_tps, n_cells=n_cells)
            TBX16_recon_obs = _perturbedPredict(latent_ode_model, TBX16_perturbed_start, train_tps, n_cells=n_cells)
            # Classify predicted cells at the last timepoint
            SOX3_Y = clf_model.predict(SOX3_recon_obs[-1])
            TBX16_Y = clf_model.predict(TBX16_recon_obs[-1])
            # -----
            TBX16_X_list.append(TBX16_recon_obs)
            SOX3_X_list.append(SOX3_recon_obs)
            TBX16_Y_list.append(TBX16_Y)
            SOX3_Y_list.append(SOX3_Y)
        # -----
        # Cell type ratio for TBX16 perturbation (PSM path)
        unperturbed_Y = np.asarray(Y)
        Y_list = [unperturbed_Y] + TBX16_Y_list
        c_list = ["PSM", "other"]
        cnt_mat = np.zeros((len(Y_list), len(c_list)))
        for y_idx, y in enumerate(Y_list):
            for c_idx, c in enumerate(c_list):
                if c == "PSM":
                    cnt_mat[y_idx, c_idx] = len(np.where(y == c)[0])
                else:
                    cnt_mat[y_idx, c_idx] = len(np.where(y != "PSM")[0])
        cnt_mat = cnt_mat / cnt_mat.sum(axis=1)[:, np.newaxis]
        cnt_df = pd.DataFrame(cnt_mat, index=["Unperturbed"] + perturb_level_str, columns=["PSM", "other"])
        print(cnt_df.T)
        PSM_df_list.append(cnt_df)
        # -----
        # Cell type ratio for SOX3 perturbation (Hindbrain path)
        unperturbed_Y = np.asarray(Y)
        Y_list = [unperturbed_Y] + SOX3_Y_list
        c_list = ["Hindbrain", "other"]
        cnt_mat = np.zeros((len(Y_list), len(c_list)))
        for y_idx, y in enumerate(Y_list):
            for c_idx, c in enumerate(c_list):
                if c == "Hindbrain":
                    cnt_mat[y_idx, c_idx] = len(np.where(y == c)[0])
                else:
                    cnt_mat[y_idx, c_idx] = len(np.where(y != "Hindbrain")[0])
        cnt_mat = cnt_mat / cnt_mat.sum(axis=1)[:, np.newaxis]
        cnt_df = pd.DataFrame(cnt_mat, index=["Unperturbed"] + perturb_level_str, columns=["Hindbrain", "other"])
        print(cnt_df.T)
        Hindbrain_df_list.append(cnt_df)
    return PSM_df_list, Hindbrain_df_list


def randomPerturbation(traj_data, traj_cell_types, perturb_type, perturb_level = [-3, -2, -1, 0, 1, 2, 3]):
    gene_list = np.load("../res/perturbation/path_DE_genes_wilcoxon.npy", allow_pickle=True).item()
    other_gene_list = np.unique(gene_list["Hindbrain"][100:] + gene_list["PSM"][100:])[:10]
    print("Other gene list: {}".format(other_gene_list))
    # Train the classifier
    X = traj_data[-1].detach().numpy()
    Y = traj_cell_types[-1]
    clf_model = _trainClassifier(X, Y)
    perturb_level_str = ["1e{}".format(x) for x in perturb_level]
    PSM_df_list = []
    Hindbrain_df_list = []
    for o_g in other_gene_list:
        print("=" * 70)
        print("Non-DE gene: {}".format(o_g))
        g_Y_list = []
        g_X_list = []
        g_idx = ann_data.var_names.values.tolist().index(o_g)
        start_data = traj_data[0].detach().numpy()
        n_cells = traj_data[-1].shape[0]
        g_mean = start_data[:, g_idx].mean()
        print("-" * 70)
        print("perturb_type={} | g_mean={}".format(perturb_type, g_mean))
        for p_level in perturb_level:
            # Gene expression perturbation
            m = 10 ** p_level
            g_m = g_mean * (10 ** p_level)
            print("p_level = {} | m={}".format(p_level, m))
            g_perturbed_start = copy.deepcopy(start_data)
            if perturb_type == "multi":
                g_perturbed_start[:, g_idx] *= m
            elif perturb_type == "replace":
                g_perturbed_start[:, g_idx] = g_m
            # scNODE prediction based on perturbed data
            g_recon_obs = _perturbedPredict(latent_ode_model, g_perturbed_start, train_tps, n_cells=n_cells)
            # Classify predicted cells at the last timepoint
            g_Y = clf_model.predict(g_recon_obs[-1])
            # -----
            g_X_list.append(g_recon_obs)
            g_Y_list.append(g_Y)
        # -----
        # Cell type ratio for TBX16 perturbation (PSM path)
        unperturbed_Y = np.asarray(Y)
        Y_list = [unperturbed_Y] + g_Y_list
        c_list = ["PSM", "other"]
        cnt_mat = np.zeros((len(Y_list), len(c_list)))
        for y_idx, y in enumerate(Y_list):
            for c_idx, c in enumerate(c_list):
                if c == "PSM":
                    cnt_mat[y_idx, c_idx] = len(np.where(y == c)[0])
                else:
                    cnt_mat[y_idx, c_idx] = len(np.where(y != "PSM")[0])
        cnt_mat = cnt_mat / cnt_mat.sum(axis=1)[:, np.newaxis]
        cnt_df = pd.DataFrame(cnt_mat, index=["Unperturbed"] + perturb_level_str, columns=["PSM", "other"])
        print(cnt_df.T)
        PSM_df_list.append(cnt_df)
        # -----
        # Cell type ratio for SOX3 perturbation (Hindbrain path)
        unperturbed_Y = np.asarray(Y)
        Y_list = [unperturbed_Y] + g_Y_list
        c_list = ["Hindbrain", "other"]
        cnt_mat = np.zeros((len(Y_list), len(c_list)))
        for y_idx, y in enumerate(Y_list):
            for c_idx, c in enumerate(c_list):
                if c == "Hindbrain":
                    cnt_mat[y_idx, c_idx] = len(np.where(y == c)[0])
                else:
                    cnt_mat[y_idx, c_idx] = len(np.where(y != "Hindbrain")[0])
        cnt_mat = cnt_mat / cnt_mat.sum(axis=1)[:, np.newaxis]
        cnt_df = pd.DataFrame(cnt_mat, index=["Unperturbed"] + perturb_level_str, columns=["Hindbrain", "other"])
        print(cnt_df.T)
        Hindbrain_df_list.append(cnt_df)
    return PSM_df_list, Hindbrain_df_list


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


def _pValAsterisk(p_val):
    if p_val <= 1e-3:
        return "***"
    if p_val <= 1e-2:
        return "**"
    if p_val <= 5e-2:
        return "*"
    return "n.s."


def comparePerturbationRes(perturb_level):
    res_dict = np.load("../res/perturbation/perturbation_diff_level_cell_composition.npy", allow_pickle=True).item()
    de_PSM_df_list = res_dict["de_PSM_df_list"]
    de_Hindbrain_df_list = res_dict["de_Hindbrain_df_list"]
    random_PSM_df_list = res_dict["random_PSM_df_list"]
    random_Hindbrain_df_list = res_dict["random_Hindbrain_df_list"]
    perturb_level_str = ["1e{}".format(x) for x in perturb_level]
    col_list = ["Unperturbed"] + perturb_level_str
    # -----
    de_PSM_mat = []
    for d in de_PSM_df_list:
        de_PSM_mat.append(d["PSM"].values)
    de_PSM_mat = np.asarray(de_PSM_mat)
    random_PSM_mat = []
    for d in random_PSM_df_list:
        random_PSM_mat.append(d["PSM"].values)
    random_PSM_mat = np.asarray(random_PSM_mat)
    # -----
    de_Hindbrain_mat = []
    for d in de_Hindbrain_df_list:
        de_Hindbrain_mat.append(d["Hindbrain"].values)
    de_Hindbrain_mat = np.asarray(de_Hindbrain_mat)
    random_Hindbrain_mat = []
    for d in random_Hindbrain_df_list:
        random_Hindbrain_mat.append(d["Hindbrain"].values)
    random_Hindbrain_mat = np.asarray(random_Hindbrain_mat)
    # -----
    de_PSM_mat = de_PSM_mat[:, 1:]
    de_Hindbrain_mat = de_Hindbrain_mat[:, 1:]
    random_PSM_mat = random_PSM_mat[:, 1:]
    random_Hindbrain_mat = random_Hindbrain_mat[:, 1:]
    PSM_perturb_pval = []
    Hindbrain_perturb_pval = []
    for p_idx, p in enumerate(perturb_level):
        PSM_p = scipy.stats.ttest_ind(de_PSM_mat[:, p_idx], random_PSM_mat[:, p_idx])
        Hindbrain_p = scipy.stats.ttest_ind(de_Hindbrain_mat[:, p_idx], random_Hindbrain_mat[:, p_idx])
        PSM_perturb_pval.append(PSM_p.pvalue)
        Hindbrain_perturb_pval.append(Hindbrain_p.pvalue)
    min_PSM_val = np.nanmin(np.vstack([de_PSM_mat, random_PSM_mat]))
    max_PSM_val = np.nanmax(np.vstack([de_PSM_mat, random_PSM_mat]))
    min_Hindbrain_val = np.nanmin(np.vstack([de_Hindbrain_mat, random_Hindbrain_mat]))
    max_Hindbrain_val = np.nanmax(np.vstack([de_Hindbrain_mat, random_Hindbrain_mat]))
    # # -----
    # fig = plt.figure(figsize=(13, 5.5))
    # offset = 0.15
    # width = 0.25
    # bplt1 = plt.boxplot(x = de_PSM_mat, positions=np.arange(len(perturb_level)) - offset, patch_artist=True, widths=width)
    # bplt2 = plt.boxplot(x = random_PSM_mat, positions=np.arange(len(perturb_level)) + offset, patch_artist=True, widths=width)
    # colors1 = [Bold_10.mpl_colors[0] for _ in range(len(perturb_level))]
    # colors2 = [Bold_10.mpl_colors[1] for _ in range(len(perturb_level))]
    # _fillColor(bplt1, colors1)
    # _fillColor(bplt2, colors2)
    # plt.bar(-1, 0.0, color=Bold_10.mpl_colors[0], label="DE genes (PSM)")
    # plt.bar(-1, 0.0, color=Bold_10.mpl_colors[1], label="random genes")
    # plt.xlim(-0.5, len(perturb_level) - 0.5)
    # plt.ylim(min_PSM_val-0.2, max_PSM_val+0.2)
    # plt.xticks(np.arange(len(perturb_level)), perturb_level_str)
    # plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0%", "20%", "40%", "60%", "80%", "100%"])
    # plt.xlabel("Perturbation Level")
    # plt.ylabel("Cell Type Composition")
    # plt.legend(frameon=False, ncol=3, loc="upper left")
    # _removeTopRightBorders()
    # plt.tight_layout()
    # plt.show()
    # # -----
    # fig = plt.figure(figsize=(13, 5.5))
    # offset = 0.15
    # width = 0.25
    # bplt1 = plt.boxplot(x=de_Hindbrain_mat, positions=np.arange(len(perturb_level)) - offset, patch_artist=True, widths=width)
    # bplt2 = plt.boxplot(x=random_Hindbrain_mat, positions=np.arange(len(perturb_level)) + offset, patch_artist=True,
    #                     widths=width)
    # colors1 = [Bold_10.mpl_colors[0] for _ in range(len(perturb_level))]
    # colors2 = [Bold_10.mpl_colors[1] for _ in range(len(perturb_level))]
    # _fillColor(bplt1, colors1)
    # _fillColor(bplt2, colors2)
    # plt.bar(-1, 0.0, color=Bold_10.mpl_colors[0], label="DE genes (Hindbrain)")
    # plt.bar(-1, 0.0, color=Bold_10.mpl_colors[1], label="random genes")
    # plt.xlim(-0.5, len(perturb_level) - 0.5)
    # plt.ylim(min_Hindbrain_val - 0.2, max_Hindbrain_val + 0.2)
    # plt.xticks(np.arange(len(perturb_level)), perturb_level_str)
    # plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0%", "20%", "40%", "60%", "80%", "100%"])
    # plt.xlabel("Perturbation Level")
    # plt.ylabel("Cell Type Composition")
    # plt.legend(frameon=False, ncol=3, loc="upper left")
    # _removeTopRightBorders()
    # plt.tight_layout()
    # plt.show()
    # -----
    fig = plt.figure(figsize=(13, 5.5))
    offset = 0.15
    width = 0.25
    bplt1 = plt.bar(height=np.mean(de_PSM_mat, axis=0), x=np.arange(len(perturb_level)) - offset, width=width, align="edge", color=Bold_10.mpl_colors[0], label="DE genes (PSM)")
    bplt2 = plt.bar(height=np.mean(random_PSM_mat, axis=0), x=np.arange(len(perturb_level)) + offset, width=width, align="edge", color=Bold_10.mpl_colors[1], label="random genes")
    plt.xlim(-0.5, len(perturb_level) - 0.5)
    plt.ylim(0.0, max_PSM_val + 0.2)
    plt.xticks(np.arange(len(perturb_level)), perturb_level_str)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0%", "20%", "40%", "60%", "80%", "100%"])
    plt.xlabel("Perturbation Level")
    plt.ylabel("Cell Type Composition")
    plt.legend(frameon=False, ncol=3, loc="upper left")
    _removeTopRightBorders()
    plt.tight_layout()
    plt.show()
    # -----
    fig = plt.figure(figsize=(13, 5.5))
    offset = 0.15
    width = 0.25
    bplt1 = plt.bar(height=np.mean(de_Hindbrain_mat, axis=0), x=np.arange(len(perturb_level)) - offset, width=width,
                    align="edge", color=Bold_10.mpl_colors[0], label="DE genes (Hindbrain)")
    bplt2 = plt.bar(height=np.mean(random_Hindbrain_mat, axis=0), x=np.arange(len(perturb_level)) + offset, width=width,
                    align="edge", color=Bold_10.mpl_colors[1], label="random genes")
    plt.xlim(-0.5, len(perturb_level) - 0.5)
    plt.ylim(0.0, max_PSM_val + 0.2)
    plt.xticks(np.arange(len(perturb_level)), perturb_level_str)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0%", "20%", "40%", "60%", "80%", "100%"])
    plt.xlabel("Perturbation Level")
    plt.ylabel("Cell Type Composition")
    plt.legend(frameon=False, ncol=3, loc="upper left")
    _removeTopRightBorders()
    plt.tight_layout()
    plt.show()

# ======================================================

if __name__ == '__main__':
    # Load data and pre-processing
    print("=" * 70)
    data_name = "zebrafish"
    split_type = "all"
    ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name=data_name, split_type=split_type, data_dir="../res/perturbation/")
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
    # =======================
    latent_ode_model = loadModel(data_name, split_type)
    data_res = np.load(
        "../res/perturbation/zebrafish-all-scNODE-aux_data.npy",
        allow_pickle=True
    ).item()
    latent_seq = data_res["latent_seq"]
    next_seq = data_res["next_seq"]
    drift_seq = data_res["drift_seq"]
    drift_magnitude = data_res["drift_magnitude"]
    umap_latent_data = data_res["umap_latent_data"]
    umap_next_data = data_res["umap_next_data"]
    umap_model = data_res["umap_model"]  # Note: The UMAP is for the model latent space
    latent_tp_list = data_res["latent_tp_list"]

    # =====================================================
    # Least action path
    print("=" * 70)
    PSM_cell_idx = np.where(cell_types == "PSM")[0]
    hindbrain_cell_idx = np.where(cell_types == "Hindbrain")[0]
    start_cell_idx = np.where(cell_tps == 1)[0]
    print("PSM={} | Hindbrain={} | Starting={}".format(len(PSM_cell_idx), len(hindbrain_cell_idx), len(start_cell_idx)))
    concat_latent_seq = np.concatenate(latent_seq, axis=0)
    start_cell_mean = np.mean(concat_latent_seq[start_cell_idx, :], axis=0)
    PSM_cell_mean = np.mean(concat_latent_seq[PSM_cell_idx, :], axis=0)
    hindbrain_cell_mean = np.mean(concat_latent_seq[hindbrain_cell_idx, :], axis=0)
    path_length = 8
    PSM_dt, PSM_P, PSM_action_list, PSM_dt_list, PSM_P_list = leastActionPath(
        x_0=start_cell_mean, x_T=PSM_cell_mean,
        path_length=path_length, vec_field=latent_ode_model.diffeq_decoder.net, D=1, iters=10
    )
    hindbrain_dt, hindbrain_P, hindbrain_action_list, hindbrain_dt_list, hindbrain_P_list = leastActionPath(
        x_0=start_cell_mean, x_T=hindbrain_cell_mean,
        path_length=path_length, vec_field=latent_ode_model.diffeq_decoder.net, D=1, iters=10
    )
    # Plot path
    umap_PSM_P = umap_model.transform(PSM_P)
    umap_hindbrain_P = umap_model.transform(hindbrain_P)
    plotZebrafishLAP(
        umap_latent_data, umap_PSM_P, umap_hindbrain_P,
        PSM_cell_idx, hindbrain_cell_idx, start_cell_idx
    )

    # =====================================================

    # Detect differentially expressed genes
    # To augment cell sets, we find KNN neighbors of path nodes and then detect genes in the gene space
    de_filename = "../res/perturbation/path_DE_genes_wilcoxon.npy"
    if not os.path.isfile(de_filename):
        print("=" * 70)
        PSM_KNN_idx = pathKNNGenes(PSM_P, concat_latent_seq, K=10)
        hindbrain_KNN_idx = pathKNNGenes(hindbrain_P, concat_latent_seq, K=10)
        concat_traj_data = np.concatenate([each.detach().numpy() for each in traj_data], axis=0)
        PSM_gene = np.concatenate([concat_traj_data[idx, :] for idx in PSM_KNN_idx], axis=0)
        hindbrain_gene = np.concatenate([concat_traj_data[idx, :] for idx in hindbrain_KNN_idx], axis=0)
        expr_mat = np.concatenate([PSM_gene, hindbrain_gene], axis=0)
        cell_idx = ["cell_{}".format(i) for i in range(expr_mat.shape[0])]
        cell_types = ["PSM" for t in range(PSM_gene.shape[0])] + ["Hindbrain" for t in range(hindbrain_gene.shape[0])]
        cell_df = pd.DataFrame(data=np.zeros((expr_mat.shape[0], 2)), index=cell_idx, columns=["TP", "TYPE"])
        cell_df.TYPE = cell_types
        expr_df = pd.DataFrame(data=expr_mat, index=cell_idx, columns=ann_data.var_names.values)
        path_ann = scanpy.AnnData(X=expr_df, obs=cell_df)
        scanpy.tl.rank_genes_groups(path_ann, 'TYPE', method="wilcoxon")  # logreg, wilcoxon
        scanpy.pl.rank_genes_groups(path_ann, n_genes=25, sharey=False, fontsize=12)
        group_id = path_ann.uns["rank_genes_groups"]["names"].dtype.names
        gene_names = path_ann.uns["rank_genes_groups"]["names"]
        print(group_id)
        print(gene_names[:10])
        marker_gene_dict = {}
        for i, g_name in enumerate(group_id):
            g_gene = [x[i] for x in gene_names]
            marker_gene_dict[g_name] = g_gene
        np.save("../res/perturbation/path_DE_genes_wilcoxon.npy", marker_gene_dict, allow_pickle=True)

    # =====================================================

    # Plot DE gene expression
    print("=" * 70)
    top_PSM_genes = ['TBX16', "TOB1A"]
    plotGeneExpression(umap_latent_data, gene_expr=ann_data[:, top_PSM_genes].X, gene_list=top_PSM_genes)
    top_hindbrain_genes = ['SOX3', 'SOX19A']
    plotGeneExpression(umap_latent_data, gene_expr=ann_data[:, top_hindbrain_genes].X, gene_list=top_hindbrain_genes)

    # =====================================================

    # Perturbation with different level
    perturb_filename = "../res/perturbation/perturbation_diff_level_cell_composition.npy"
    perturb_type = "multi"  # multi, replace
    perturb_level = [-3, -2, -1, 0, 1, 2, 3]
    if not os.path.isfile(perturb_filename):
        print("=" * 70)
        print("DE genes perturbation...")
        de_PSM_df_list, de_Hindbrain_df_list = perturbationAnalysis(traj_data, traj_cell_types, perturb_type, perturb_level)
        print("Random genes perturbation...")
        random_PSM_df_list, random_Hindbrain_df_list = randomPerturbation(traj_data, traj_cell_types, perturb_type, perturb_level)
        np.save(perturb_filename, {
            "de_PSM_df_list": de_PSM_df_list,
            "de_Hindbrain_df_list": de_Hindbrain_df_list,
            "random_PSM_df_list": random_PSM_df_list,
            "random_Hindbrain_df_list": random_Hindbrain_df_list,
        })
    comparePerturbationRes(perturb_level)

