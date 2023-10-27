'''
Description:
    Investigate the latent vector field learned by our scNODE model.

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

from benchmark.BenchmarkUtils import loadSCData, splitBySpec
from plotting.PlottingUtils import umapWithoutPCA
from optim.running import constructscNODEModel, scNODETrainWithPreTrain
from plotting import _removeTopRightBorders, _removeAllBorders
from plotting.__init__ import *

# ======================================================

merge_dict = {
    "zebrafish": ["Hematopoeitic", 'Hindbrain', "Notochord", "PSM", "Placode", "Somites", "Spinal", "Neural", "Endoderm"],
}

def mergeCellTypes(cell_types, merge_list):
    '''Merge cell types.'''
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
    '''Use scNODe to learn cell developmental vector field.'''
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
    dict_filename = "../res/downstream_analysis/vector_field/{}-{}-latent_ODE_OT_pretrain-state_dict.pt".format(data_name,split_type)
    torch.save(latent_ode_model.state_dict(), dict_filename)


def loadModel(data_name, split_type):
    '''Load scNODE model.'''
    dict_filename = "../res/downstream_analysis/vector_field/{}-{}-latent_ODE_OT_pretrain-state_dict.pt".format(data_name,split_type)
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

def computeDrift(traj_data, latent_ode_model):
    # Compute latent and drift seq
    print("Computing latent and drift sequence...")
    latent_seq, drift_seq = latent_ode_model.encodingSeq(traj_data)
    next_seq = [each + 0.1 * drift_seq[i] for i, each in enumerate(latent_seq)]
    return latent_seq, next_seq, drift_seq


def computeEmbedding(latent_seq, next_seq, n_neighbors, min_dist):
    latent_tp_list = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(latent_seq)])
    umap_latent_data, umap_model = umapWithoutPCA(np.concatenate(latent_seq, axis=0), n_neighbors=n_neighbors, min_dist=min_dist)
    # umap_latent_data, umap_model = onlyPCA(np.concatenate(latent_seq, axis=0), pca_pcs=2)
    umap_latent_data = [umap_latent_data[np.where(latent_tp_list == t)[0], :] for t in range(len(latent_seq))]
    umap_next_data = [umap_model.transform(each) for each in next_seq]
    return umap_latent_data, umap_next_data, umap_model, latent_tp_list

# ======================================================

def plotLatent(umap_scatter_data, umap_latent_data, umap_next_data, color_list):
    '''
    Plot streams for cell velocities.

    Variables:
        umap_scatter_data: UMAP embeddings for scatter plot.
        umap_latent_data, umap_next_data: UMAP embeddings for fitting the NN and computing velocities.
    References:
        [1] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.streamplot.html
        [2] https://github.com/theislab/scvelo/blob/master/scvelo/plotting/velocity_embedding_stream.py
    '''
    plt.figure(figsize=(13, 6))
    plt.title("Latent Velocity")
    for t, each in enumerate(umap_scatter_data):
        plt.scatter(each[:, 0], each[:, 1], color=color_list[t], s=10, alpha=0.6)
        plt.scatter([], [], color=color_list[t], s=10, label=t)
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.show()


def plotLatentCellType(umap_scatter_data, umap_latent_data, umap_next_data, traj_cell_type, color_list):
    '''
        Plot streams for cell velocities w.r.t. cell types

        References:
            [1] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.streamplot.html
            [2] https://github.com/theislab/scvelo/blob/master/scvelo/plotting/velocity_embedding_stream.py
        '''
    concat_umap_scatter = np.concatenate(umap_scatter_data)
    concat_cell_types = np.concatenate(traj_cell_type)
    unique_cell_types = np.unique(concat_cell_types)
    cell_num_list = [len(np.where(concat_cell_types == c)[0]) for c in unique_cell_types]
    max10_idx = np.argsort(cell_num_list)[::-1][1:11]
    max10_cell_names = unique_cell_types[max10_idx].tolist()
    # Plotting
    plt.figure(figsize=(13, 6))
    plt.title("Latent Velocity")
    for i, c in enumerate(np.unique(concat_cell_types)):
        c_idx = np.where(concat_cell_types == c)[0]
        cell_c = color_list[max10_cell_names.index(c)] if c in max10_cell_names else gray_color
        plt.scatter(
            concat_umap_scatter[c_idx, 0], concat_umap_scatter[c_idx, 1],
            color=cell_c, s=15, alpha=0.8
        )
        if c in max10_cell_names:
            plt.scatter([], [], color=cell_c, s=20, label=c)
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1, fontsize=12)
    plt.show()

# ======================================================

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
    spline_hindbrain_x, spline_hindbrain_y = _interpSpline(umap_hindbrain_path[:, 0], umap_hindbrain_path[:, 1])
    plt.plot(
        spline_PSM_x, spline_PSM_y, "--", lw=3,
        color=color_list[0],
    )
    plt.plot(
        spline_hindbrain_x, spline_hindbrain_y, "--", lw=3,
        color=color_list[1],
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
    plt.savefig("../res/figs/zebrafish_lap.pdf")
    plt.show()


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


def _interpSpline(x, y):
    x_idx = np.argsort(x)
    sort_x = x[x_idx]
    sort_y = y[x_idx]
    cs = scipy.interpolate.CubicSpline(sort_x, sort_y)
    new_x = np.linspace(sort_x[0], sort_x[-1], 100)
    new_y = cs(new_x)
    return new_x, new_y

# ======================================================

def pathKNNGenes(path_data, latent_data, K):
    nn = NearestNeighbors(n_neighbors=K, n_jobs=-1)
    nn.fit(latent_data)
    dists, neighs = nn.kneighbors(path_data)
    return neighs


def plotGeneExpression(umap_latent_data, gene_expr, gene_list, type_name):
    concat_umap_latent = np.concatenate(umap_latent_data)
    n_genes = len(gene_list)
    fig, ax_list = plt.subplots(1, n_genes, figsize=(10, 5))
    for i, g in enumerate(gene_list):
        g_expr = gene_expr[:, i]
        ax_list[i].set_title(g)
        ax_list[i].scatter(concat_umap_latent[:, 0], concat_umap_latent[:, 1], color=gray_color, s=10, alpha=0.5)
        sc = ax_list[i].scatter(concat_umap_latent[:, 0], concat_umap_latent[:, 1], c=g_expr, s=5, cmap="Reds", alpha=0.8)
        ax_list[i].set_xticks([])
        ax_list[i].set_yticks([])
        _removeAllBorders(ax_list[i])
    plt.colorbar(sc)
    plt.tight_layout()
    plt.savefig("../res/figs/{}_key_genes.png".format(type_name), dpi=600)
    plt.show()

# ======================================================

def perturbationAnalysis(traj_data, traj_cell_types):
    # Train the classifier
    X = traj_data[-1].detach().numpy()
    Y = traj_cell_types[-1]
    clf_model = _trainClassifier(X, Y)
    # Gene expression perturbation
    m = 100
    start_data = traj_data[0].detach().numpy()
    SOX3_idx = ann_data.var_names.values.tolist().index("SOX3")  # for Hindbrain
    SOX3_perturbed_start = copy.deepcopy(start_data)
    SOX3_perturbed_start[:, SOX3_idx] *= m
    TBX16_idx = ann_data.var_names.values.tolist().index("TBX16")  # for PSM
    TBX16_perturbed_start = copy.deepcopy(start_data)
    TBX16_perturbed_start[:, TBX16_idx] *= m
    # scNODE prediction based on perturbed data
    n_cells = traj_data[-1].shape[0]
    SOX3_latent_seq, SOX3_drift_seq, SOX3_recon_obs = _perturbedPredict(latent_ode_model, SOX3_perturbed_start, train_tps, n_cells=n_cells)
    TBX16_latent_seq, TBX16_drift_seq, TBX16_recon_obs = _perturbedPredict(latent_ode_model, TBX16_perturbed_start, train_tps, n_cells=n_cells)
    # Classify predicted cells at the last timepoint
    SOX3_Y = clf_model.predict(SOX3_recon_obs[-1])
    TBX16_Y = clf_model.predict(TBX16_recon_obs[-1])
    # -----
    # Cell type ratio for TBX16 perturbation (PSM path)
    unperturbed_Y = np.asarray(Y)
    Y_list = [unperturbed_Y, TBX16_Y]
    c_list = ["PSM", "other"]
    cnt_mat = np.zeros((len(Y_list), len(c_list)))
    for y_idx, y in enumerate(Y_list):
        for c_idx, c in enumerate(c_list):
            if c == "PSM":
                cnt_mat[y_idx, c_idx] = len(np.where(y == c)[0])
            else:
                cnt_mat[y_idx, c_idx] = len(np.where(y != "PSM")[0])
    cnt_mat = cnt_mat / cnt_mat.sum(axis=1)
    cnt_df = pd.DataFrame(cnt_mat, index=["Unperturbed", "TBX16"], columns=["PSM", "other"])
    print(cnt_df)
    plt.figure(figsize=(5, 3))
    bottom = np.zeros((len(Y_list),))
    width = 0.7
    color_list = [Bold_10.mpl_colors[1], gray_color]
    for c_idx, c in enumerate(["PSM", "other"]):
        plt.bar(np.arange(len(Y_list)), cnt_mat[:, c_idx], width, label=c, bottom=bottom, color=color_list[c_idx])
        bottom += cnt_mat[:, c_idx]
    plt.ylabel("Ratio")
    plt.ylim(0.0, 1.0)
    plt.xticks(np.arange(len(Y_list)), cnt_df.index.values)
    _removeTopRightBorders()
    plt.savefig("../res/figs/PSM_perturbation_cell_ratio.pdf")
    plt.tight_layout()
    plt.show()
    # -----
    # Cell type ratio for SOX3 perturbation (Hindbrain path)
    unperturbed_Y = np.asarray(Y)
    Y_list = [unperturbed_Y, SOX3_Y]
    c_list = ["Hindbrain", "other"]
    cnt_mat = np.zeros((len(Y_list), len(c_list)))
    for y_idx, y in enumerate(Y_list):
        for c_idx, c in enumerate(c_list):
            if c == "Hindbrain":
                cnt_mat[y_idx, c_idx] = len(np.where(y == c)[0])
            else:
                cnt_mat[y_idx, c_idx] = len(np.where(y != "Hindbrain")[0])
    cnt_mat = cnt_mat / cnt_mat.sum(axis=1)
    cnt_df = pd.DataFrame(cnt_mat, index=["Unperturbed", "SOX3"], columns=["Hindbrain", "other"])
    print(cnt_df)
    plt.figure(figsize=(5, 3))
    bottom = np.zeros((len(Y_list),))
    width = 0.7
    color_list = [Bold_10.mpl_colors[1], gray_color]
    for c_idx, c in enumerate(["Hindbrain", "other"]):
        plt.bar(np.arange(len(Y_list)), cnt_mat[:, c_idx], width, label=c, bottom=bottom, color=color_list[c_idx])
        bottom += cnt_mat[:, c_idx]
    plt.ylabel("Ratio")
    plt.ylim(0.0, 1.0)
    plt.xticks(np.arange(len(Y_list)), cnt_df.index.values)
    _removeTopRightBorders()
    plt.savefig("../res/figs/Hindbrain_perturbation_cell_ratio.pdf")
    plt.tight_layout()
    plt.show()


def _trainClassifier(X, Y):
    clf_model = RandomForestClassifier(n_estimators=100)
    clf_model.fit(X, Y)
    score = clf_model.score(X, Y)
    print("Train accuracy = {}".format(score))
    return clf_model


def _perturbedPredict(latent_ode_model, start_data, tps, n_cells):
    latent_ode_model.eval()
    start_data_tensor = torch.FloatTensor(start_data)
    latent_seq, drift_seq, recon_obs = latent_ode_model.computeDiffLatentDrift([start_data_tensor], tps, n_cells)
    latent_seq = [latent_seq[:, t, :].detach().numpy() for t in range(latent_seq.shape[1])]
    recon_obs = [recon_obs[:, t, :].detach().numpy() for t in range(recon_obs.shape[1])]
    return latent_seq, drift_seq, recon_obs


if __name__ == '__main__':
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

    # =======================
    train_model = False
    if train_model:
        print("Model training...")
        latent_ode_model = learnVectorField(train_data, train_tps, tps)
        saveModel(latent_ode_model, data_name, split_type)
        print("Compute latent variables...")
        latent_seq, next_seq, drift_seq = computeDrift(traj_data, latent_ode_model)
        drift_magnitude = [np.linalg.norm(each, axis=1) for each in drift_seq]
        umap_latent_data, umap_next_data, umap_model, latent_tp_list = computeEmbedding(
            latent_seq, next_seq, n_neighbors=50, min_dist=0.1
        )
        # -----
        print("Save intermediate results...")
        data_filename = "../res/downstream_analysis/vector_field/{}-{}-latent_ODE_OT_pretrain-aux_data.npy".format(data_name, split_type)
        np.save(
            data_filename,
            {
                "latent_seq": latent_seq,
                "next_seq": next_seq,
                "drift_seq": drift_seq,
                "drift_magnitude": drift_magnitude,
                "umap_latent_data": umap_latent_data,
                "umap_next_data": umap_next_data,
                "umap_model": umap_model,
                "latent_tp_list": latent_tp_list,
            }
        )
    else:
        print("Load model and intermediate results...")
        latent_ode_model = loadModel(data_name, split_type)
        data_res = np.load(
            "../res/downstream_analysis/vector_field/{}-{}-latent_ODE_OT_pretrain-aux_data.npy".format(data_name, split_type),
            allow_pickle=True
        ).item()
        latent_seq = data_res["latent_seq"]
        next_seq = data_res["next_seq"]
        drift_seq = data_res["drift_seq"]
        drift_magnitude = data_res["drift_magnitude"]
        umap_latent_data = data_res["umap_latent_data"]
        umap_next_data = data_res["umap_next_data"]
        umap_model = data_res["umap_model"] # Note: The UMAP is for the model latent space
        latent_tp_list = data_res["latent_tp_list"]
        # -----
        # Visualize learned latent space
        plotLatent(umap_latent_data, umap_latent_data, umap_next_data, color_list = linearSegmentCMap(n_tps, "viridis"))
        plotLatentCellType(umap_latent_data, umap_latent_data, umap_next_data, traj_cell_types, color_list = Bold_10.mpl_colors)
        # -----
        # Least action path from starting timepoint (t=0) to PSM and Hindbrain cell populations
        PSM_cell_idx = np.where(cell_types == "PSM")[0]  # PSM cells
        hindbrain_cell_idx = np.where(cell_types == "Hindbrain")[0]  # Hindbrain cells
        start_cell_idx = np.where(cell_tps == 1)[0]
        print("# of cells: PSM={} | Hindbrain={} | Starting(t=0)={}".format(len(PSM_cell_idx), len(hindbrain_cell_idx), len(start_cell_idx)))
        concat_latent_seq = np.concatenate(latent_seq, axis=0)
        start_cell_mean = np.mean(concat_latent_seq[start_cell_idx, :], axis=0)
        PSM_cell_mean = np.mean(concat_latent_seq[PSM_cell_idx, :], axis=0)
        hindbrain_cell_mean = np.mean(concat_latent_seq[hindbrain_cell_idx, :], axis=0)
        path_length = 8 # K
        PSM_dt, PSM_P, PSM_action_list, PSM_dt_list, PSM_P_list = leastActionPath(
            x_0=start_cell_mean, x_T=PSM_cell_mean,
            path_length=path_length, vec_field=latent_ode_model.diffeq_decoder.net, D=1, iters=10
        )
        hindbrain_dt, hindbrain_P, hindbrain_action_list, hindbrain_dt_list, hindbrain_P_list = leastActionPath(
            x_0=start_cell_mean, x_T=hindbrain_cell_mean,
            path_length=path_length, vec_field=latent_ode_model.diffeq_decoder.net, D=1, iters=10
        )
        # Plot LAP
        umap_PSM_P = umap_model.transform(PSM_P)
        umap_hindbrain_P = umap_model.transform(hindbrain_P)
        plotZebrafishLAP(
            umap_latent_data, umap_PSM_P, umap_hindbrain_P,
            PSM_cell_idx, hindbrain_cell_idx, start_cell_idx
        )
        # -----
        # Recover path back to gene space
        PSM_KNN_idx = pathKNNGenes(PSM_P, concat_latent_seq, K=10)
        hindbrain_KNN_idx = pathKNNGenes(hindbrain_P, concat_latent_seq, K=10)
        concat_traj_data = np.concatenate([each.detach().numpy() for each in traj_data], axis=0)
        PSM_gene = np.concatenate([concat_traj_data[idx, :] for idx in PSM_KNN_idx], axis=0)
        hindbrain_gene = np.concatenate([concat_traj_data[idx, :] for idx in hindbrain_KNN_idx], axis=0)
        # -----
        # Detect differentially expressed (DE) genes
        expr_mat = np.concatenate([PSM_gene, hindbrain_gene], axis=0)
        cell_idx = ["cell_{}".format(i) for i in range(expr_mat.shape[0])]
        cell_types = ["PSM" for t in range(PSM_gene.shape[0])] + ["Hindbrain" for t in range(hindbrain_gene.shape[0])]
        cell_df = pd.DataFrame(data=np.zeros((expr_mat.shape[0], 2)), index=cell_idx, columns=["TP", "TYPE"])
        cell_df.TYPE = cell_types
        expr_df = pd.DataFrame(data=expr_mat, index=cell_idx, columns=ann_data.var_names.values)
        path_ann = scanpy.AnnData(X=expr_df, obs=cell_df)
        scanpy.tl.rank_genes_groups(path_ann, 'TYPE', method="wilcoxon")
        scanpy.pl.rank_genes_groups(path_ann, n_genes=25, sharey=False, fontsize=12)
        group_id = path_ann.uns["rank_genes_groups"]["names"].dtype.names
        gene_names = path_ann.uns["rank_genes_groups"]["names"]
        print(group_id)
        print(gene_names[:10])
        marker_gene_dict = {}
        for i, g_name in enumerate(group_id):
            g_gene = [x[i] for x in gene_names]
            marker_gene_dict[g_name] = g_gene
        np.save(
            "../res/downstream_analysis/vector_field/Hindbrain_path_marker_genes.npy",
            marker_gene_dict, allow_pickle=True
        )
        # -----
        # Plot gene expression of two DE genes on Hindbrain path
        top_hinbrain_genes = ['SOX3', 'SOX19A']
        plotGeneExpression(umap_latent_data, gene_expr=ann_data[:, top_hinbrain_genes].X, gene_list=top_hinbrain_genes, type_name="Hindbrain")
        # Plot gene expression of two DE genes on PSM path
        top_PSM_genes = ['SOX3', 'TOB1A'] # KNN
        plotGeneExpression(umap_latent_data, gene_expr=ann_data[:, top_PSM_genes].X, gene_list=top_PSM_genes, type_name="PSM")
        # ------
        # Perturbation analysis
        perturbationAnalysis()