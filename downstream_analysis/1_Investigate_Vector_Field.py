'''
Description:
    Investigate the latent vector field learned by our model.
'''
import copy

import scipy.interpolate
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize

from benchmark.BenchmarkUtils import loadSCData
from plotting.visualization import umapWithoutPCA
from data.preprocessing import splitBySpec
from optim.running import constructscNODEModel, scNODETrainWithPreTrain
from plotting import _removeTopRightBorders, _removeAllBorders
from plotting.__init__ import *

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
    latent_ode_model, loss_list, recon_obs, first_latent_dist, latent_seq = scNODETrainWithPreTrain(train_data,
                                                                                                    train_tps,
                                                                                                    latent_ode_model,
                                                                                                    latent_coeff=latent_coeff,
                                                                                                    epochs=epochs,
                                                                                                    iters=iters,
                                                                                                    batch_size=batch_size,
                                                                                                    lr=lr,
                                                                                                    pretrain_iters=pretrain_iters,
                                                                                                    pretrain_lr=pretrain_lr)
    return latent_ode_model


def saveModel(latent_ode_model, data_name, split_type):
    dict_filename = "../res/downstream_analysis/vector_field/{}-{}-latent_ODE_OT_pretrain-state_dict.pt".format(data_name,split_type)
    torch.save(latent_ode_model.state_dict(), dict_filename)


def loadModel(data_name, split_type):
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


def computeNextDecoder(next_seq, latent_ode_model):
    print("Compute decoder next sequence...")
    next_decode_seq = [latent_ode_model.obs_decoder(torch.FloatTensor(each)).detach().numpy() for each in next_seq]
    return next_decode_seq


def computeEmbedding(latent_seq, next_seq, n_neighbors, min_dist):
    latent_tp_list = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(latent_seq)])
    umap_latent_data, umap_model = umapWithoutPCA(np.concatenate(latent_seq, axis=0), n_neighbors=n_neighbors, min_dist=min_dist)
    # umap_latent_data, umap_model = onlyPCA(np.concatenate(latent_seq, axis=0), pca_pcs=2)
    umap_latent_data = [umap_latent_data[np.where(latent_tp_list == t)[0], :] for t in range(len(latent_seq))]
    umap_next_data = [umap_model.transform(each) for each in next_seq]
    return umap_latent_data, umap_next_data, umap_model, latent_tp_list


def plotStream(umap_scatter_data, umap_latent_data, umap_next_data, color_list, num_sep=200, n_neighbors=10):
    '''
    Plot streams for cell velocities.

    Variables:
        umap_scatter_data: UMAP embeddings for scatter plot.
        umap_latent_data, umap_next_data: UMAP embeddings for fitting the NN and computing velocities.
    References:
        [1] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.streamplot.html
        [2] https://github.com/theislab/scvelo/blob/master/scvelo/plotting/velocity_embedding_stream.py
    '''
    concat_umap_scatter = np.concatenate(umap_scatter_data)
    concat_umap_latent = np.concatenate(umap_latent_data)
    concat_umap_next = np.concatenate(umap_next_data)
    # Construct grid coordinates
    min_X, max_X = concat_umap_scatter[:, 0].min(), concat_umap_scatter[:, 0].max()
    min_Y, max_Y = concat_umap_scatter[:, 1].min(), concat_umap_scatter[:, 1].max()
    X, Y = np.meshgrid(np.linspace(min_X, max_X, num_sep), np.linspace(min_Y, max_Y, num_sep))
    grid_coord = np.vstack([i.flat for i in (X, Y)]).T
    # For each grid point, find its nearest neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(concat_umap_latent)
    dists, neighs = nn.kneighbors(grid_coord)
    # Use the average of neighbors to compute velocity
    grid_latent = concat_umap_latent[neighs.squeeze()]
    grid_next = concat_umap_next[neighs.squeeze()]
    velocity = (grid_next - grid_latent).mean(axis=1)
    U, V = velocity[:, 0].reshape(num_sep, num_sep), velocity[:, 1].reshape(num_sep, num_sep)
    # Filter outlier points
    grid_min_dist = dists.min(axis=1)
    # cutoff_val = np.percentile(grid_min_dist, 50)
    cutoff_val = 0.01
    selected_idx = np.where(grid_min_dist < cutoff_val)[0]
    # Plotting
    plt.figure(figsize=(13, 6))
    plt.title("Latent Velocity")
    for t, each in enumerate(umap_scatter_data):
        plt.scatter(each[:, 0], each[:, 1], color=color_list[t], s=10, alpha=0.6)
        plt.scatter([], [], color=color_list[t], s=10, label=t)
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.streamplot(X, Y, U, V, density=2.0, arrowsize=1.5, color="k", linewidth=1.0, start_points=grid_coord[selected_idx])
    plt.show()


def plotStreamByCellType(umap_scatter_data, umap_latent_data, umap_next_data, traj_cell_type, num_sep=200, n_neighbors=10):
    '''
        Plot streams for cell velocities w.r.t. cell types

        References:
            [1] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.streamplot.html
            [2] https://github.com/theislab/scvelo/blob/master/scvelo/plotting/velocity_embedding_stream.py
        '''
    concat_umap_scatter = np.concatenate(umap_scatter_data)
    concat_umap_latent = np.concatenate(umap_latent_data)
    concat_umap_next = np.concatenate(umap_next_data)
    concat_cell_types = np.concatenate(traj_cell_type)
    color_list = sbn.color_palette("dark", len(np.unique(concat_cell_types)))
    # Construct grid coordinates
    min_X, max_X = concat_umap_scatter[:, 0].min(), concat_umap_scatter[:, 0].max()
    min_Y, max_Y = concat_umap_scatter[:, 1].min(), concat_umap_scatter[:, 1].max()
    X, Y = np.meshgrid(np.linspace(min_X, max_X, num_sep), np.linspace(min_Y, max_Y, num_sep))
    grid_coord = np.vstack([i.flat for i in (X, Y)]).T
    # For each grid point, find its nearest neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(concat_umap_latent)
    dists, neighs = nn.kneighbors(grid_coord)
    # Use the average of neighbors to compute velocity
    grid_latent = concat_umap_latent[neighs.squeeze()]
    grid_next = concat_umap_next[neighs.squeeze()]
    velocity = (grid_next - grid_latent).mean(axis=1)
    U, V = velocity[:, 0].reshape(num_sep, num_sep), velocity[:, 1].reshape(num_sep, num_sep)
    # Filter outlier points
    grid_min_dist = dists.min(axis=1)
    # cutoff_val = np.percentile(grid_min_dist, 50)
    cutoff_val = 0.01
    selected_idx = np.where(grid_min_dist < cutoff_val)[0]
    # Plotting
    plt.figure(figsize=(13, 6))
    plt.title("Latent Velocity")
    for i, c in enumerate(np.unique(concat_cell_types)):
        c_idx = np.where(concat_cell_types == c)[0]
        plt.scatter(concat_umap_scatter[c_idx, 0], concat_umap_scatter[c_idx, 1], color=color_list[i] if c != "NAN" else gray_color, s=10, alpha=0.6)
        plt.scatter([], [], color=color_list[i], s=20, label=c)
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.streamplot(X, Y, U, V, density=2.0, arrowsize=1.5, color="k", linewidth=1.0, start_points=grid_coord[selected_idx])
    plt.show()


def visVectorField(traj_data, latent_ode_model, traj_cell_types):
    latent_seq, next_seq, drift_seq = computeDrift(traj_data, latent_ode_model)
    drift_magnitude = [np.linalg.norm(each, axis=1) for each in drift_seq]
    umap_latent_data, umap_next_data, umap_model, latent_tp_list = computeEmbedding(
        latent_seq, next_seq, n_neighbors=50, min_dist=0.1  # 0.25
    )
    umap_scatter_data = umap_latent_data
    color_list = linearSegmentCMap(n_tps, "viridis")
    plotStream(umap_scatter_data, umap_latent_data, umap_next_data, color_list, num_sep=200, n_neighbors=20)
    if traj_cell_types is not None:
        plotStreamByCellType(umap_scatter_data, umap_latent_data, umap_next_data, traj_cell_types, num_sep=200, n_neighbors=5)
    return latent_seq, next_seq, drift_seq, drift_magnitude, umap_latent_data, umap_next_data, umap_model, latent_tp_list

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

import elpigraph
def constructLineageGraph(umap_latent_data, num_nodes):
    umap_data = np.concatenate(umap_latent_data, axis=0)
    pg_tree = elpigraph.computeElasticPrincipalTree(
        umap_data, NumNodes=num_nodes, Lambda=0.001, Mu=0.001, Do_PCA=False, CenterData=False
    )[0]
    return pg_tree


def plotLineageGraph(umap_latent_data, pg_tree, traj_cell_type, color_list):
    node_pos = pg_tree["NodePositions"]
    edges = pg_tree["Edges"]
    edges_idx = edges[0].T
    # -----
    plt.figure(figsize=(15, 6))
    for t, each in enumerate(umap_latent_data):
        plt.scatter(each[:, 0], each[:, 1], color=color_list[t], s=10, alpha=0.6)
        plt.scatter([], [], color=color_list[t], s=10, label=t)
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.scatter(
        node_pos[:, 0],
        node_pos[:, 1],
        c="k",
        s=50,
        alpha=0.6,
    )
    for j in range(edges_idx.shape[1]):
        x_coo = np.concatenate(
            (node_pos[edges_idx[0, j], [0]], node_pos[edges_idx[1, j], [0]])
        )
        y_coo = np.concatenate(
            (node_pos[edges_idx[0, j], [1]], node_pos[edges_idx[1, j], [1]])
        )
        plt.plot(x_coo, y_coo, c="black", linewidth=3, alpha=0.6)
    plt.tight_layout()
    plt.show()
    # -----
    concat_umap_scatter = np.concatenate(umap_latent_data)
    concat_cell_types = np.concatenate(traj_cell_type)
    # color_list = sbn.color_palette("dark", len(np.unique(concat_cell_types)))
    color_list = Bold_10.mpl_colors
    unique_cell_types = np.unique(concat_cell_types)
    cell_num_list = [len(np.where(concat_cell_types == c)[0]) for c in unique_cell_types]
    max10_idx = np.argsort(cell_num_list)[::-1][1:11]
    max10_cell_names = unique_cell_types[max10_idx].tolist()
    plt.figure(figsize=(13, 6))
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
    plt.scatter(
        node_pos[:, 0],
        node_pos[:, 1],
        c="k",
        s=50,
        alpha=0.6,
    )
    for j in range(edges_idx.shape[1]):
        x_coo = np.concatenate(
            (node_pos[edges_idx[0, j], [0]], node_pos[edges_idx[1, j], [0]])
        )
        y_coo = np.concatenate(
            (node_pos[edges_idx[0, j], [1]], node_pos[edges_idx[1, j], [1]])
        )
        plt.plot(x_coo, y_coo, c="black", linewidth=3, alpha=0.6)
    plt.tight_layout()
    plt.show()

# ======================================================
from scipy.optimize import fsolve
def findFixedPoint(latent_ode_model, traj_data, n_points):
    latent_ode_model.eval()
    last_latent = latent_ode_model.latent_encoder.net(traj_data[-1])

    def func(x):
        x_tensor = torch.FloatTensor(x)
        y = latent_ode_model.diffeq_decoder.net(x_tensor).detach().numpy()
        return y

    starting_list = []
    root_list = []
    for i in range(n_points):
        starting_point = last_latent[np.random.choice(np.arange(last_latent.shape[0]), 1).item(), :].detach().numpy()
        root = fsolve(func, starting_point, full_output=True)
        starting_list.append(starting_point)
        root_list.append(root)
    return starting_list, root_list


def assignFixedPointType(latent_ode_model, fixed_points):
    jacobian_list = []
    for x in fixed_points:
        x_tensor = torch.FloatTensor(x)
        x_jacobian = torch.autograd.functional.jacobian(latent_ode_model.diffeq_decoder.net, x_tensor)
        jacobian_list.append(x_jacobian.detach().numpy())
    divergence = [np.linalg.det(each) for each in jacobian_list] # Note: all 0, constant flow
    return jacobian_list, divergence


def plotStreamWithFixedPoint(umap_scatter_data, umap_latent_data, umap_next_data, fixed_points, color_list, num_sep=200, n_neighbors=10):
    '''
    Plot streams for cell velocities.

    Variables:
        umap_scatter_data: UMAP embeddings for scatter plot.
        umap_latent_data, umap_next_data: UMAP embeddings for fitting the NN and computing velocities.
    References:
        [1] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.streamplot.html
        [2] https://github.com/theislab/scvelo/blob/master/scvelo/plotting/velocity_embedding_stream.py
    '''
    concat_umap_scatter = np.concatenate(umap_scatter_data)
    concat_umap_latent = np.concatenate(umap_latent_data)
    concat_umap_next = np.concatenate(umap_next_data)
    # Construct grid coordinates
    min_X, max_X = concat_umap_scatter[:, 0].min(), concat_umap_scatter[:, 0].max()
    min_Y, max_Y = concat_umap_scatter[:, 1].min(), concat_umap_scatter[:, 1].max()
    X, Y = np.meshgrid(np.linspace(min_X, max_X, num_sep), np.linspace(min_Y, max_Y, num_sep))
    grid_coord = np.vstack([i.flat for i in (X, Y)]).T
    # For each grid point, find its nearest neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(concat_umap_latent)
    dists, neighs = nn.kneighbors(grid_coord)
    # Use the average of neighbors to compute velocity
    grid_latent = concat_umap_latent[neighs.squeeze()]
    grid_next = concat_umap_next[neighs.squeeze()]
    velocity = (grid_next - grid_latent).mean(axis=1)
    U, V = velocity[:, 0].reshape(num_sep, num_sep), velocity[:, 1].reshape(num_sep, num_sep)
    # Filter outlier points
    grid_min_dist = dists.min(axis=1)
    # cutoff_val = np.percentile(grid_min_dist, 50)
    cutoff_val = 0.01
    selected_idx = np.where(grid_min_dist < cutoff_val)[0]
    # Plotting
    plt.figure(figsize=(13, 6))
    plt.title("Latent Velocity (w/ fixed points)")
    for t, each in enumerate(umap_scatter_data):
        plt.scatter(each[:, 0], each[:, 1], color=color_list[t], s=10, alpha=0.6)
        plt.scatter([], [], color=color_list[t], s=10, label=t)
    plt.scatter(fixed_points[:, 0], fixed_points[:, 1], color=BlueRed_12.mpl_colors[-2], s=70, alpha=1.0)
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.streamplot(X, Y, U, V, density=2.0, arrowsize=1.5, color="k", linewidth=0.9, start_points=grid_coord[selected_idx])
    plt.show()

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
        umap_latent_data, umap_PSM_path, umap_optic_path,
        PSM_idx, optic_idx, start_idx
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
        concat_umap_latent[optic_idx, 0], concat_umap_latent[optic_idx, 1],
        color=color_list[1], s=10, alpha=0.4
    )
    plt.scatter(
        concat_umap_latent[start_idx, 0], concat_umap_latent[start_idx, 1],
        color=color_list[2], s=10, alpha=0.4
    )

    spline_PSM_x, spline_PSM_y = _interpSpline(umap_PSM_path[:, 0], umap_PSM_path[:, 1])
    spline_optic_x, spline_optic_y = _interpSpline(umap_optic_path[:, 0], umap_optic_path[:, 1])
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
    plt.scatter(umap_optic_path[: ,0], umap_optic_path[:, 1], c=color_list[1], s=200, marker="o", edgecolors= "black")

    plt.scatter([], [], color=color_list[0], s=50, alpha=1.0, label="PSM")
    plt.scatter([], [], color=color_list[1], s=50, alpha=1.0, label="Optic Cup")
    plt.scatter([], [], color=color_list[2], s=50, alpha=1.0, label="t=0")

    plt.xticks([], [])
    plt.yticks([], [])
    _removeAllBorders()
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=11)
    plt.tight_layout()
    plt.savefig("../res/figs/zebrafish_lap.pdf")
    plt.show()


# ======================================================

def pathKNNGenes(path_data, latent_data, K):
    nn = NearestNeighbors(n_neighbors=K, n_jobs=-1)
    nn.fit(latent_data)
    dists, neighs = nn.kneighbors(path_data)
    return neighs

# ======================================================

def _map2GeneSpace(latent, decoder):
    gene_exp = decoder(torch.FloatTensor(latent)).detach().numpy()
    return gene_exp


def _MSD(gene_path):
    gene0 = np.tile(gene_path[[0]].T, gene_path.shape[0]).T
    MSD = np.sum((gene_path - gene0)**2, axis=0)
    return MSD


def rankGenesByMSD(gene_latent_path, decoder, gene_list):
    gene_exp = _map2GeneSpace(gene_latent_path, decoder)
    gene_MSD = _MSD(gene_exp)
    gene_MSD = [(g, gene_MSD[i]) for i, g in enumerate(gene_list)]
    sorted_gene = sorted(gene_MSD, key=lambda x: x[1], reverse=True)
    return sorted_gene


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
    plt.savefig("../res/figs/PSM_key_genes.png", dpi=600)
    plt.show()


def plotOneGeneExpression(umap_latent_data, gene_expr, gene_list):
    concat_umap_latent = np.concatenate(umap_latent_data)
    n_genes = len(gene_list)
    plt.figure(figsize=(6, 4))
    g_expr = gene_expr[:, 0]
    plt.title(gene_list[0])
    plt.scatter(concat_umap_latent[:, 0], concat_umap_latent[:, 1], color=gray_color, s=10, alpha=0.5)
    sc = plt.scatter(concat_umap_latent[:, 0], concat_umap_latent[:, 1], c=g_expr, s=5, cmap="Reds", alpha=0.8)
    plt.xticks([], [])
    plt.yticks([], [])
    _removeAllBorders()
    plt.colorbar(sc)
    plt.tight_layout()
    plt.savefig("../res/figs/zebrafish_PSM_TBX16_expression.pdf")
    plt.show()


from matplotlib_venn import venn2
def plotZebrafishTopGenes(ann_data, PSM_gene_ranks, optic_gene_ranks):
    PSM_gene_ranks = [x[0] for x in PSM_gene_ranks[:100]]
    optic_gene_ranks = [x[0] for x in optic_gene_ranks[:100]]

    venn2(
        [set(PSM_gene_ranks), set(optic_gene_ranks)],
        set_labels=("PSM", "Optic Cup")
    )
    plt.show()

    unique_PSM_gene_ranks = [x for x in PSM_gene_ranks if x not in np.intersect1d(PSM_gene_ranks, optic_gene_ranks)]
    unique_optic_gene_ranks = [x for x in optic_gene_ranks if x not in np.intersect1d(PSM_gene_ranks, optic_gene_ranks)]

    print("Top genes for PSM: ", unique_PSM_gene_ranks[:10])
    top_PSM_genes = [each for each in unique_PSM_gene_ranks[:3]]
    plotGeneExpression(umap_latent_data, gene_expr=ann_data[:, top_PSM_genes].X, gene_list=top_PSM_genes)

    print("Top genes for Optic Cup: ", unique_optic_gene_ranks[:10])
    top_optic_genes = [each for each in unique_optic_gene_ranks[:3]]
    plotGeneExpression(umap_latent_data, gene_expr=ann_data[:, top_optic_genes].X, gene_list=top_optic_genes)

# ======================================================
from sklearn.ensemble import RandomForestClassifier
def trainClassifier(X, Y):
    clf_model = RandomForestClassifier(n_estimators=100)
    clf_model.fit(X, Y)
    score = clf_model.score(X, Y)
    print("Train accuracy = {}".format(score))
    return clf_model


def perturbedPredict(latent_ode_model, start_data, tps, n_cells):
    latent_ode_model.eval()
    start_data_tensor = torch.FloatTensor(start_data)
    latent_seq, drift_seq, recon_obs = latent_ode_model.computeDiffLatentDrift([start_data_tensor], tps, n_cells)
    latent_seq = [latent_seq[:, t, :].detach().numpy() for t in range(latent_seq.shape[1])]
    recon_obs = [recon_obs[:, t, :].detach().numpy() for t in range(recon_obs.shape[1])]
    return latent_seq, drift_seq, recon_obs


def compareZebrafishPerturb(umap_latent_data, umap_PSM_data, umap_optic_data, color_list):
    concat_umap_scatter = np.concatenate(umap_latent_data)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    ax1.scatter(
        concat_umap_scatter[:, 0], concat_umap_scatter[:, 1],
        color=gray_color, s=10, alpha=0.5
    )
    ax2.scatter(
        concat_umap_scatter[:, 0], concat_umap_scatter[:, 1],
        color=gray_color, s=10, alpha=0.5
    )
    ax1.set_title("Perturb TBX16")
    ax2.set_title("Perturb SOX3")
    for t in range(len(umap_PSM_data)):
        ax1.scatter(
            umap_PSM_data[t][:, 0], umap_PSM_data[t][:, 1],
            color=color_list[t], s=15, alpha=0.9, label=t
        )
        ax2.scatter(
            umap_optic_data[t][:, 0], umap_optic_data[t][:, 1],
            color=color_list[t], s=15, alpha=0.9, label=t
        )
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1, fontsize=12)
    plt.tight_layout()
    plt.show()
    # -----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    ax1.scatter(
        concat_umap_scatter[:, 0], concat_umap_scatter[:, 1],
        color=gray_color, s=10, alpha=0.5
    )
    ax2.scatter(
        concat_umap_scatter[:, 0], concat_umap_scatter[:, 1],
        color=gray_color, s=10, alpha=0.5
    )
    ax1.set_title("Perturb TBX16")
    ax2.set_title("Perturb SOX3")
    ax1.scatter(
        umap_PSM_data[-1][:, 0], umap_PSM_data[-1][:, 1],
        color=color_list[-1], s=15, alpha=0.9
    )
    ax2.scatter(
        umap_optic_data[-1][:, 0], umap_optic_data[-1][:, 1],
        color=color_list[-1], s=15, alpha=0.9
    )
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1, fontsize=12)
    plt.tight_layout()
    plt.show()


# ======================================================
def plotTFPrecision():
    # Load gene list
    TF_list = pd.read_csv("../data/TF/Danio_rerio_TF.txt", index_col=None, header=0, sep="\t")
    key_gene_dict = np.load("../res/downstream_analysis/vector_field/path_marker_genes.npy", allow_pickle=True).item()
    PSM_genes = key_gene_dict["PSM"]
    Optic_genes = key_gene_dict["Optic"]
    TF_gene_list = [str.lower(g) for g in TF_list.Symbol.values.tolist()]
    PSM_genes = [str.lower(g) for g in PSM_genes]
    Optic_genes = [str.lower(g) for g in Optic_genes]
    all_genes = [str.lower(g) for g in ann_data.var_names.values]
    # compute precision = TP/PP
    k_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    PSM_pre_list = []
    Optic_pre_list = []
    random_pre_list = []
    overlap_rate_list = []
    PSM_random_overlap_rate_list = []
    Optic_random_overlap_rate_list = []
    for k in k_list:
        PSM_top = PSM_genes[:k]
        Optic_top = Optic_genes[:k]
        PSM_TP = set.intersection(set(TF_gene_list), set(PSM_top))
        Optic_TP = set.intersection(set(TF_gene_list), set(Optic_top))
        PSM_pre = len(PSM_TP) / k
        Optic_pre = len(Optic_TP) / k
        path_overlap = set.intersection(set(Optic_top), set(PSM_top))
        path_overlap_rate = len(path_overlap) / k
        # -----
        n_trials = 10
        rand_pre = 0.0
        PSM_rand_overlap_rate = 0.0
        Optic_rand_overlap_rate = 0.0
        for t in range(n_trials):
            rand_top = np.asarray(all_genes)[np.random.choice(np.arange(len(all_genes)), k, replace=False)].tolist()
            rand_TP = set.intersection(set(TF_gene_list), set(rand_top))
            rand_pre += len(rand_TP) / k
            PSM_rand_overlap = set.intersection(set(rand_top), set(PSM_top))
            Optic_rand_overlap = set.intersection(set(rand_top), set(Optic_TP))
            PSM_rand_overlap_rate += len(PSM_rand_overlap) / k
            Optic_rand_overlap_rate += len(Optic_rand_overlap) / k
        rand_pre = rand_pre / n_trials
        PSM_rand_overlap_rate = PSM_rand_overlap_rate / n_trials
        Optic_rand_overlap_rate = Optic_rand_overlap_rate / n_trials
        # -----
        PSM_pre_list.append(PSM_pre)
        Optic_pre_list.append(Optic_pre)
        random_pre_list.append(rand_pre)
        overlap_rate_list.append(path_overlap_rate)
        PSM_random_overlap_rate_list.append(PSM_rand_overlap_rate)
        Optic_random_overlap_rate_list.append(Optic_rand_overlap_rate)
    np.save(
        "../res/downstream_analysis/vector_field/path_marker_genes-precision.npy",
        {
            "PSM_pre_list": PSM_pre_list,
            "Optic_pre_list": Optic_pre_list,
            "random_pre_list": random_pre_list,
            "overlap_rate_list": overlap_rate_list,
            "PSM_random_overlap_rate_list": PSM_random_overlap_rate_list,
            "Optic_random_overlap_rate_list": Optic_random_overlap_rate_list,
        }
    )
    plt.figure(figsize=(7, 3))
    width = 0.3
    plt.bar(
        np.arange(len(k_list)) - width, random_pre_list,
        width=width, color=gray_color, label="random", edgecolor="k", linewidth=0.2
    )
    plt.bar(
        np.arange(len(k_list)), PSM_pre_list,
        width=width, color=Bold_10.mpl_colors[0], label="PSM", edgecolor="k", linewidth=0.2
    )
    plt.xticks(np.arange(len(k_list)) - width / 2, k_list)
    plt.xlabel("# DE Genes")
    plt.ylabel("TF Precision")
    plt.ylim(0., 0.4)
    plt.legend(ncol=2, fontsize=12)
    _removeTopRightBorders()
    plt.tight_layout()
    plt.savefig("../res/figs/PSM_TF.pdf")
    plt.show()

    plt.figure(figsize=(7, 3))
    width = 0.3
    plt.bar(
        np.arange(len(k_list)) - width, random_pre_list,
        width=width, color=gray_color, label="random", edgecolor="k", linewidth=0.2
    )
    plt.bar(
        np.arange(len(k_list)), Optic_pre_list,
        width=width, color=Bold_10.mpl_colors[1], label="Optic Cup", edgecolor="k", linewidth=0.2
    )
    plt.xticks(np.arange(len(k_list)) - width / 2, k_list)
    plt.xlabel("# Marker Genes")
    plt.ylabel("TF Precision")
    plt.ylim(0., 0.4)
    plt.legend(ncol=2, fontsize=12)
    _removeTopRightBorders()
    plt.tight_layout()
    plt.savefig("../res/figs/Optic_TF.pdf")
    plt.show()


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
        print("Visualize latent space...")
        (
            latent_seq, next_seq, drift_seq, drift_magnitude,
            umap_latent_data, umap_next_data, umap_model, latent_tp_list
        ) = visVectorField(traj_data, latent_ode_model, traj_cell_types)
        # -----
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
        # # -----
        # plotLatent(umap_latent_data, umap_latent_data, umap_next_data, color_list = linearSegmentCMap(n_tps, "viridis"))
        # plotLatentCellType(umap_latent_data, umap_latent_data, umap_next_data, traj_cell_types, color_list = Bold_10.mpl_colors)
        # # -----
        # # Visualize latent space w/ stream
        # umap_scatter_data = umap_latent_data
        # color_list = linearSegmentCMap(n_tps, "viridis")
        # plotStream(umap_scatter_data, umap_latent_data, umap_next_data, color_list, num_sep=200, n_neighbors=20)
        # if traj_cell_types is not None:
        #     plotStreamByCellType(umap_scatter_data, umap_latent_data, umap_next_data, traj_cell_types, num_sep=200,
        #                          n_neighbors=5)
        # # -----
        # # Construct lineage graph
        # color_list = linearSegmentCMap(n_tps, "viridis")
        # pg_tree = constructLineageGraph(umap_latent_data, num_nodes=10)
        # plotLineageGraph(umap_latent_data, pg_tree, traj_cell_types, color_list)
        # # -----
        # # Find fixed point
        # starting_list, root_list = findFixedPoint(latent_ode_model, traj_data, n_points=20)
        # feasible_root = np.asarray([each[0] for each in root_list if each[2] == 1])
        # print("The num of feasible roots = {}".format(feasible_root.shape[0]))
        # jacobian_list, divergence = assignFixedPointType(latent_ode_model, feasible_root)
        # print("Divergence: {}".format(divergence))
        # umap_root_data = umap_model.transform(feasible_root)
        # color_list = linearSegmentCMap(n_tps, "viridis")
        # plotStreamWithFixedPoint(
        #     umap_latent_data, umap_latent_data, umap_next_data, umap_root_data, color_list,
        #     num_sep=200, n_neighbors=20
        # )
        # # -----
        # # Least action path
        # PSM_cell_idx = np.where(cell_types == "PSM")[0] # PSM cells
        # optic_cell_idx = np.where(cell_types == "Optic Cup")[0] # Optic Cup cells
        # start_cell_idx = np.where(cell_tps == 1)[0]
        # print("PSM={} | Optic Cup={} | Starting={}".format(len(PSM_cell_idx), len(optic_cell_idx), len(start_cell_idx)))
        # concat_latent_seq = np.concatenate(latent_seq, axis=0)
        # start_cell_mean = np.mean(concat_latent_seq[start_cell_idx,:], axis=0)
        # PSM_cell_mean = np.mean(concat_latent_seq[PSM_cell_idx,:], axis=0)
        # optic_cell_mean = np.mean(concat_latent_seq[optic_cell_idx,:], axis=0)
        # path_length = 8
        # PSM_dt, PSM_P, PSM_action_list, PSM_dt_list, PSM_P_list = leastActionPath(
        #     x_0=start_cell_mean, x_T=PSM_cell_mean,
        #     path_length=path_length, vec_field=latent_ode_model.diffeq_decoder.net, D=1, iters=10
        # )
        # optic_dt, optic_P, optic_action_list, optic_dt_list, optic_P_list = leastActionPath(
        #     x_0=start_cell_mean, x_T=optic_cell_mean,
        #     path_length=path_length, vec_field=latent_ode_model.diffeq_decoder.net, D=1, iters=10
        # )
        #
        # # Recover path back to gene space
        # use_KNN = True
        # if use_KNN:
        #     PSM_KNN_idx = pathKNNGenes(PSM_P, concat_latent_seq, K=10)
        #     optic_KNN_idx = pathKNNGenes(optic_P, concat_latent_seq, K=10)
        #     concat_traj_data = np.concatenate([each.detach().numpy() for each in traj_data], axis=0)
        #     PSM_gene = np.concatenate([concat_traj_data[idx, :] for idx in PSM_KNN_idx], axis=0)
        #     optic_gene =np.concatenate([concat_traj_data[idx, :] for idx in optic_KNN_idx], axis=0)
        # else:
        #     PSM_gene = _map2GeneSpace(PSM_P, latent_ode_model.obs_decoder.net)
        #     optic_gene = _map2GeneSpace(optic_P, latent_ode_model.obs_decoder.net)
        # expr_mat = np.concatenate([PSM_gene, optic_gene], axis=0)
        # cell_idx = ["cell_{}".format(i) for i in range(expr_mat.shape[0])]
        # # cell_tp = [t for t in range(8)] + [t for t in range(8)]
        # cell_types = ["PSM" for t in range(PSM_gene.shape[0])] + ["Optic" for t in range(optic_gene.shape[0])]
        # cell_df = pd.DataFrame(data=np.zeros((expr_mat.shape[0], 2)), index=cell_idx, columns=["TP", "TYPE"])
        # # cell_df.TP = cell_tp
        # cell_df.TYPE = cell_types
        # expr_df = pd.DataFrame(data=expr_mat, index=cell_idx, columns=ann_data.var_names.values)
        # path_ann = scanpy.AnnData(X=expr_df, obs=cell_df)
        # scanpy.tl.rank_genes_groups(path_ann, 'TYPE', method="wilcoxon") # logreg, wilcoxon
        # scanpy.pl.rank_genes_groups(path_ann, n_genes=25, sharey=False, fontsize=12)
        # group_id = path_ann.uns["rank_genes_groups"]["names"].dtype.names
        # gene_names = path_ann.uns["rank_genes_groups"]["names"]
        # print(group_id)
        # print(gene_names[:10])
        # marker_gene_dict = {}
        # for i, g_name in enumerate(group_id):
        #     g_gene = [x[i] for x in gene_names]
        #     marker_gene_dict[g_name] = g_gene
        # np.save("../res/downstream_analysis/vector_field/path_marker_genes.npy", marker_gene_dict, allow_pickle=True)

        # # top_PSM_genes = ['PRICKLE1B', 'ITM2CB', 'TBX16'] # path
        # # top_PSM_genes = ['TBX16', 'ANP32E', 'TOB1A'] # KNN
        # top_PSM_genes = ['TBX16', 'TOB1A'] # KNN
        # plotGeneExpression(umap_latent_data, gene_expr=ann_data[:, top_PSM_genes].X, gene_list=top_PSM_genes)
        # # top_PSM_genes = ['TBX16'] # KNN
        # # plotOneGeneExpression(umap_latent_data, gene_expr=ann_data[:, top_PSM_genes].X, gene_list=top_PSM_genes)

        # # top_optic_genes = ['SOX19A', 'SOX3', 'ZIC3'] # path
        # top_optic_genes = ['SI:CH211-152C2.3', 'SOX3', 'ALDOB'] # KNN
        # plotGeneExpression(umap_latent_data, gene_expr=ann_data[:, top_optic_genes].X, gene_list=top_optic_genes)

        # # Plot path
        # umap_PSM_P = umap_model.transform(PSM_P)
        # umap_optic_P = umap_model.transform(optic_P)
        # plotZebrafishLAP(
        #     umap_latent_data, umap_PSM_P, umap_optic_P,
        #     PSM_cell_idx, optic_cell_idx, start_cell_idx
        # )
        # # Diver genes
        # PSM_gene_ranks = rankGenesByMSD(PSM_P, latent_ode_model.obs_decoder.net, ann_data.var_names.values)
        # optic_gene_ranks = rankGenesByMSD(optic_P, latent_ode_model.obs_decoder.net, ann_data.var_names.values)
        # plotZebrafishTopGenes(ann_data, PSM_gene_ranks, optic_gene_ranks)

        # ------
        # Train the classifier
        X = traj_data[-1].detach().numpy()
        # Y = ["other" if x != "PSM" and x != "Optic Cup" else str(x) for x in traj_cell_types[-1]]
        Y = ["other" if x != "PSM" else str(x) for x in traj_cell_types[-1]]
        clf_model = trainClassifier(X, Y)
        # Perturbation analysis
        m = 100
        v = 5
        start_data = traj_data[0].detach().numpy()
        TBX16_idx = ann_data.var_names.values.tolist().index("TBX16")
        # SOX3_idx = ann_data.var_names.values.tolist().index("SOX3")
        TBX16_perturbed_start = copy.deepcopy(start_data)
        TBX16_perturbed_start[:, TBX16_idx] *= m
        # TBX16_perturbed_start[:, TBX16_idx] = v
        # SOX3_perturbed_start = copy.deepcopy(start_data)
        # SOX3_perturbed_start[:, SOX3_idx] *= m
        # SOX3_perturbed_start[:, SOX3_idx] = v

        n_cells = traj_data[-1].shape[0]
        TBX16_latent_seq, TBX16_drift_seq, TBX16_recon_obs = perturbedPredict(latent_ode_model, TBX16_perturbed_start, train_tps, n_cells=n_cells)
        # SOX3_latent_seq, SOX3_drift_seq, SOX3_recon_obs = perturbedPredict(latent_ode_model, SOX3_perturbed_start, train_tps, n_cells=n_cells)

        TBX16_Y = clf_model.predict(TBX16_recon_obs[-1])
        # SOX3_Y = clf_model.predict(SOX3_recon_obs[-1])
        unperturbed_Y = np.asarray(Y)
        # Y_list = [unperturbed_Y, TBX16_Y, SOX3_Y]
        Y_list = [unperturbed_Y, TBX16_Y]
        # c_list = ["PSM", "Optic Cup", "other"]
        c_list = ["PSM", "other"]
        cnt_mat = np.zeros((len(Y_list), len(c_list)))
        for y_idx, y in enumerate(Y_list):
            for c_idx, c in enumerate(c_list):
                if c == "PSM":
                    cnt_mat[y_idx, c_idx] = len(np.where(y == c)[0])
                else:
                    cnt_mat[y_idx, c_idx] = len(np.where(y != "PSM")[0])
        # cnt_mat = cnt_mat / cnt_mat.sum(axis=1)
        # cnt_df = pd.DataFrame(cnt_mat, index=["Unperturbed", "TBX16(PSM)", "SOX3(Optic)"], columns=c_list)
        cnt_mat = cnt_mat / cnt_mat.sum(axis=1)
        # cnt_mat = cnt_mat[:2, :]
        # cnt_mat[:, 1] = 1 - cnt_mat[:, 0] # combines Optic celss and other cells
        # cnt_mat = cnt_mat[:, :2]
        cnt_df = pd.DataFrame(cnt_mat, index=["Unperturbed", "TBX16"], columns=["PSM", "other"])
        print(cnt_df)
        plt.figure(figsize=(5, 3))
        bottom = np.zeros((len(Y_list), ))
        width = 0.7
        color_list = [Bold_10.mpl_colors[0], gray_color]
        for c_idx, c in enumerate(["PSM", "other"]):
            plt.bar(np.arange(len(Y_list)), cnt_mat[:, c_idx], width, label=c, bottom=bottom, color=color_list[c_idx])
            bottom += cnt_mat[:, c_idx]
        plt.ylabel("Ratio")
        plt.ylim(0.0, 1.0)
        plt.xticks(np.arange(len(Y_list)), cnt_df.index.values)
        # plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1, fontsize=12)
        _removeTopRightBorders()
        plt.savefig("../res/figs/PSM_perturbation_cell_ratio.pdf")
        plt.tight_layout()
        plt.show()
        #
        # print("Visualization...")
        # compareZebrafishPerturb(
        #     umap_latent_data,
        #     [umap_model.transform(each) for each in TBX16_latent_seq],
        #     [umap_model.transform(each) for each in SOX3_latent_seq],
        #     # color_list = Bold_10.mpl_colors
        #     color_list = Cube1_12.mpl_colors
        # )

        # ------
        # plotTFPrecision()



