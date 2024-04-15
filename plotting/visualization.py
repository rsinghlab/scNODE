'''
Description:
    Visualization.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import numpy as np
import tabulate
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from plotting.__init__ import *
from plotting import _removeAllBorders, _removeTopRightBorders

# ======================================

def plotPredAllTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps):
    '''Plot predictions at all timepoints.'''
    unique_tps = np.unique(true_cell_tps).astype(int).tolist()
    n_tps = len(unique_tps)
    color_list = linearSegmentCMap(n_tps, "viridis")
    # color_list = Vivid_10.mpl_colors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("True Data", fontsize=15)
    ax2.set_title("Predictions", fontsize=15)
    for i, t in enumerate(unique_tps):
        true_t_idx = np.where(true_cell_tps == t)[0]
        pred_t_idx = np.where(pred_cell_tps == t)[0]
        ax1.scatter(true_umap_traj[true_t_idx, 0], true_umap_traj[true_t_idx, 1], label=t, color=color_list[i], s=20, alpha=1.0)
        ax2.scatter(pred_umap_traj[pred_t_idx, 0], pred_umap_traj[pred_t_idx, 1], label=t, color=color_list[i], s=20, alpha=1.0)
    ax2.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    # plt.tight_layout()
    plt.show()


def plotPredTestTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps, test_tps):
    '''Plot predictions at testing timepoints.'''
    n_tps = len(np.unique(true_cell_tps))
    # color_list = linearSegmentCMap(n_tps, "viridis")
    n_test_tps = len(test_tps)
    # color_list = linearSegmentCMap(n_test_tps, "viridis")
    color_list = Vivid_10.mpl_colors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("True Data", fontsize=15)
    ax2.set_title("Predictions", fontsize=15)
    ax1.scatter(true_umap_traj[:, 0], true_umap_traj[:, 1], label="other", c=gray_color, s=40, alpha=0.5)
    ax2.scatter(true_umap_traj[:, 0], true_umap_traj[:, 1], label="other", c=gray_color, s=40, alpha=0.5)
    for i, t in enumerate(test_tps):
        c = color_list[i]
        true_t_idx = np.where(true_cell_tps == t)[0]
        pred_t_idx = np.where(pred_cell_tps == t)[0]
        ax1.scatter(true_umap_traj[true_t_idx, 0], true_umap_traj[true_t_idx, 1], label=int(t), color=c, s=20, alpha=1.0)
        ax2.scatter(pred_umap_traj[pred_t_idx, 0], pred_umap_traj[pred_t_idx, 1], label=int(t), color=c, s=20, alpha=1.0)
    ax2.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    # plt.tight_layout()
    plt.show()

# ======================================
def compareUMAPTestTime(
        true_umap_traj, model_pred_umap_traj, true_cell_tps, model_cell_tps, test_tps,
        model_list, mdoel_name_dict, data_name, split_type, embed_name, save_dir
):
    n_tps = len(np.unique(true_cell_tps))
    n_models = len(model_list)
    color_list = Vivid_10.mpl_colors
    fig, ax_list = plt.subplots(1, n_models+1, figsize=(14, 4))
    # Plot true data
    ax_list[0].set_title("True Data", fontsize=15)
    ax_list[0].scatter(true_umap_traj[:, 0], true_umap_traj[:, 1], c=gray_color, s=40, alpha=0.5)
    for t_idx, t in enumerate(test_tps):
        c = color_list[t_idx]
        true_t_idx = np.where(true_cell_tps == t)[0]
        ax_list[0].scatter(
            true_umap_traj[true_t_idx, 0], true_umap_traj[true_t_idx, 1],
            label=int(t), color=c, s=20, alpha=0.9
        )
    ax_list[0].set_xticks([])
    ax_list[0].set_yticks([])
    _removeAllBorders(ax_list[0])
    # Plot pred data
    for m_idx, m in enumerate(model_list):
        ax_list[m_idx+1].set_title(mdoel_name_dict[m], fontsize=15)
        ax_list[m_idx+1].scatter(true_umap_traj[:, 0], true_umap_traj[:, 1], c=gray_color, s=40, alpha=0.5)
        for t_idx, t in enumerate(test_tps):
            c = color_list[t_idx]
            pred_t_idx = np.where(model_cell_tps[m_idx] == t)[0]
            ax_list[m_idx+1].scatter(
                model_pred_umap_traj[m_idx][pred_t_idx, 0], model_pred_umap_traj[m_idx][pred_t_idx, 1],
                label=int(t), color=c, s=20, alpha=0.9)
        ax_list[m_idx+1].set_xticks([])
        ax_list[m_idx+1].set_yticks([])
        _removeAllBorders(ax_list[m_idx+1])
    ax_list[-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="Test TPs", title_fontsize=14, fontsize=13)
    plt.tight_layout()
    # plt.savefig("{}/{}-{}-{}.pdf".format(save_dir, data_name, split_type, embed_name))
    # plt.savefig("{}/{}-{}-{}.png".format(save_dir, data_name, split_type, embed_name))
    plt.show()

# ======================================


def _plotTable(df_data):
    print(tabulate.tabulate(
        df_data, headers=[df_data.index.names[0] + '/' + df_data.columns.names[1]] + list(df_data.columns),
        tablefmt="grid"
    ))


def printMetric(model_ot, model_list, column_names):
    print("\n" * 2)
    metric_df = pd.DataFrame(
        data=model_ot,
        index=pd.MultiIndex.from_tuples([
            ("OT", m) for m in model_list
        ], names=("Metric", "Model")),
        columns=pd.MultiIndex.from_tuples(column_names, names=("Type", "TP"))
    )
    _plotTable(metric_df)


# ======================================

def computeDrift(traj_data, latent_ode_model):
    # Compute latent and drift seq
    print("Computing latent and drift sequence...")
    latent_seq, _ = latent_ode_model.vaeReconstruct(traj_data)
    drift_seq = [latent_ode_model.computeDrift(each_latent).detach().numpy() for each_latent in latent_seq]
    latent_seq = [each_latent.detach().numpy() for each_latent in latent_seq]
    next_seq = [each + 0.1 * drift_seq[i] for i, each in enumerate(latent_seq)]
    return latent_seq, next_seq, drift_seq


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
