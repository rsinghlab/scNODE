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
        # ax_list[m_idx+1].set_title("{} \n miLISI={:.2f}".format(mdoel_name_dict[m], miLISI), fontsize=15)
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


def plotMetricBar(metric_path, dataset_list, inter_model_list, extra_model_list, dataset_name_dict, mdoel_name_dict, save_dir):
    inter_avg_ot = []
    inter_std_ot = []
    inter_avg_l2 = []
    inter_std_l2 = []
    extra_avg_ot = []
    extra_std_ot = []
    extra_avg_l2 = []
    extra_std_l2 = []
    for data_name in dataset_list:
        # interpolation
        if data_name in ["embryoid", "pancreatic"]:
            inter_split = "one_interpolation"
        if data_name in ["zebrafish", "mammalian", "drosophila", "WOT"]:
            inter_split = "three_interpolation"
        metric_res = np.load("{}/{}-{}-model_metrics.npy".format(metric_path, data_name, inter_split), allow_pickle=True).item()
        tmp_ot = np.asarray([[metric_res[t][m]["global"]["ot"] if m in metric_res[t] else np.nan for m in inter_model_list] for t in metric_res])
        tmp_l2 = np.asarray([[metric_res[t][m]["global"]["l2"] if m in metric_res[t] else np.nan for m in inter_model_list] for t in metric_res])
        inter_avg_ot.append(np.nanmean(tmp_ot, axis=0))
        inter_std_ot.append(np.nanstd(tmp_ot, axis=0))
        inter_avg_l2.append(np.nanmean(tmp_l2, axis=0))
        inter_std_l2.append(np.nanstd(tmp_l2, axis=0))
        # extrapolation
        if data_name in ["embryoid", "pancreatic"]:
            extra_split = "one_forecasting"
        if data_name in ["mammalian", "drosophila", "WOT"]:
            extra_split = "three_forecasting"
        if data_name in ["zebrafish"]:
            extra_split = "two_forecasting"
        metric_res = np.load("{}/{}-{}-model_metrics.npy".format(metric_path, data_name, extra_split), allow_pickle=True).item()
        tmp_ot = np.asarray([[metric_res[t][m]["global"]["ot"] if m in metric_res[t] else np.nan for m in extra_model_list] for t in metric_res])
        tmp_l2 = np.asarray([[metric_res[t][m]["global"]["l2"] if m in metric_res[t] else np.nan for m in extra_model_list] for t in metric_res])
        extra_avg_ot.append(np.nanmean(tmp_ot, axis=0))
        extra_std_ot.append(np.nanmean(tmp_ot, axis=0))
        extra_avg_l2.append(np.nanmean(tmp_l2, axis=0))
        extra_std_l2.append(np.nanstd(tmp_l2, axis=0))
    # -----
    inter_avg_ot = np.asarray(inter_avg_ot)
    extra_avg_ot = np.asarray(extra_avg_ot)
    color_list = Vivid_10.mpl_colors
    fig, ax_list = plt.subplots(1, 2, figsize=(12, 4))
    inter_width = 0.1
    x_base = np.arange(len(dataset_list)) - inter_width*len(inter_model_list)
    ax_list[0].set_title("Interpolation")
    for i in range(len(inter_model_list)):
        ax_list[0].bar(
            x=x_base+i*inter_width,
            height=inter_avg_ot[:, i],
            width=inter_width,
            align="edge",
            color=model_colors_dict[inter_model_list[i]],
            label=mdoel_name_dict[inter_model_list[i]],
            edgecolor="k",
            linewidth=0.2,
        )
        # ax_list[0].legend(fontsize=12)
    ax_list[0].set_ylim(0.0, 600)
    ax_list[0].set_yticks([0, 200, 400, 600])
    ax_list[0].set_yticklabels(["0", "200", "400", r"$\geq$600"])
    ax_list[0].set_xticks([])
    ax_list[0].set_ylabel("Wasserstein")
    ax_list[0].set_xlabel("Dataset")
    ax_list[0].set_xticks(np.arange(len(dataset_list)) - inter_width * len(inter_model_list) / 2)
    ax_list[0].set_xticklabels([dataset_name_dict[x] for x in dataset_list])
    _removeTopRightBorders(ax_list[0])
    extra_width = 0.2
    x_base = np.arange(len(dataset_list)) - extra_width * len(extra_model_list)
    ax_list[1].set_title("Extrapolation")
    for i in range(len(extra_model_list)):
        ax_list[1].bar(
            x=x_base + i * extra_width,
            height=extra_avg_ot[:, i],
            width=extra_width,
            align="edge",
            color=model_colors_dict[extra_model_list[i]],
            label=mdoel_name_dict[extra_model_list[i]],
            edgecolor="k",
            linewidth=0.2,
        )
        # ax_list[1].legend(fontsize=12)
    ax_list[1].set_ylim(0.0, 800)
    ax_list[1].set_yticks([0, 200, 400, 600, 800])
    ax_list[1].set_yticklabels(["0", "200", "400", "600", r"$\geq$800"])
    # ax_list[1].set_ylabel("Wasserstein")
    _removeTopRightBorders(ax_list[1])
    plt.xticks(np.arange(len(dataset_list)) - extra_width*len(extra_model_list)/2, [dataset_name_dict[x] for x in dataset_list])
    plt.xlabel("Dataset")
    # ax_list[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.1), title_fontsize=14, fontsize=10, ncol=3)
    # ax_list[0].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=11)
    plt.subplots_adjust(hspace=2.5)
    plt.tight_layout()
    # plt.savefig("{}/exp_metric_bar_hori.pdf".format(save_dir))
    plt.show()
    # -----
    inter_avg_l2 = np.asarray(inter_avg_l2)
    extra_avg_l2 = np.asarray(extra_avg_l2)
    color_list = Vivid_10.mpl_colors
    fig, ax_list = plt.subplots(1, 2, figsize=(12, 4))
    inter_width = 0.1
    x_base = np.arange(len(dataset_list)) - inter_width*len(inter_model_list)
    ax_list[0].set_title("Interpolation")
    for i in range(len(inter_model_list)):
        bar1 = ax_list[0].bar(
            x=x_base+i*inter_width,
            height=inter_avg_l2[:, i],
            width=inter_width,
            align="edge",
            color=model_colors_dict[inter_model_list[i]],
            label=mdoel_name_dict[inter_model_list[i]],
            edgecolor="k",
            linewidth=0.2,
        )
        # ax_list[0].legend(fontsize=12)
    ax_list[0].set_ylim(0.0, 50)
    ax_list[0].set_yticks([0, 20, 40, 50])
    ax_list[0].set_yticklabels(["0", "20", "40", r"$\geq$50"])
    ax_list[0].set_xticks([])
    ax_list[0].set_ylabel("L2")
    ax_list[0].set_xlabel("Dataset")
    _removeTopRightBorders(ax_list[0])
    ax_list[0].set_xticks(np.arange(len(dataset_list)) - inter_width * len(inter_model_list) / 2)
    ax_list[0].set_xticklabels([dataset_name_dict[x] for x in dataset_list])
    plt.xlabel("Dataset")
    extra_width = 0.2
    x_base = np.arange(len(dataset_list)) - extra_width * len(extra_model_list)
    ax_list[1].set_title("Extrapolation")
    for i in range(len(extra_model_list)):
        bar2 = ax_list[1].bar(
            x=x_base + i * extra_width,
            height=extra_avg_l2[:, i],
            width=extra_width,
            align="edge",
            color=model_colors_dict[extra_model_list[i]],
            label=mdoel_name_dict[extra_model_list[i]],
            edgecolor="k",
            linewidth=0.2,
        )
        # ax_list[1].legend(fontsize=12)
    ax_list[1].set_ylim(0.0, 50)
    ax_list[1].set_yticks([0, 20, 40, 50])
    ax_list[1].set_yticklabels(["0", "20", "40", r"$\geq$50"])
    _removeTopRightBorders(ax_list[1])
    plt.xticks(np.arange(len(dataset_list)) - extra_width*len(extra_model_list)/2, [dataset_name_dict[x] for x in dataset_list])
    plt.xlabel("Dataset")
    handles, labels = ax_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.25, 1), fontsize=12, ncol=5)

    plt.subplots_adjust(hspace=2.5)
    plt.tight_layout()
    # plt.savefig("{}/exp_l2_bar_hori.pdf".format(save_dir))
    plt.show()


# ======================================


def _plotTable(df_data):
    print(tabulate.tabulate(
        df_data, headers=[df_data.index.names[0] + '/' + df_data.columns.names[1]] + list(df_data.columns),
        tablefmt="grid"
    ))


def printMetric(model_l2, model_ot, model_list, column_names):
    print("\n" * 2)
    metric_df = pd.DataFrame(
        data=model_l2,
        index=pd.MultiIndex.from_tuples([
            ("L2", m) for m in model_list
        ], names=("Metric", "Model")),
        columns=pd.MultiIndex.from_tuples(column_names, names=("Type", "TP"))
    )
    _plotTable(metric_df)

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
