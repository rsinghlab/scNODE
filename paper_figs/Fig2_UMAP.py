'''
Description:
    Compare model predictions with UMAP embeddings.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import numpy as np
from plotting import *
from plotting import _removeAllBorders


mdoel_name_dict = {
    "latent_ODE_OT_pretrain": "scNODE",
    "scNODE": "scNODE",
    "PRESCIENT": "PRESCIENT",
    "MIOFlow": "MIOFlow",
}


def compareUMAPTestTime(
        true_umap_traj, model_pred_umap_traj,
        true_cell_tps, model_cell_tps, test_tps, model_list, wot_n_tps, task):
    n_tps = len(np.unique(true_cell_tps))
    n_models = len(model_list)
    color_list = Vivid_10.mpl_colors
    fig, ax_list = plt.subplots(1, n_models+1, figsize=(12, 2.5))
    # Plot true data
    bg_s_size = 20
    s_size = 5
    ax_list[0].set_title("True Data", fontsize=15)
    # ax_list[0].scatter(true_umap_traj[:, 0], true_umap_traj[:, 1], c=gray_color, s=40, alpha=0.5)
    ax_list[0].scatter(true_umap_traj[:, 0], true_umap_traj[:, 1], c=gray_color, s=bg_s_size, alpha=0.5)
    for t_idx, t in enumerate(test_tps):
        c = color_list[t_idx]
        true_t_idx = np.where(true_cell_tps == t)[0]
        ax_list[0].scatter(
            true_umap_traj[true_t_idx, 0], true_umap_traj[true_t_idx, 1],
            label=int(t), color=c, s=s_size, alpha=0.9
        )
    ax_list[0].set_xticks([])
    ax_list[0].set_yticks([])
    _removeAllBorders(ax_list[0])
    # Plot pred data
    for m_idx, m in enumerate(model_list):
        ax_list[m_idx+1].set_title(mdoel_name_dict[m], fontsize=15)
        ax_list[m_idx+1].scatter(true_umap_traj[:, 0], true_umap_traj[:, 1], c=gray_color, s=bg_s_size, alpha=0.5)
        for t_idx, t in enumerate(test_tps):
            if m == "WOT" and t >= wot_n_tps:
                continue
            c = color_list[t_idx]
            pred_t_idx = np.where(model_cell_tps[m_idx] == t)[0]
            ax_list[m_idx+1].scatter(
                model_pred_umap_traj[m_idx][pred_t_idx, 0], model_pred_umap_traj[m_idx][pred_t_idx, 1], color=c, alpha=0.9,
                # label=int(t),
                s=s_size,
            )
            ax_list[m_idx + 1].scatter(
                [], [],
                label=int(t), color=c, s=50, alpha=0.9)
        ax_list[m_idx+1].set_xticks([])
        ax_list[m_idx+1].set_yticks([])
        _removeAllBorders(ax_list[m_idx+1])
    ax_list[-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="Test TPs", title_fontsize=14, fontsize=13, ncol=2 if data_name == "wot" and task == "hard" else 1)
    plt.tight_layout()
    plt.show()


# ================================================

def plotForEasyTask(data_name):
    if data_name in ["zebrafish", "drosophila", "wot"]:
        split_type= "three_interpolation"
    print("[ {}-{} ] Compare Predictions".format(data_name, split_type))
    embed_name = "pca_umap"
    res = np.load("../res/low_dim/{}-{}-pca_umap.npy".format(data_name, split_type), allow_pickle=True).item()
    true_umap_traj = res["true"]
    model_pred_umap_traj = res["pred"]
    model_list = res["model"]
    embed_name = res["embed_name"]
    true_cell_tps = res["true_cell_tps"]
    model_cell_tps = res["model_cell_tps"]
    test_tps = res["test_tps"]
    wot_n_tps = None
    print("Visualization")
    model_list = ["scNODE", "MIOFlow", "PRESCIENT"]
    compareUMAPTestTime(true_umap_traj, model_pred_umap_traj, true_cell_tps, model_cell_tps, test_tps, model_list, wot_n_tps, task="easy")


def plotForMediumTask(data_name):
    if data_name in ["zebrafish"]:
        split_type = "two_forecasting"
    elif data_name in ["drosophila", "wot"]:
        split_type = "three_forecasting"
    print("[ {}-{} ] Compare Predictions".format(data_name, split_type))
    embed_name = "pca_umap"
    res = np.load("../res/low_dim/{}-{}-pca_umap.npy".format(data_name, split_type), allow_pickle=True).item()
    true_umap_traj = res["true"]
    model_pred_umap_traj = res["pred"]
    model_list = res["model"]
    embed_name = res["embed_name"]
    true_cell_tps = res["true_cell_tps"]
    model_cell_tps = res["model_cell_tps"]
    test_tps = res["test_tps"]
    wot_n_tps = None
    print("Visualization")
    model_list = ["scNODE", "MIOFlow", "PRESCIENT"]
    compareUMAPTestTime(true_umap_traj, model_pred_umap_traj, true_cell_tps, model_cell_tps, test_tps, model_list, wot_n_tps, task="medium")


def plotForHardTask(data_name):
    split_type = "remove_recovery"
    print("[ {}-{} ] Compare Predictions".format(data_name, split_type))
    embed_name = "pca_umap"
    res = np.load("../res/remove_recovery/{}-{}-{}.npy".format(data_name, split_type, embed_name), allow_pickle=True).item()
    true_umap_traj = res["true"]
    model_pred_umap_traj = res["pred"]
    model_list = res["model"]
    embed_name = res["embed_name"]
    true_cell_tps = res["true_cell_tps"]
    model_cell_tps = res["model_cell_tps"]
    test_tps = res["test_tps"]
    wot_n_tps = res["wot_n_tps"]
    print("Visualization")
    model_list = ["scNODE", "MIOFlow", "PRESCIENT"]
    compareUMAPTestTime(true_umap_traj, model_pred_umap_traj, true_cell_tps, model_cell_tps, test_tps, model_list, wot_n_tps, task="hard")




if __name__ == '__main__':
    data_name = "wot"  # zebrafish, drosophila, wot
    plotForEasyTask(data_name)
    plotForMediumTask(data_name)
    plotForHardTask(data_name)

