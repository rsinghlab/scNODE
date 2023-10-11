'''
Description:
    Visualization.
'''
import numpy as np
import scanpy
from umap import UMAP
from sklearn.decomposition import PCA
from plotting.__init__ import *
from plotting.utils import linearSegmentCMap
from benchmark.BenchmarkUtils import traj2Ann, ann2traj


def plotLoss(loss_list):
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    plt.title("Overall loss")
    plt.plot([each[0] for each in loss_list])
    plt.subplot(3, 1, 2)
    plt.title("Recon loss")
    plt.plot([each[1] for each in loss_list])
    plt.subplot(3, 1, 3)
    plt.title("KL loss")
    plt.plot([each[2] for each in loss_list])
    plt.tight_layout()
    plt.show()


def plotRecon(true_data, recon_data, tps):
    n_trajs, n_tps, n_features = true_data.shape
    plt.figure(figsize=(14, 5))
    for t in range(n_tps):
        plt.subplot(1, n_tps, t+1)
        plt.title("t={:.2f}".format(tps[t]))
        for f in range(n_features):
            true_feature = true_data[:, t, f]
            recon_feature = recon_data[:, t, f]
            plt.scatter(x=true_feature, y=recon_feature, s=50, c=Vivid_10.mpl_colors[f], alpha=0.6, label=f)
        plt.xlabel("True")
        plt.ylabel("Est.")
        plt.legend()
    plt.tight_layout()
    plt.show()


def plotMSEPerTime(mse_per_tp, tps):
    plt.figure(figsize=(8, 4))
    plt.title("MSE per time point")
    plt.plot(tps, mse_per_tp, "o-", lw=3, ms=10)
    plt.tight_layout()
    plt.show()


def plotUMAPSingle(ann_data, n_pc, cell_feature, title=""):
    print("PCA...")
    scanpy.tl.pca(ann_data, svd_solver="arpack", n_comps=n_pc)
    print("UMAP...")
    scanpy.pp.neighbors(ann_data, n_neighbors=10, n_pcs=n_pc)
    scanpy.tl.umap(ann_data, min_dist=1.0, spread=1.0)
    scanpy.pl.umap(
        ann_data,
        color=cell_feature,
        title=title
    )


def comparePredTrajWithTruth(pred_traj, true_traj, n_pc):
    color_col = "time_point"
    pred_ann = traj2Ann(pred_traj)
    true_ann = traj2Ann(true_traj)

    plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(1, 2, 1)
    scanpy.tl.pca(true_ann, svd_solver="arpack", n_comps=n_pc)
    scanpy.pp.neighbors(true_ann, n_neighbors=10, n_pcs=n_pc)
    scanpy.tl.umap(true_ann, min_dist=1.0, spread=1.0)
    scanpy.pl.umap(true_ann, color=color_col, title="True Traj", ax = ax1, show=False)
    ax2 = plt.subplot(1, 2, 2)
    scanpy.tl.pca(pred_ann, svd_solver="arpack", n_comps=n_pc)
    scanpy.pp.neighbors(pred_ann, n_neighbors=10, n_pcs=n_pc)
    scanpy.tl.umap(pred_ann, min_dist=1.0, spread=1.0)
    scanpy.pl.umap(pred_ann, color=color_col, title="Pred Traj", ax=ax2, show=False)
    plt.tight_layout()
    plt.show()


# -------

def umapEmbedding(traj_data, n_pcs):
    pca_model = PCA(n_components=50, svd_solver="arpack")
    umap_model = UMAP(n_components=n_pcs)
    umap_traj_data = umap_model.fit_transform(pca_model.fit_transform(traj_data))
    return umap_traj_data, pca_model, umap_model


def plotUMAP(true_umap_traj, pred_umap_traj):
    plt.figure(figsize=(12, 5))
    # Plot truth
    plt.subplot(1, 2, 1)
    plt.title("True Data", fontsize=15)
    plt.scatter(true_umap_traj[:, 0], true_umap_traj[:, 1], label="true", c=BlueRed_12.mpl_colors[0], s=40, alpha=0.8)
    # Plot predictions
    plt.subplot(1, 2, 2)
    plt.title("Predictions", fontsize=15)
    plt.scatter(true_umap_traj[:, 0], true_umap_traj[:, 1], c=gray_color, s=40, alpha=0.5)
    plt.scatter(pred_umap_traj[:, 0], pred_umap_traj[:, 1], c=BlueRed_12.mpl_colors[-1], s=40, alpha=0.8)
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.show()


def plotUMAPFrame(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps):
    def _plotAll():
        plt.scatter(true_umap_traj[:, 0], true_umap_traj[:, 1], label="other", c=gray_color, s=40, alpha=0.5)
    # -----
    n_tps = len(np.unique(true_cell_tps))
    for t in range(n_tps):
        plt.figure(figsize=(12, 5))
        plt.suptitle("[ t={} ]".format(t))
        true_t_idx = np.where(true_cell_tps == t)[0]
        pred_t_idx = np.where(pred_cell_tps == t)[0]
        # Plot truth
        plt.subplot(1, 2, 1)
        plt.title("True Data", fontsize=15)
        _plotAll()
        plt.scatter(true_umap_traj[true_t_idx, 0], true_umap_traj[true_t_idx, 1], label="true", c=BlueRed_12.mpl_colors[0], s=40, alpha=0.8)
        # Plot predictions
        plt.subplot(1, 2, 2)
        plt.title("Predictions", fontsize=15)
        _plotAll()
        plt.scatter(pred_umap_traj[pred_t_idx, 0], pred_umap_traj[pred_t_idx, 1], label="pred", c=BlueRed_12.mpl_colors[-1], s=40, alpha=0.8)
        # plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        plt.tight_layout()
        plt.show()


def plotUMAPTimePoint(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps):
    unique_tps = np.unique(true_cell_tps).astype(int).tolist()
    n_tps = len(unique_tps)
    color_list = linearSegmentCMap(n_tps, "viridis")
    # color_list = Vivid_10.mpl_colors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("True Data", fontsize=15)
    ax2.set_title("Predictions", fontsize=15)
    # for t in range(n_tps):
    for i, t in enumerate(unique_tps):
        true_t_idx = np.where(true_cell_tps == t)[0]
        pred_t_idx = np.where(pred_cell_tps == t)[0]
        ax1.scatter(true_umap_traj[true_t_idx, 0], true_umap_traj[true_t_idx, 1], label=t, color=color_list[i], s=20, alpha=1.0)
        ax2.scatter(pred_umap_traj[pred_t_idx, 0], pred_umap_traj[pred_t_idx, 1], label=t, color=color_list[i], s=20, alpha=1.0)
    ax2.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    # plt.tight_layout()
    plt.show()


def plotUMAPTestTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps, test_tps):
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


def compareUMAPTestTime(
        true_umap_traj, vae_pred_umap_traj, prescient_pred_umap_traj,
        true_cell_tps, vae_pred_cell_tps, prescient_pred_cell_tps, test_tps):
    n_tps = len(np.unique(true_cell_tps))
    color_list = linearSegmentCMap(n_tps, "viridis")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 5))
    ax1.set_title("True Data", fontsize=15)
    ax2.set_title("PRESCIENT Predictions", fontsize=15)
    ax3.set_title("Our Predictions", fontsize=15)
    ax1.scatter(true_umap_traj[:, 0], true_umap_traj[:, 1], label="other", c=gray_color, s=40, alpha=0.5)
    ax2.scatter(true_umap_traj[:, 0], true_umap_traj[:, 1], label="other", c=gray_color, s=40, alpha=0.5)
    ax3.scatter(true_umap_traj[:, 0], true_umap_traj[:, 1], label="other", c=gray_color, s=40, alpha=0.5)
    for t in test_tps:
        c = color_list[int(t)]
        true_t_idx = np.where(true_cell_tps == t)[0]
        vae_pred_t_idx = np.where(vae_pred_cell_tps == t)[0]
        prescient_pred_t_idx = np.where(prescient_pred_cell_tps == t)[0]
        ax1.scatter(true_umap_traj[true_t_idx, 0], true_umap_traj[true_t_idx, 1], label=int(t), color=c, s=20, alpha=1.0)
        ax2.scatter(prescient_pred_umap_traj[prescient_pred_t_idx, 0], prescient_pred_umap_traj[prescient_pred_t_idx, 1], label=int(t), color=c, s=20, alpha=1.0)
        ax3.scatter(vae_pred_umap_traj[vae_pred_t_idx, 0], vae_pred_umap_traj[vae_pred_t_idx, 1], label=int(t), color=c, s=20, alpha=1.0)
    ax3.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    # plt.tight_layout()
    plt.show()


def compareUMAPTestTime2(
        true_umap_traj, vae_pred_umap_traj, prescient_pred_umap_traj, sde_pred_umap_traj,
        true_cell_tps, vae_pred_cell_tps, prescient_pred_cell_tps, sde_pred_cell_tps, test_tps):
    n_tps = len(np.unique(true_cell_tps))
    color_list = linearSegmentCMap(n_tps, "viridis")
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))
    ax1.set_title("True Data", fontsize=15)
    ax2.set_title("PRESCIENT Predictions", fontsize=15)
    ax3.set_title("ODE Predictions", fontsize=15)
    # ax4.set_title("SDE Predictions", fontsize=15)
    ax4.set_title("Compact Loss Predictions", fontsize=15)

    ax1.scatter(true_umap_traj[:, 0], true_umap_traj[:, 1], label="other", c=gray_color, s=40, alpha=0.5)
    ax2.scatter(true_umap_traj[:, 0], true_umap_traj[:, 1], label="other", c=gray_color, s=40, alpha=0.5)
    ax3.scatter(true_umap_traj[:, 0], true_umap_traj[:, 1], label="other", c=gray_color, s=40, alpha=0.5)
    ax4.scatter(true_umap_traj[:, 0], true_umap_traj[:, 1], label="other", c=gray_color, s=40, alpha=0.5)
    for t in test_tps:
        c = color_list[int(t)]
        true_t_idx = np.where(true_cell_tps == t)[0]
        vae_pred_t_idx = np.where(vae_pred_cell_tps == t)[0]
        sde_pred_t_idx = np.where(sde_pred_cell_tps == t)[0]
        prescient_pred_t_idx = np.where(prescient_pred_cell_tps == t)[0]
        ax1.scatter(true_umap_traj[true_t_idx, 0], true_umap_traj[true_t_idx, 1], label=int(t), color=c, s=20, alpha=1.0)
        ax2.scatter(prescient_pred_umap_traj[prescient_pred_t_idx, 0], prescient_pred_umap_traj[prescient_pred_t_idx, 1], label=int(t), color=c, s=20, alpha=1.0)
        ax3.scatter(vae_pred_umap_traj[vae_pred_t_idx, 0], vae_pred_umap_traj[vae_pred_t_idx, 1], label=int(t), color=c, s=20, alpha=1.0)
        ax4.scatter(sde_pred_umap_traj[sde_pred_t_idx, 0], sde_pred_umap_traj[sde_pred_t_idx, 1], label=int(t), color=c, s=20, alpha=1.0)
    ax4.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    # plt.tight_layout()
    plt.show()


# -----

def plotBasicStats(
        true_cell_avg, true_cell_var, true_gene_avg, true_gene_var,
        pred_cell_avg, pred_cell_var, pred_gene_avg, pred_gene_var
):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    plt.title("Cell Avg.")
    # plt.boxplot([true_cell_avg, pred_cell_avg], positions=[0, 1])
    sbn.stripplot(data=[true_cell_avg, pred_cell_avg], jitter=True)
    plt.xticks([0, 1], ["true", "pred"])
    plt.subplot(1, 4, 2)
    plt.title("Cell Var.")
    # plt.boxplot([true_cell_var, pred_cell_var], positions=[0, 1])
    sbn.stripplot(data=[true_cell_var, pred_cell_var], jitter=True)
    plt.xticks([0, 1], ["true", "pred"])
    plt.subplot(1, 4, 3)
    plt.title("Gene Avg.")
    # plt.boxplot([true_gene_avg, pred_gene_avg], positions=[0, 1])
    sbn.stripplot(data=[true_gene_avg, pred_gene_avg], jitter=True)
    plt.xticks([0, 1], ["true", "pred"])
    plt.subplot(1, 4, 4)
    plt.title("Gene Var.")
    # plt.boxplot([true_gene_var, pred_gene_var], positions=[0, 1])
    sbn.stripplot(data=[true_gene_var, pred_gene_var], jitter=True)
    plt.xticks([0, 1], ["true", "pred"])
    plt.tight_layout()
    plt.show()


# -----

def umapWithPCA(traj_data, n_neighbors, min_dist, pca_pcs):
    pca_model = PCA(n_components=pca_pcs, svd_solver="arpack")
    umap_model = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist)
    umap_traj_data = umap_model.fit_transform(pca_model.fit_transform(traj_data))
    return umap_traj_data, umap_model, pca_model


def umapWithoutPCA(traj_data, n_neighbors, min_dist):
    umap_model = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist)
    umap_traj_data = umap_model.fit_transform(traj_data)
    return umap_traj_data, umap_model


def onlyPCA(traj_data, pca_pcs):
    pca_model = PCA(n_components=pca_pcs, svd_solver="arpack")
    pca_traj_data = pca_model.fit_transform(traj_data)
    return pca_traj_data, pca_model