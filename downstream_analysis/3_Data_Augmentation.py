'''
Description:
    Augmenting data to improve age prediction.
'''
import copy

import matplotlib.pyplot as plt
import scanpy
import scipy.interpolate
import torch
import torch.distributions as dist
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import itertools
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import seaborn as sbn
from scipy.optimize import minimize

from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, tunedOurPars
from plotting.visualization import plotUMAP, plotPredAllTime, plotPredTestTime, umapWithoutPCA, umapWithPCA
from data.preprocessing import splitBySpec
from benchmark.Compare_SingleCell_Predictions import basicStats, globalEvaluation
from optim.running import constructscNODEModel, scNODETrainWithPreTrain, scNODEPredict
from plotting import linearSegmentCMap, _removeAllBorders
from plotting.__init__ import *
import matplotlib.patheffects as pe

# ======================================================
from sklearn.model_selection import train_test_split

def downsample(data, ratio):
    n = data.shape[0]
    sample_d = data[np.random.choice(np.arange(n), int(n*ratio), replace=False), :]
    return sample_d


def upsample(data, n_cells):
    n = data.shape[0]
    sample_d = data[np.random.choice(np.arange(n), n_cells), :]
    return np.concatenate([data, sample_d], axis=0)


def splitData(traj_X, traj_Y, test_ratio):
    train_X = []
    test_X = []
    train_Y = []
    test_Y = []
    for i in range(len(traj_X)):
        x_train, x_test, y_train, y_test = train_test_split(traj_X[i], traj_Y[i], test_size=test_ratio)
        train_X.append(x_train)
        test_X.append(x_test)
        train_Y.append(y_train)
        test_Y.append(y_test)
    return train_X, test_X, train_Y, test_Y

# ======================================================

latent_dim = 50
drift_latent_size = [50]
enc_latent_list = [50]
dec_latent_list = [50]
act_name = "relu"

def augmentation(train_data, train_tps, tps, n_sim_cells):
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
    all_recon_obs = scNODEPredict(latent_ode_model, first_latent_dist, tps, n_cells=n_sim_cells)  # (# trajs, # tps, # genes)
    return latent_ode_model, all_recon_obs


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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
def trainClassifier(X, Y):
    clf_model = RandomForestClassifier(n_estimators=100)
    clf_model.fit(X, Y)
    score = clf_model.score(X, Y)
    print("Train accuracy = {}".format(score))
    return clf_model

# ======================================================

def _train4DownsampleData(train_traj, test_X, test_Y, pos_label, downsample_t):
    print("Train classifier with down-sampled data...")
    train_X = np.concatenate(train_traj, axis=0)
    train_Y = np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(train_traj)])
    # train_Y = np.asarray([str(y) if y == downsample_t else "other" for y in train_Y])
    train_Y = np.asarray(train_Y, dtype=str)
    # ----
    # pca_model = PCA(n_components=50, svd_solver="arpack")
    # pca_model = Normalizer()
    # train_X = pca_model.fit_transform(train_X)
    clf_model = trainClassifier(train_X, train_Y)
    test_pred_Y = clf_model.predict(np.concatenate(test_X, axis=0))
    # test_pred_Y = clf_model.predict(pca_model.transform(np.concatenate(test_X, axis=0)))
    test_true_Y = test_Y
    test_acc = accuracy_score(test_true_Y, test_pred_Y)
    test_f1 = f1_score(test_true_Y, test_pred_Y, pos_label=pos_label)
    print("Test acc={}".format(test_acc))
    print("Test F1={}".format(test_f1))
    conf_mat = confusion_matrix(test_true_Y, test_pred_Y)
    conf_mat = conf_mat / np.sum(conf_mat, axis=1)[:, np.newaxis]
    return clf_model, test_pred_Y, test_acc, test_f1, conf_mat


def _train4UpsampleData(train_traj, test_X, test_Y, pos_label, downsample_t, n_t_cells):
    print("Train classifier with up-sampled data...")
    up_train_traj = copy.deepcopy(train_traj)
    up_sample = upsample(train_traj[downsample_t], n_t_cells-train_traj[downsample_t].shape[0])
    up_train_traj[downsample_t] = up_sample
    upsample_n_cells = [each.shape[0] for each in up_train_traj]
    print("# up-sampled cells={}".format(upsample_n_cells))
    # -----
    train_X = np.concatenate(up_train_traj, axis=0)
    train_Y = np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(up_train_traj)])
    # train_Y = np.asarray([str(y) if y == downsample_t else "other" for y in train_Y])
    train_Y = np.asarray(train_Y, dtype=str)
    # -----
    # pca_model = PCA(n_components=50, svd_solver="arpack")
    # pca_model = Normalizer()
    # train_X = pca_model.fit_transform(train_X)
    clf_model = trainClassifier(train_X, train_Y)
    test_pred_Y = clf_model.predict(np.concatenate(test_X, axis=0))
    # test_pred_Y = clf_model.predict(pca_model.transform(np.concatenate(test_X, axis=0)))
    test_true_Y = test_Y
    test_acc = accuracy_score(test_true_Y, test_pred_Y)
    test_f1 = f1_score(test_true_Y, test_pred_Y, pos_label=pos_label)
    print("Test acc={}".format(test_acc))
    print("Test F1={}".format(test_f1))
    conf_mat = confusion_matrix(test_true_Y, test_pred_Y)
    conf_mat = conf_mat / np.sum(conf_mat, axis=1)[:, np.newaxis]
    return clf_model, test_pred_Y, test_acc, test_f1, conf_mat


def _train4AugmentData(full_traj, train_traj, test_X, test_Y, pos_label, downsample_idx, downsample_t, n_t_cells):
    print("Train our dynamic model...")
    latent_ode_model, all_recon_obs = augmentation(
        train_data=[torch.FloatTensor(x) for x in full_traj],
        train_tps=torch.FloatTensor(np.arange(len(full_traj))),
        tps=torch.FloatTensor(np.arange(len(full_traj))),
        n_sim_cells=n_t_cells-train_traj[downsample_idx].shape[0],
    )
    augment_data = all_recon_obs[:, downsample_t, :]
    print("Train classifier with augmented data...")
    aug_train_traj = copy.deepcopy(train_traj)
    augment_data = np.concatenate([train_traj[downsample_idx], augment_data], axis=0)
    augment_data = augment_data[np.random.choice(np.arange(augment_data.shape[0]), augment_data.shape[0], replace=False), :]
    aug_train_traj[downsample_idx] = augment_data
    aug_n_cells = [each.shape[0] for each in aug_train_traj]
    print("# augmented cells={}".format(aug_n_cells))
    # -----
    train_X = np.concatenate(aug_train_traj, axis=0)
    train_Y = np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(aug_train_traj)])
    train_Y = np.asarray(train_Y, dtype=str)
    # -----
    # pca_model = PCA(n_components=50, svd_solver="arpack")
    # pca_model = Normalizer()
    # train_X = pca_model.fit_transform(train_X)
    clf_model = trainClassifier(train_X, train_Y)
    test_pred_Y = clf_model.predict(np.concatenate(test_X, axis=0))
    # test_pred_Y = clf_model.predict(pca_model.transform(np.concatenate(test_X, axis=0)))
    test_true_Y = test_Y
    test_acc = accuracy_score(test_true_Y, test_pred_Y)
    test_f1 = f1_score(test_true_Y, test_pred_Y, pos_label=pos_label)
    print("Test acc={}".format(test_acc))
    print("Test F1={}".format(test_f1))
    conf_mat = confusion_matrix(test_true_Y, test_pred_Y)
    conf_mat = conf_mat / np.sum(conf_mat, axis=1)[:, np.newaxis]
    return clf_model, test_pred_Y, test_acc, test_f1, conf_mat


def _train4InterpolateData(full_traj, train_traj, test_X, test_Y, pos_label, downsample_idx, downsample_t, n_t_cells, use_tps):
    print("Train our dynamic model...")
    tps = [use_tps[0], use_tps[0]+0.25, use_tps[0]+0.5, use_tps[0]+0.75, use_tps[1]]
    latent_ode_model, all_recon_obs = augmentation(
        train_data=[torch.FloatTensor(x) for x in full_traj],
        train_tps=torch.FloatTensor(np.arange(len(full_traj))),
        tps=torch.FloatTensor(tps),
        n_sim_cells=n_t_cells,
    )
    interpolate_data = [all_recon_obs[:, t, :] for t in range(len(tps))]
    interpolate_data[0] = train_traj[0]
    interpolate_data[-1] = train_traj[-1]
    print("Train classifier with interpolated data...")
    inter_n_cells = [each.shape[0] for each in interpolate_data]
    print("# interpolated cells={}".format(inter_n_cells))
    # -----
    y_list = ["0", "0.25", "0.5", "0.75", "1"]
    train_X = np.concatenate(interpolate_data, axis=0)
    train_Y = np.concatenate([np.repeat(y_list[t], x.shape[0]) for t, x in enumerate(interpolate_data)])
    train_Y = np.asarray(train_Y, dtype=str)
    # -----
    # pca_model = PCA(n_components=50, svd_solver="arpack")
    pca_model = Normalizer()
    train_X = pca_model.fit_transform(train_X)
    clf_model = trainClassifier(train_X, train_Y)
    # test_pred_Y = clf_model.predict(np.concatenate(test_X, axis=0))
    test_pred_Y = clf_model.predict(pca_model.transform(np.concatenate(test_X, axis=0)))
    test_true_Y = test_Y
    test_acc = accuracy_score(test_true_Y, test_pred_Y)
    test_f1 = f1_score(test_true_Y, test_pred_Y, pos_label=pos_label, average=None)
    print("Test acc={}".format(test_acc))
    print("Test F1={}".format(test_f1))
    conf_mat = confusion_matrix(test_true_Y, test_pred_Y)
    conf_mat = conf_mat / np.sum(conf_mat, axis=1)[:, np.newaxis]
    return clf_model, test_pred_Y, test_acc, test_f1, conf_mat


def twoTimepointClassification(traj_data, test_ratio, downsample_ratio, use_tps, downsample_t):
    used_traj_data = [traj_data[i] for i in use_tps]
    # split into train/test sets for selected timepoints
    print("Split train/test sets...")
    used_traj_tp = [np.repeat(t, x.shape[0]) for t, x in enumerate(used_traj_data)]
    train_X, test_X, train_Y, test_Y = splitData(used_traj_data, used_traj_tp, test_ratio=test_ratio)
    train_n_cells = [each.shape[0] for each in train_X]
    print("# train cells={}".format(train_n_cells))
    test_n_cells = [each.shape[0] for each in test_X]
    print("# test cells={}".format(test_n_cells))
    # =======================
    # Down-sampling
    downsample_idx = use_tps.index(downsample_t)
    print("Down-sampling t={}...".format(use_tps[downsample_idx]))
    sample_data = downsample(train_X[downsample_idx], ratio=downsample_ratio)
    downsample_traj_data = copy.deepcopy(train_X)
    downsample_traj_data[downsample_idx] = sample_data
    downsample_n_cells = [each.shape[0] for each in downsample_traj_data]
    print("# sampled cells={}".format(downsample_n_cells))
    # Construct labels for binary classification
    pos_label = "0"
    test_true_Y = np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(test_X)])
    test_true_Y = np.asarray(test_true_Y, dtype=str)
    # =======================
    print("-" * 50)
    print("Down-sample...")
    down_clf_model, down_test_pred_Y, down_test_acc, down_test_f1, down_conf_mat = _train4DownsampleData(
        downsample_traj_data, test_X, test_true_Y, pos_label, downsample_idx
    )
    print("-" * 50)
    print("Up-sample...")
    up_clf_model, up_test_pred_Y, up_test_acc, up_test_f1, up_conf_mat = _train4UpsampleData(
        downsample_traj_data, test_X, test_true_Y, pos_label, downsample_idx,
        n_t_cells=5000
    )
    print("-" * 50)
    print("Augment...")
    full_traj_data = copy.deepcopy(traj_data)
    for t_idx, t in enumerate(use_tps):
        full_traj_data[t] = downsample_traj_data[t_idx]
    aug_clf_model, aug_test_pred_Y, aug_test_acc, aug_test_f1, aug_conf_mat = _train4AugmentData(
        full_traj_data, downsample_traj_data, test_X, test_true_Y, pos_label, downsample_idx, downsample_t,
        n_t_cells=5000
    )
    print("-" * 50)
    print("Interpolate...")
    full_traj_data = copy.deepcopy(traj_data)
    for t_idx, t in enumerate(use_tps):
        full_traj_data[t] = downsample_traj_data[t_idx]
    inter_clf_model, inter_test_pred_Y, inter_test_acc, inter_test_f1, inter_conf_mat = _train4InterpolateData(
        full_traj_data, downsample_traj_data, test_X, test_true_Y, pos_label, downsample_idx, downsample_t,
        n_t_cells=1000, use_tps=use_tps
    )
    # =======================
    print("Visualization...")
    tick_labels = [str(x) for x in use_tps]
    fig, ax_list = plt.subplots(1, 4, figsize=(14, 5))
    ax_list[0].set_title("Down-sample \n Acc={:.2f}, F1={:.2f}".format(down_test_acc, down_test_f1))
    ax_list[1].set_title("Up-sample \n Acc={:.2f}, F1={:.2f}".format(up_test_acc, up_test_f1))
    ax_list[2].set_title("Augmentation \n Acc={:.2f}, F1={:.2f}".format(aug_test_acc, aug_test_f1))
    ax_list[3].set_title("Interpolation \n Acc={:.2f}, F1={}".format(inter_test_acc, inter_test_f1))
    sbn.heatmap(
        down_conf_mat, square=True, cmap="binary", fmt=".2f", annot=True, linewidths=0.1, annot_kws={"size":18},
        xticklabels=tick_labels, yticklabels=tick_labels, cbar=False, ax=ax_list[0]
    )
    sbn.heatmap(
        up_conf_mat, square=True, cmap="binary", fmt=".2f", annot=True, linewidths=0.1, annot_kws={"size": 18},
        xticklabels=tick_labels, yticklabels=tick_labels, cbar=False, ax=ax_list[1]
    )
    sbn.heatmap(
        aug_conf_mat, square=True, cmap="binary", fmt=".2f", annot=True, linewidths=0.1, annot_kws={"size": 18},
        xticklabels=tick_labels, yticklabels=tick_labels, cbar=False, ax=ax_list[2]
    )
    sbn.heatmap(
        inter_conf_mat, square=True, cmap="binary", fmt=".2f", annot=True, linewidths=0.1, annot_kws={"size": 18},
        yticklabels=tick_labels, cbar=False, ax=ax_list[3]
    )
    for a in ax_list:
        a.set_xlabel("Prediction")
        a.set_ylabel("Truth")
    plt.tight_layout()
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
    # Convert to torch project
    traj_data = [data[np.where(cell_tps == t)[0], :] for t in range(1, n_tps + 1)]
    if cell_types is not None:
        traj_cell_types = [cell_types[np.where(cell_tps == t)[0]] for t in range(1, n_tps + 1)]
    else:
        traj_cell_types = None
    all_tps = list(range(n_tps))
    tps = torch.FloatTensor(all_tps)
    n_cells = [each.shape[0] for each in traj_data]
    print("# tps={}, # genes={}".format(n_tps, n_genes))
    print("# cells={}".format(n_cells))
    # =======================
    twoTimepointClassification(traj_data, test_ratio=0.2, downsample_ratio=1.0, use_tps=[10, 11], downsample_t=10)
    # twoTimepointClassification(traj_data, test_ratio=0.2, downsample_ratio=1.0, use_tps=[5, 6], downsample_t=5)




