'''
Description:
    scNODE performance improveements vs. distribution shifts.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''

from plotting.__init__ import _removeTopRightBorders
from plotting.__init__ import *
import numpy as np
from scipy.stats import spearmanr
from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, splitBySpec


split_type = "remove_recovery"

# -----
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import r2_score
lr_model = RANSACRegressor()

data_name_dict = {
    "zebrafish": "ZB",
    "drosophila": "DR",
    "wot": "SC",
}

fig, ax_list = plt.subplots(1, 3, figsize=(9, 3))
for idx, data_name in enumerate(["zebrafish", "drosophila", "wot"]):
    metric_res = np.load("../../res/distribution_shift/{}-{}-distribution_shift_metric.npy".format(data_name, split_type), allow_pickle=True).item()
    l2_list = metric_res["l2_list"]
    l2_list = np.asarray(l2_list)
    # -----
    model_list = ["scNODE", "MIOFlow", "PRESCIENT"]
    wass_filename = "../../res/comparison/{}-{}-model_metrics.npy".format(data_name, split_type) # evaluation metrics for model predictions
    train_tps, test_tps = tpSplitInd(data_name, split_type)
    wass_dict = np.load(wass_filename, allow_pickle=True).item()
    model_ot = [[wass_dict[t][m]["global"]["ot"] for t in test_tps] for m in model_list]
    # -----
    # metrics from more trials
    trial_metric_list = []
    for m in model_list:
        metric_filename = "../../res/distribution_shift/{}-{}-{}-res.npy".format(data_name, split_type, m)
        metric_dict = np.load(metric_filename, allow_pickle=True).item()
        metric = np.asarray(metric_dict["ot_list"])
        if metric.shape[0] == 4:
            metric = metric.tolist()
            metric.append(metric[-1])
            metric = np.asarray(metric)
        trial_metric_list.append(metric)
    trial_metric_list = np.asarray(trial_metric_list)
    # Compute avg Wasserstein
    stack_metric_list = []
    for i in range(3):
        stack_metric_list.append(np.vstack([trial_metric_list[i], np.asarray(model_ot[i]).reshape(1, -1)]))
    model_ot = [np.mean(x, axis=0) for x in stack_metric_list]
    scNODE_res = model_ot[0]
    other_res = np.min(model_ot[1:], axis=0)
    diff = other_res - scNODE_res
    # -----
    # remove outliers with the max residual from RANSAC regression
    lr_model.fit(l2_list.reshape(-1, 1), diff)
    pred_diff = lr_model.predict(l2_list.reshape(-1, 1))
    residual = np.abs(diff - pred_diff)
    max_val = np.max(residual)
    inlier_idx = np.where(residual != max_val)
    # -----
    ax_list[idx].set_title(data_name_dict[data_name], fontsize=18)
    ax_list[idx].scatter(l2_list, diff, c="k", s=50, alpha=0.95)
    rho_fontsize = 15
    if idx == 0:
        ax_list[0].text(53.5, 25.0, s="$\\rho=${:.2f}".format(spearmanr(l2_list[inlier_idx], diff[inlier_idx]).correlation), fontsize=rho_fontsize)
        # ax_list[0].text(53.5, 32.0, s="$\\tau=${:.2f}".format(kendalltau(l2_list, diff)[0]), fontsize=13)
    if idx == 1:
        ax_list[1].text(50.0, -5.0, s="$\\rho=${:.2f}".format(spearmanr(l2_list[inlier_idx], diff[inlier_idx]).correlation), fontsize=rho_fontsize)
        # ax_list[1].text(50.0, -5.0, s="$\\tau=${:.2f}".format(kendalltau(l2_list, diff)[0]), fontsize=13)
    if idx == 2:
        ax_list[2].text(19.0, 7.0, s="$\\rho=${:.2f}".format(spearmanr(l2_list[inlier_idx], diff[inlier_idx]).correlation), fontsize=rho_fontsize)
        # ax_list[2].text(19.0, 7.0, s="$\\tau=${:.2f}".format(kendalltau(l2_list, diff)[0]), fontsize=13)

ax_list[0].set_ylabel("scNODE Improvement", fontsize=18)
ax_list[1].set_xlabel("Distribution Shift ($\\ell_2$)", fontsize=18)

for ax in ax_list:
    _removeTopRightBorders(ax)
plt.tight_layout()
plt.show()


