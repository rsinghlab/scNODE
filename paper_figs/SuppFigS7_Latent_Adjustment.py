'''
Description:
    Figure plotting for including/excluding the latent space update.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import numpy as np
import scipy.stats
import pickle as pkl
from plotting import *
from plotting.__init__ import _removeTopRightBorders, _removeAllBorders

# ======================================================

def loadRes(data_name, split_type, trial_id):
    filename = "../res/pretraining/{}-{}-scNODE-diff_pretrain-res-trial{}.npy".format(data_name, split_type, trial_id)
    res_dict = np.load(filename, allow_pickle=True).item()
    return res_dict


def loadResWOAdjust(data_name, split_type, trial_id):
    filename = "../res/pretraining/res_fixed_latent/{}-{}-scNODE-diff_pretrain-res-trial{}.npy".format(data_name, split_type, trial_id)
    res_dict = np.load(filename, allow_pickle=True).item()
    return res_dict


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


def compareMetric4Adjustment(w_adjust_res_dict, wo_adjust_res_dict, n_tps_list):
    trial_list = list(w_adjust_res_dict.keys())
    w_metric_matrix = [[] for _ in range(len(trial_list))]
    wo_metric_matrix = [[] for _ in range(len(trial_list))]
    for t_idx, trial in enumerate(trial_list):
        w_res_dict = w_adjust_res_dict[trial]
        wo_res_dict = wo_adjust_res_dict[trial]
        for n in n_tps_list:
            # w/ adjustment
            if w_res_dict[n]["first"] is not None:
                w_metric = w_res_dict[n]["first"]["pred_metric"]
                w_ot = np.mean([w_metric[t]["ot"] for t in w_metric])
            else:
                w_ot = np.nan
            w_metric_matrix[t_idx].append(w_ot)
            # w/ adjustment
            if wo_res_dict[n]["first"] is not None:
                wo_metric = wo_res_dict[n]["first"]["pred_metric"]
                wo_ot = np.mean([wo_metric[t]["ot"] for t in wo_metric])
            else:
                wo_ot = np.nan
            wo_metric_matrix[t_idx].append(wo_ot)
    w_metric_matrix_np = np.asarray(w_metric_matrix)
    wo_metric_matrix_np = np.asarray(wo_metric_matrix)
    min_val = np.nanmin(np.vstack([w_metric_matrix_np, wo_metric_matrix_np]))
    max_val = np.nanmax(np.vstack([w_metric_matrix_np, wo_metric_matrix_np]))
    w_metric_matrix = [w_metric_matrix_np[~np.isnan(w_metric_matrix_np[:, t]), t] for t in range(w_metric_matrix_np.shape[1])]
    wo_metric_matrix = [wo_metric_matrix_np[~np.isnan(wo_metric_matrix_np[:, t]), t] for t in range(wo_metric_matrix_np.shape[1])]
    # -----
    # p-values
    p_value_list = []
    for t_idx in range(len(n_tps_list)):
        w_tmp = w_metric_matrix_np[:, t_idx]
        wo_tmp = wo_metric_matrix_np[:, t_idx]
        val_idx = np.intersect1d(np.where(~np.isnan(w_tmp)), np.where(~np.isnan(wo_tmp)))
        w_tmp = w_tmp[val_idx]
        wo_tmp = wo_tmp[val_idx]
        p_res = scipy.stats.ttest_ind(w_tmp, wo_tmp)
        p_val = p_res.pvalue
        p_value_list.append(p_val)
    # -----
    fig = plt.figure(figsize=(8.5, 4))
    width = 0.3
    plt.bar(
        x=np.arange(len(n_tps_list)) - width,
        height=[np.mean(x) for x in wo_metric_matrix],
        yerr=[np.std(x) for x in wo_metric_matrix],
        error_kw = {"elinewidth": 3, "capsize": 5, "ecolor": "k"},
        width=width,
        align="edge",
        color=Bold_10.mpl_colors[0],
        label="exclude",
        edgecolor="k",
        linewidth=0.2,
    )
    plt.bar(
        x=np.arange(len(n_tps_list)),
        height=[np.mean(x) for x in w_metric_matrix],
        yerr=[np.std(x) for x in w_metric_matrix],
        error_kw={"elinewidth": 3, "capsize": 5, "ecolor": "k"},
        width=width,
        align="edge",
        color=Bold_10.mpl_colors[1],
        label="include",
        edgecolor="k",
        linewidth=0.2,
    )
    plt.xlim(-0.5, len(n_tps_list) - 0.5)
    plt.ylim(min_val - 10, max_val + 10)
    plt.xticks(np.arange(len(n_tps_list)), n_tps_list)
    plt.xlabel("# of pre-training tps", fontsize=18)
    plt.ylabel("Wasserstein Distance", fontsize=18)
    # baseline results for remove_recovery task
    if split_type == "remove_recovery" or split_type == "new_remove_recovery":
        if data_name == "zebrafish":
            remove_recovery_MIOFlow_res = [587.26, 536.66, 453.61, 538.80, 671.95, 757.03]
        elif data_name == "drosophila":
            remove_recovery_MIOFlow_res = [437.24, 472.73, 536.59, 616.57, 633.22, 747.79]
        elif data_name == "wot":
            remove_recovery_MIOFlow_res = [85.36, 87.47, 114.16, 142.03, 150.53, 161.59, 147.23, 155.06]  # PRESCIENT metrics
        MIOFlow_avg = np.mean(remove_recovery_MIOFlow_res)
        plt.plot(
            np.arange(-1.0, len(n_tps_list) + 1), np.repeat(MIOFlow_avg, len(n_tps_list) + 2),
            "--", lw=4, color=Bold_10.mpl_colors[2], label="best baseline"
        )
    plt.legend(frameon=False, ncol=3, loc="upper right")
    _removeTopRightBorders()
    plt.tight_layout()
    plt.show()


# ======================================================

def mainCompareLatentAdjustment():
    print("Loading prediction results...")
    w_adjust_res_dict = {}
    wo_adjust_res_dict = {}
    # for trial_id in [0, 1, 2, 3, 4]:
    trial_list = [0, 1, 2, 3, 4] if data_name not in ["zebrafish"] else [0, 1, 2, 3]
    for trial_id in trial_list:
        res_dict = loadRes(data_name, split_type, trial_id)
        w_adjust_res_dict[trial_id] = res_dict
        res_dict = loadResWOAdjust(data_name, split_type, trial_id)
        wo_adjust_res_dict[trial_id] = res_dict
    n_tps_list = list(res_dict.keys())
    strategy_list = ["random", "first"]
    print("Compare metrics...")
    compareMetric4Adjustment(w_adjust_res_dict, wo_adjust_res_dict, n_tps_list)


# ======================================================

if __name__ == '__main__':
    # Configuration
    print("=" * 70)
    data_name = "wot"  # zebrafish, mammalian, drosophila, wot, pancreatic, embryoid
    print("[ {} ]".format(data_name).center(60))
    split_type = "remove_recovery"
    print("Split type: {}".format(split_type))
    # -----
    mainCompareLatentAdjustment()

