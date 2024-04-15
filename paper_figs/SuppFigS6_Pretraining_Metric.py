'''
Description:
    Figure plotting for using different number of pre-training timepoints.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import numpy as np
import scipy.stats
from plotting import *
from plotting.__init__ import _removeTopRightBorders, _removeAllBorders

# ======================================================

def loadRes(data_name, split_type, trial_id):
    filename = "../res/pretraining/{}-{}-scNODE-diff_pretrain-res-trial{}.npy".format(data_name, split_type, trial_id)
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


def compareMetric(trial_res_dict, n_tps_list):
    trial_list = list(trial_res_dict.keys())
    random_metric_matrix = [[] for _ in range(len(trial_list))]
    first_metric_matrix = [[] for _ in range(len(trial_list))]
    for t_idx, trial in enumerate(trial_list):
        res_dict = trial_res_dict[trial]
        for n in n_tps_list:
            # random metric
            if res_dict[n]["random"] is not None:
                random_metric = res_dict[n]["random"]["pred_metric"]
                random_ot = np.mean([random_metric[t]["ot"] for t in random_metric])
            else:
                random_ot = np.nan
            random_metric_matrix[t_idx].append(random_ot)
            # first metric
            if res_dict[n]["first"] is not None:
                first_metric = res_dict[n]["first"]["pred_metric"]
                first_ot = np.mean([first_metric[t]["ot"] for t in first_metric])
            else:
                first_ot = np.nan
            first_metric_matrix[t_idx].append(first_ot)
    random_metric_matrix_np = np.asarray(random_metric_matrix)
    first_metric_matrix_np = np.asarray(first_metric_matrix)
    min_val = np.nanmin(np.vstack([first_metric_matrix_np, random_metric_matrix_np]))
    max_val = np.nanmax(np.vstack([first_metric_matrix_np, random_metric_matrix_np]))
    random_metric_matrix = [random_metric_matrix_np[~np.isnan(random_metric_matrix_np[:, t]), t] for t in range(random_metric_matrix_np.shape[1])]
    first_metric_matrix = [first_metric_matrix_np[~np.isnan(first_metric_matrix_np[:, t]), t] for t in range(first_metric_matrix_np.shape[1])]
    # -----
    # p-values
    p_value_list = []
    for t_idx in range(len(n_tps_list)):
        random_tmp = random_metric_matrix_np[:, t_idx]
        first_tmp = first_metric_matrix_np[:, t_idx]
        val_idx = np.intersect1d(np.where(~np.isnan(random_tmp)), np.where(~np.isnan(first_tmp)))
        random_tmp = random_tmp[val_idx]
        first_tmp = first_tmp[val_idx]
        p_res = scipy.stats.ttest_ind(random_tmp, first_tmp)
        p_val = p_res.pvalue
        p_value_list.append(p_val)
    # -----
    fig = plt.figure(figsize=(8.5, 4))
    offset = 0.15
    width = 0.25
    bplt1 = plt.boxplot(x = random_metric_matrix, positions=np.arange(len(n_tps_list)) - offset, patch_artist=True, widths=width)
    bplt2 = plt.boxplot(x = first_metric_matrix, positions=np.arange(len(n_tps_list)) + offset, patch_artist=True, widths=width)
    colors1 = [Bold_10.mpl_colors[0] for _ in range(len(n_tps_list))]
    colors2 = [Bold_10.mpl_colors[1] for _ in range(len(n_tps_list))]
    _fillColor(bplt1, colors1)
    _fillColor(bplt2, colors2)
    plt.bar(-1, 0.0, color=Bold_10.mpl_colors[0], label="random")
    plt.bar(-1, 0.0, color=Bold_10.mpl_colors[1], label="first")
    plt.xlim(-0.5, len(n_tps_list) - 0.5)
    plt.ylim(min_val-10, max_val+20)
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
            remove_recovery_MIOFlow_res = [85.36, 87.47, 114.16, 142.03, 150.53, 161.59, 147.23, 155.06] # PRESCIENT metrics
        MIOFlow_avg = np.mean(remove_recovery_MIOFlow_res)
        plt.plot(
            np.arange(-1.0, len(n_tps_list)+1), np.repeat(MIOFlow_avg, len(n_tps_list)+2),
            "--", lw=4, color=Bold_10.mpl_colors[2], label="best baseline"
        )
    # add pvalue annotation
    line_offset = 1.0
    text_offset = 0.1
    x_ticks_list = np.arange(len(n_tps_list))
    for t_idx in range(len(n_tps_list)):
        p_val = p_value_list[t_idx]
        if p_val > 0.05:
            continue
        else:
            p_asterisk = _pValAsterisk(p_val)
            left_x = x_ticks_list[t_idx] - offset
            right_x = x_ticks_list[t_idx] + offset
            left_y = np.max(random_metric_matrix[t_idx]) + 2
            right_y = np.max(first_metric_matrix[t_idx]) + 2
            plt.plot([left_x, left_x], [max_val, max_val+line_offset], lw=2.5, color=dark_gray_color)  # left vline
            plt.plot([right_x, right_x], [max_val, max_val+line_offset], lw=2.5, color=dark_gray_color)  # right vline
            plt.plot([left_x, right_x], [max_val+line_offset, max_val+line_offset], lw=2, color=dark_gray_color)  # top hline
            plt.text(
                x=x_ticks_list[t_idx], y=max_val+line_offset+text_offset,
                s=p_asterisk, fontsize=20, fontweight="bold", color=dark_gray_color, horizontalalignment='center'
            )
    plt.legend(frameon=False, ncol=3, loc="upper left")
    _removeTopRightBorders()
    plt.tight_layout()
    plt.show()

# ======================================================

def mainPretrainingMetric():
    print("Loading prediction results...")
    trial_res_dict = {}
    for trial_id in [0, 1, 2, 3, 4]:
        res_dict = loadRes(data_name, split_type, trial_id)
        trial_res_dict[trial_id] = res_dict
    n_tps_list = list(res_dict.keys())
    strategy_list = ["random", "first"]
    print("Compare metrics...")
    compareMetric(trial_res_dict, n_tps_list)


# ======================================================

if __name__ == '__main__':
    # Configuration
    print("=" * 70)
    data_name = "wot"  # zebrafish, drosophila, wot
    print("[ {} ]".format(data_name).center(60))
    split_type = "remove_recovery"
    print("Split type: {}".format(split_type))
    # -----
    mainPretrainingMetric()

