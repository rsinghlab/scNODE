'''
Description:
    Figure plotting for extrapolating multiple timepoints.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import numpy as np
from plotting import *
from plotting.__init__ import _removeTopRightBorders, _removeAllBorders


# ======================================================

model_colors_dict = {
    "scNODE": "#ba1c30",
    "PRESCIENT": Kelly20[0],
    "MIOFlow": "#92ae31",
}

data_name_dict = {
    "zebrafish": "ZB",
    "drosophila": "DR",
    "mammalian": "MB",
    "wot": "SC",
}




def compareForecastingMetric(data_metric_list, data_list):
    data_wass_list = []
    for f_num_metric in data_metric_list:
        model_wass_dist = []
        for m in model_list:
            m_wass = [[f_num_metric[i][j][m]["global"]["ot"] for j in f_num_metric[i]] for i in f_num_metric]
            m_avg_wass = [np.mean(x) for x in m_wass]
            model_wass_dist.append(m_avg_wass)
        data_wass_list.append(model_wass_dist)
    # -----
    n_tps = len(f_num_list)
    fig, ax_list = plt.subplots(1, 3, figsize=(6, 2))
    for d_idx, d_name in enumerate(data_list):
        x_ticks = np.arange(n_tps)
        x_tick_labels = [int(n) + 1 for n in np.arange(n_tps)]
        ax_list[d_idx].set_title(data_name_dict[d_name])
        for m_idx, m in enumerate(model_list):
            ax_list[d_idx].plot(x_ticks, data_wass_list[d_idx][m_idx], "o-", lw=1.5, color=model_colors_dict[m])
        ax_list[d_idx].set_xticks(x_ticks)
        ax_list[d_idx].set_xticklabels(x_tick_labels, fontsize=13)
        ax_list[d_idx].tick_params(axis='y', labelsize=11)
    ax_list[1].set_xlabel("# of extrapolating timepoints", fontsize=12)
    ax_list[0].set_ylabel("Avg. Wass. Dist.", fontsize=12)
    ax_list[1].set_ylim(500, 800)
    ax_list[2].set_ylim(90, 240)
    for ax in ax_list:
        _removeTopRightBorders(ax)
    plt.tight_layout()
    plt.show()


# ======================================================

def mainCompareMetric():
    data_metric_list = []
    for data_name in data_list:
        metric_filename = "../res/extrapolation/{}-diff_forecasting{}-model_metrics.npy".format(data_name, len(f_num_list))
        f_num_metric = np.load(metric_filename, allow_pickle=True).item()
        data_metric_list.append(f_num_metric)
    # -----
    print("Compare metrics...")
    compareForecastingMetric(data_metric_list, data_list) # 5 extrapolating timepoints


# ======================================================

if __name__ == '__main__':
    # Configuration
    print("=" * 70)
    data_list = ["zebrafish", "drosophila", "wot"]
    print("[ {} ]".format(data_list).center(60))
    model_list = ["scNODE", "MIOFlow", "PRESCIENT"]
    f_num_list = [1, 2, 3, 4, 5]
    # -----
    mainCompareMetric()

