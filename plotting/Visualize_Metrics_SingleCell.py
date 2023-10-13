'''
Description:
    Compare and visualize evaluation metrics.
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scanpy
import tabulate

from plotting.__init__ import *
from plotting import _removeTopRightBorders


def plotTable(df_data):
    print(tabulate.tabulate(
        df_data, headers=[df_data.index.names[0] + '/' + df_data.columns.names[1]] + list(df_data.columns),
        tablefmt="grid"
    ))



# Load metrics
data_name = "WOT"  # zebrafish, mammalian, WOT, drosophila, Weinreb, embryoid, pancreatic
split_type = "three_forecasting"  # three_interpolation, three_forecasting, one_interpolation, one_forecasting
metric_filename = "../res/comparison/{}-{}-model_metrics.npy".format(data_name, split_type)
stats_filename = "../res/comparison/{}-{}-model_basic_stats.npy".format(data_name, split_type)
metric_dict = np.load(metric_filename, allow_pickle=True).item()
stats_dict = np.load(stats_filename, allow_pickle=True).item()
if split_type == "three_interpolation":
    model_list = ["latent_ODE_OT_pretrain", "PRESCIENT", "WOT", "FNN", "dummy"]
elif split_type == "three_forecasting":
    model_list = ["latent_ODE_OT_pretrain", "PRESCIENT", "FNN", "dummy"]
elif split_type == "one_forecasting":
    model_list = ["latent_ODE_OT_pretrain", "PRESCIENT", "FNN", "dummy"]
elif split_type == "one_interpolation":
    model_list = ["latent_ODE_OT_pretrain", "PRESCIENT", "WOT", "TrajectoryNet", "FNN", "dummy"]
elif split_type == "two_forecasting":
    model_list = ["latent_ODE_OT_pretrain", "PRESCIENT", "FNN", "dummy"]
test_tps = list(metric_dict.keys())
n_test_tps = len(test_tps)
column_names = [("t", t) for t in test_tps]

# Compare pair-wise L2 dist and OT
model_l2 = [[metric_dict[t][m]["global"]["l2"] for t in test_tps ] for m in model_list]
model_ot = [[metric_dict[t][m]["global"]["ot"] for t in test_tps ] for m in model_list]

print("\n" * 2)
metric_df = pd.DataFrame(
    data=model_l2,
    index=pd.MultiIndex.from_tuples([
        ("L2", m) for m in model_list
    ], names=("Metric", "Model")),
    columns=pd.MultiIndex.from_tuples(column_names, names=("Type", "TP"))
)
plotTable(metric_df)


print("\n" * 2)
metric_df = pd.DataFrame(
    data=model_ot,
    index=pd.MultiIndex.from_tuples([
        ("OT", m) for m in model_list
    ], names=("Metric", "Model")),
    columns=pd.MultiIndex.from_tuples(column_names, names=("Type", "TP"))
)
plotTable(metric_df)
