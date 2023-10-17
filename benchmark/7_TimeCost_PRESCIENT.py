'''
Description:
    Compute time cost for PRESCIENT on the zebrafish (interpolation) data.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
sys.path.append("../baseline/")
sys.path.append("../baseline/prescient_model/")
sys.path.append("../baseline/prescient_model/prescient")
from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, tunedPRESCIENTPars
from baseline.prescient_model.process_data import main as prepare_data
from baseline.prescient_model.running import prescientTrainWithTimer

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

# ======================================================
# Load data
print("=" * 70)
data_name= "zebrafish"
print("[ {} ]".format(data_name).center(60))
split_type = "three_interpolation"
print("Split type: {}".format(split_type))
ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name, split_type)
train_tps, test_tps = tpSplitInd(data_name, split_type)
data = ann_data.X
all_tps = list(np.arange(n_tps))


traj_data = [data[np.where(cell_tps == t)[0], :] for t in range(1, n_tps + 1)]  # (# tps, # cells, # genes)

processed_data = ann_data[
    [t for t in range(ann_data.shape[0]) if (ann_data.obs["tp"].values[t] - 1.0) in train_tps]
]
cell_tps = (processed_data.obs["tp"].values - 1.0).astype(int)
cell_types = np.repeat("NAN", processed_data.shape[0])
genes = processed_data.var.index.values

# Parameter settings
k_dim, layers, sd, tau, clip = tunedPRESCIENTPars(data_name, split_type)

# PCA and construct data dict
data_dict, scaler, pca, um = prepare_data(
    processed_data.X, cell_tps, cell_types, genes,
    num_pcs=k_dim, num_neighbors_umap=10
)
data_dict["y"] = list(range(train_tps[-1] + 1))
data_dict["x"] = [data_dict["x"][train_tps.index(t)] if t in train_tps else [] for t in data_dict["y"]]
data_dict["xp"] = [data_dict["xp"][train_tps.index(t)] if t in train_tps else [] for t in data_dict["y"]]
data_dict["w"] = [data_dict["w"][train_tps.index(t)] if t in train_tps else [] for t in data_dict["y"]]
data_dict["scalar"] = scaler
data_dict["true_x"] = traj_data

# ======================================================
# Model training
train_epochs = 1000
train_lr = 1e-3
pretrain_time, iter_time, iter_metric = prescientTrainWithTimer(
    data_dict, data_name=data_name, out_dir="", train_t=train_tps[1:], timestamp=timestamp,
    k_dim=k_dim, layers=layers, train_epochs=train_epochs, train_lr=train_lr,
    train_sd=sd, train_tau=tau, train_clip=clip, num_steps=(n_tps - 1)*10, num_cells=2000, test_tps=test_tps
)
# -----
time_list = np.cumsum(iter_time) + pretrain_time
time_ot = np.asarray(iter_metric).mean(axis=1)
plt.plot(time_list, time_ot)
plt.xlabel("Iteration")
plt.ylabel("Wasserstein")
plt.tight_layout()
plt.show()

# # -----
# print("Saving results...")
# np.save(
#     "../res/time_cost/{}-{}-PRESCIENT-time_cost.npy".format(data_name, split_type),
#     {
#         "pretrain_time": pretrain_time,
#         "iter_time": iter_time,
#         "iter_metric": iter_metric,
#     },
#     allow_pickle=True
# )
