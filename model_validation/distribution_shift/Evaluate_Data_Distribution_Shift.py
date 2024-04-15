'''
Description:
    Compute testing data distribution shift based on l2 distance.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import numpy as np
from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, splitBySpec

# ======================================================
from scipy.spatial.distance import cdist

def _unbalancedDist(true_data, pred_data):
    l2_dist = cdist(true_data, pred_data, metric="euclidean")
    avg_l2 = l2_dist.sum() / np.prod(l2_dist.shape)
    return avg_l2

# ======================================================

print("=" * 70)
data_name = "wot"  # zebrafish, drosophila, wot
split_type = "remove_recovery"
print("[ {} ]".format(data_name).center(60))
print("Split type: {}".format(split_type))
ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name, split_type)
train_tps, test_tps = tpSplitInd(data_name, split_type)
data = ann_data.X

# Convert to torch project
traj_data = [data[np.where(cell_tps == t)[0], :] for t in range(1, n_tps + 1)]  # (# tps, # cells, # genes)

all_tps = list(range(n_tps))
train_data, test_data = splitBySpec(traj_data, train_tps, test_tps)
tps = all_tps
train_tps = train_tps
test_tps = test_tps
n_cells = [each.shape[0] for each in traj_data]
print("# tps={}, # genes={}".format(n_tps, n_genes))
print("# cells={}".format(n_cells))
print("Train tps={}".format(train_tps))
print("Test tps={}".format(test_tps))

# Compute testing data distribution shift (l2 distance)
all_train_data = np.concatenate(train_data, axis=0)
l2_list = []
for i, test_t in enumerate(test_tps):
    print("-" * 50)
    print("Test tp: ", test_t)
    t_data = test_data[i]
    avg_l2 = _unbalancedDist(all_train_data, t_data)
    print("Avg. L2 = {}".format(avg_l2))
    l2_list.append(avg_l2)
print("=" * 50)
print("L2 list = {}".format(l2_list))

np.save("./res/distribution_shift/{}-{}-distribution_shift_metric.npy".format(data_name, split_type), {
    "l2_list": l2_list,
})