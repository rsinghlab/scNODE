'''
Description:
    Run Dummy (Naive) model on the single-cell dataset.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import torch
import numpy as np

from benchmark.BenchmarkUtils import loadSCData, tpSplitInd
from plotting.visualization import plotPredAllTime, plotPredTestTime
from plotting.PlottingUtils import umapWithPCA
from benchmark.BenchmarkUtils import splitBySpec
from plotting.Compare_SingleCell_Predictions import globalEvaluation
from optim.running import constructDummyModel, dummySimulate

# ======================================================
# Load data and pre-processing
print("=" * 70)
data_name= "zebrafish" #zebrafish, mammalian, drosophila, wot, pancreatic, embryoid
print("[ {} ]".format(data_name).center(60))
split_type = "three_interpolation"  # three_interpolation, three_forecasting, one_interpolation, one_forecasting
print("Split type: {}".format(split_type))
ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name, split_type)
train_tps, test_tps = tpSplitInd(data_name, split_type)
data = ann_data.X

# Convert to torch project
traj_data = [torch.FloatTensor(data[np.where(cell_tps == t)[0], :]) for t in range(1, n_tps + 1)]  # (# tps, # cells, # genes)
if cell_types is not None:
    traj_cell_types = [cell_types[np.where(cell_tps == t)[0]] for t in range(1, n_tps + 1)]

all_tps = list(range(n_tps))
train_data, test_data = splitBySpec(traj_data, train_tps, test_tps)
tps = torch.FloatTensor(all_tps)
train_tps = torch.FloatTensor(train_tps)
test_tps = torch.FloatTensor(test_tps)
n_cells = [each.shape[0] for each in traj_data]
print("# tps={}, # genes={}".format(n_tps, n_genes))
print("# cells={}".format(n_cells))
print("Train tps={}".format(train_tps))
print("Test tps={}".format(test_tps))

# ======================================================
# Model running
model_name = "dummy"
dummy_model = constructDummyModel()
all_recon_obs = dummySimulate(dummy_model, traj_data, train_tps, know_all=False)

# ======================================================
# Visualization
print("Compare true and reconstructed data...")
true_data = [each.detach().numpy() for each in traj_data]
true_cell_tps = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(true_data)])
pred_cell_tps = np.concatenate([np.repeat(t, all_recon_obs[t].shape[0]) for t in range(n_tps)])
reorder_pred_data = all_recon_obs

true_umap_traj, umap_model, pca_model = umapWithPCA(np.concatenate(true_data, axis=0), n_neighbors=50, min_dist=0.1, pca_pcs=50)
pred_umap_traj = umap_model.transform(pca_model.transform(np.concatenate(reorder_pred_data, axis=0)))

plotPredAllTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps)
plotPredTestTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps, test_tps.detach().numpy())

# Compute metric for testing time points
print("Compute metrics...")
test_tps_list = [int(t) for t in test_tps]
for t in test_tps_list:
    print("-" * 70)
    print("t = {}".format(t))
    # -----
    pred_global_metric = globalEvaluation(traj_data[t].detach().numpy(), all_recon_obs[t])
    print(pred_global_metric)

# # ======================================================
# # Save results
# save_dir = "../res/single_cell/experimental/{}".format(data_name)
# res_filename = "{}/{}-{}-{}-res.npy".format(save_dir, data_name, split_type, model_name)
# state_filename = "{}/{}-{}-{}-state_dict.pt".format(save_dir, data_name, split_type, model_name)
# print("Saving to {}".format(res_filename))
# np.save(
#     res_filename,
#     {"true": [each.detach().numpy() for each in traj_data],
#      "pred": all_recon_obs,
#      "tps": {"all": tps.detach().numpy(), "train": train_tps.detach().numpy(), "test": test_tps.detach().numpy()},
#      },
#     allow_pickle=True
# )