'''
Description:
    Pre-processing of Waddington-OT/Schiebinger2019 (SC) data.
'''
import scanpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Change file directory to the raw SC data
days_df = pd.read_csv("./raw/cell_days.txt", index_col='id', sep='\t')
adata = scanpy.read_h5ad("./raw/ExprMatrix.h5ad") # cell x gene
adata.obs = adata.obs.join(days_df)

adata = adata[~pd.isna(adata.obs['day']), :]
meta_df = adata.obs
unique_days = adata.obs['day'].unique()
unique_days = unique_days[np.isnan(unique_days) == False]
print("Data shape: ", adata.shape)
print("Num of unique days = {}".format(len(unique_days)))
# -----
print("-" * 70)
print("Merge timepoints...")
cell_tps = np.floor(adata.obs['day'])
adata.obs['day'] = cell_tps
unique_days = adata.obs['day'].unique()
print("Data shape: ", adata.shape)
print("Num of unique days (merged) = {}".format(len(unique_days)))
print("Num of cell:")
cell_idx_per_tp = [np.where(adata.obs["day"] == t)[0] for t in unique_days]
cell_num_per_tp = [len(x) for x in cell_idx_per_tp]
print(cell_num_per_tp)
# -----
print("-" * 70)
ratio = 0.1
print("Subsampling (ratio={})...".format(ratio))
sample_cell_idx_per_tp = [np.random.choice(x, int(len(x)*ratio), replace=False) for x in cell_idx_per_tp]
adata = adata[np.concatenate(sample_cell_idx_per_tp), :]
unique_days = adata.obs['day'].unique()
print("Data shape: ", adata.shape)
print("Num of unique days (sampled) = {}".format(len(unique_days)))
cell_idx_per_tp = [np.where(adata.obs["day"] == t)[0] for t in unique_days]
cell_num_per_tp = [len(x) for x in cell_idx_per_tp]
print("Num of cell:")
print(cell_num_per_tp)
# -----
print("-" * 70)
split_type = "three_forecasting" # three_forecasting, three_interpolation, remove_recovery
print("Data shape: ", adata.shape)
print("Num of tps: ", len(unique_days))
print("Split type: {}".format(split_type))
if split_type == "three_forecasting": # medium
    train_tps = unique_days[:16].tolist()
    test_tps = unique_days[16:].tolist()
elif split_type == "three_interpolation": # easy
    train_tps = unique_days.tolist()
    test_tps = [train_tps[5], train_tps[10], train_tps[15]]
    train_tps.remove(unique_days[5])
    train_tps.remove(unique_days[10])
    train_tps.remove(unique_days[15])
elif split_type == "remove_recovery": # hard
    train_tps = unique_days.tolist()
    test_idx = [5, 7, 9, 11, 15, 16, 17, 18]
    test_tps = [train_tps[t] for t in test_idx]
    for t in test_idx:
        train_tps.remove(unique_days[t])
print("Train tps: ", train_tps)
print("Test tps: ", test_tps)
# -----
train_adata = adata[adata.obs['day'].apply(lambda x: x in train_tps)]
print("Train data shape: ", train_adata.shape)
hvgs_summary = scanpy.pp.highly_variable_genes(train_adata, n_top_genes=2000, inplace=False)
hvgs = train_adata.var.index.values[hvgs_summary.highly_variable]
adata = adata[:, hvgs]
meta_df = adata.obs
print("HVG data shape: ", adata.shape)
print("HVG meta shape: ", meta_df.shape)
adata.to_df().to_csv("./reduced_processed/{}-norm_data-hvg.csv".format(split_type)) # cell x genes
pd.DataFrame(hvgs).to_csv("./reduced_processed/{}-var_genes_list.csv".format(split_type))
meta_df.to_csv("./reduced_processed/{}-meta_data.csv".format(split_type))