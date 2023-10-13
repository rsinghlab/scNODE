'''
Description:
    Pre-processing of Schiebinger data.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>

Reference:
    [1] https://broadinstitute.github.io/wot/tutorial/
    [2] https://www.cell.com/cell/fulltext/S0092-8674(19)30039-X
'''
import scanpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

days_df = pd.read_csv("./raw/cell_days.txt", index_col='id', sep='\t')
adata = scanpy.read_h5ad("./raw/ExprMatrix.h5ad") # cell x gene
adata.obs = adata.obs.join(days_df)
meta_df = adata.obs
unique_days = adata.obs['day'].unique()
unique_days = unique_days[np.isnan(unique_days) == False]
print("Data shape: ", adata.shape)
print("Num of tps: ", len(unique_days))
split_type = "three_interpolation" # interpolation, forecasting, three_forecasting, three_interpolation
if split_type == "interpolation":
    train_tps = np.concatenate([unique_days[:17], unique_days[23:35]]).tolist() # 0~8.0 + 10.5~16.0
    test_tps = np.concatenate([unique_days[17:23], unique_days[35:]]).tolist() # 8.25 ~10.0 + 16.5~18.0
elif split_type == "forecasting":
    train_tps = unique_days[:35].tolist()  # 0~16.0
    test_tps = unique_days[35:].tolist()  # 16.5~18.0
elif split_type == "three_forecasting":
    train_tps = unique_days[:36].tolist()  # 0~16.5
    test_tps = unique_days[36:].tolist()  # 17.0~18.0
elif split_type == "three_interpolation":
    train_tps = unique_days.tolist()
    test_tps = [train_tps[15], train_tps[20], train_tps[25]]
    train_tps.remove(test_tps[0])
    train_tps.remove(test_tps[1])
    train_tps.remove(test_tps[2])
print("Train tps: ", train_tps)
print("Test tps: ", test_tps)
# -----
train_adata = adata[adata.obs['day'].apply(lambda x: x in train_tps)]
print("Train data shape: ", train_adata.shape)
hvgs_summary = scanpy.pp.highly_variable_genes(train_adata, n_top_genes=2000, inplace=False)
hvgs = train_adata.var.index.values[hvgs_summary.highly_variable]
adata = adata[:, hvgs]
print("HVG data shape: ", adata.shape)
adata.to_df().to_csv("./processed/{}-norm_data-hvg.csv".format(split_type)) # cell x genes
pd.DataFrame(hvgs).to_csv("./processed/{}-var_genes_list.csv".format(split_type))
meta_df.to_csv("./processed/meta_data.csv")