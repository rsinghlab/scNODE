'''
Description:
    Utility functions for benchmarking.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import numpy as np
import scanpy
import pandas as pd
import natsort
import torch
import torch.distributions as dist
from optim.evaluation import _ot


# --------------------------------
# Load scRNA-seq datasets

def loadZebrafishData(data_dir, split_type):
    cnt_data = pd.read_csv("{}/{}-count_data-hvg.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/meta_data.csv".format(data_dir), header=0, index_col=0)
    meta_data = meta_data.loc[cnt_data.index,:]
    cell_stage = meta_data["stage.nice"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.zeros((len(cell_stage), ))
    cell_tp[cell_tp == 0] = np.nan
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_tp += 1
    meta_data["tp"] = cell_tp
    # -----
    cell_set_meta = pd.read_csv("{}/cell_groups_meta.csv".format(data_dir), header=0, index_col=0)
    meta_data = pd.concat([meta_data, cell_set_meta.loc[meta_data.index, :]], axis=1)
    ann_data = scanpy.AnnData(X=cnt_data, obs=meta_data)
    return ann_data


def loadMammalianData(data_dir, split_type):
    cnt_data = pd.read_csv("{}/{}-norm_data-hvg.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/meta_data.csv".format(data_dir), header=0, index_col=0)
    meta_data = meta_data.set_index("NAME")
    meta_data = meta_data.loc[cnt_data.index,:]
    cell_stage = meta_data["orig_ident"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.zeros((len(cell_stage), ))
    cell_tp[cell_tp == 0] = np.nan
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_tp += 1
    meta_data["tp"] = cell_tp
    ann_data = scanpy.AnnData(X=cnt_data, obs=meta_data)
    return ann_data


def loadDrosophilaData(data_dir, split_type):
    cnt_data = pd.read_csv("{}/{}-count_data-hvg.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/subsample_meta_data.csv".format(data_dir), header=0, index_col=0)
    meta_data = meta_data.loc[cnt_data.index,:]
    cell_stage = meta_data["time"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.zeros((len(cell_stage), ))
    cell_tp[cell_tp == 0] = np.nan
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_tp += 1
    meta_data["tp"] = cell_tp
    ann_data = scanpy.AnnData(X=cnt_data, obs=meta_data)
    return ann_data


def loadWOTData(data_dir, split_type):
    cnt_data = pd.read_csv("{}/{}-norm_data-hvg.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/meta_data.csv".format(data_dir), header=0, index_col=0)
    cell_idx = np.where(~np.isnan(meta_data["day"].values))[0] # remove cells with nan labels
    cnt_data = cnt_data.iloc[cell_idx, :]
    meta_data = meta_data.loc[cnt_data.index,:]
    cell_stage = meta_data["day"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.zeros((len(cell_stage), ))
    cell_tp[cell_tp == 0] = np.nan
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_tp += 1
    meta_data["tp"] = cell_tp
    ann_data = scanpy.AnnData(X=cnt_data, obs=meta_data)
    return ann_data


def loadPancreaticData(data_dir, split_type):
    cnt_data = pd.read_csv("{}/{}-count_data-hvg.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/meta_data.csv".format(data_dir), header=0, index_col=0)
    meta_data = meta_data.loc[cnt_data.index,:]
    cell_stage = meta_data["day"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.zeros((len(cell_stage), ))
    cell_tp[cell_tp == 0] = np.nan
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_tp += 1
    meta_data["tp"] = cell_tp
    ann_data = scanpy.AnnData(X=cnt_data, obs=meta_data)
    return ann_data


def loadEmbryoidData(data_dir, split_type):
    cnt_data = pd.read_csv("{}/{}-count_data-hvg.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/meta_data.csv".format(data_dir), header=0, index_col=0)
    meta_data = meta_data.loc[cnt_data.index,:]
    cell_stage = meta_data["day"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.zeros((len(cell_stage), ))
    cell_tp[cell_tp == 0] = np.nan
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_tp += 1
    meta_data["tp"] = cell_tp
    ann_data = scanpy.AnnData(X=cnt_data, obs=meta_data)
    return ann_data

# --------------------------------
# Dataset directories
#TODO: data directory

# zebrafish_data_dir = "../data/single_cell/experimental/zebrafish_embryonic/new_processed"
zebrafish_data_dir = "../data/single_cell/experimental/zebrafish_embryonic/new_processed"
mammalian_data_dir = "../data/single_cell/experimental/mammalian_cerebral_cortex/new_processed"
wot_data_dir = "../data/single_cell/experimental/Schiebinger2019/processed/"
drosophila_data_dir = "../data/single_cell/experimental/drosophila_embryonic/processed/"
pancreatic_data_dir = "../data/single_cell/experimental/mouse_pancreatic/processed/"
embryoid_data_dir = "../data/single_cell/experimental/embryoid_body/processed/"



def loadSCData(data_name, split_type):
    '''
    Main function to load scRNA-seq dataset and pre-process it.
    '''
    print("[ Data={} | Split={} ] Loading data...".format(data_name, split_type))
    if data_name == "zebrafish":
        ann_data = loadZebrafishData(zebrafish_data_dir, split_type)
        ann_data.X = ann_data.X.astype(float)
        processed_data = preprocess(ann_data.copy())
        cell_types =  processed_data.obs["ZF6S-Cluster"].apply(lambda x: "NAN" if pd.isna(x) else x).values
    elif data_name == "mammalian":
        ann_data = loadMammalianData(mammalian_data_dir, split_type)
        processed_data = ann_data.copy()
        cell_types = processed_data.obs.New_cellType.values
    elif data_name == "drosophila":
        ann_data = loadDrosophilaData(drosophila_data_dir, split_type)
        print("Pre-processing...")
        ann_data.X = ann_data.X.astype(float)
        processed_data = preprocess(ann_data.copy())
        cell_types = processed_data.obs.seurat_clusters.values
    elif data_name == "wot":
        ann_data = loadWOTData(wot_data_dir, split_type)
        processed_data = ann_data.copy()
        cell_types = None
    elif data_name == "pancreatic":
        ann_data = loadPancreaticData(pancreatic_data_dir, split_type)
        ann_data.X = ann_data.X.astype(float)
        processed_data = preprocess(ann_data.copy())
        cell_types = None
    elif data_name == "embryoid":
        ann_data = loadEmbryoidData(embryoid_data_dir, split_type)
        ann_data.X = ann_data.X.astype(float)
        processed_data = preprocess(ann_data.copy())
        cell_types = None
    else:
        raise ValueError("Unknown data name.")
    cell_tps = ann_data.obs["tp"]
    n_tps = len(np.unique(cell_tps))
    n_genes = ann_data.shape[1]
    return processed_data, cell_tps, cell_types, n_genes, n_tps


def tpSplitInd(data_name, split_type):
    '''
    Get the training/testing timepoint split for each dataset.
    '''
    if data_name == "zebrafish":
        if split_type == "interpolation":
            train_tps = [0, 1, 2, 3, 4, 5, 8, 9]
            test_tps = [6, 7, 10, 11]
        elif split_type == "forecasting":
            train_tps = [0, 1, 2, 3, 4, 5]
            test_tps = [6, 7, 8, 9, 10, 11]
        elif split_type == "three_forecasting":
            train_tps = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            test_tps = [9, 10, 11]
        elif split_type == "two_forecasting":
            train_tps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            test_tps = [10, 11]
        elif split_type == "three_interpolation":
            train_tps = [0, 1, 2, 3, 5, 7, 9, 10, 11]
            test_tps = [4, 6, 8]
        else:
            raise ValueError("Unknown split type {}!".format(split_type))
    elif data_name == "mammalian":
        if split_type == "interpolation":
            train_tps = [0, 1, 2, 3, 4, 5, 8, 9]
            test_tps = [6, 7, 10, 11, 12]
        elif split_type == "forecasting":
            train_tps = [0, 1, 2, 3, 4, 5, 6, 7]
            test_tps = [8, 9, 10, 11, 12]
        elif split_type == "three_forecasting":
            train_tps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            test_tps = [10, 11, 12]
        elif split_type == "three_interpolation":
            train_tps = [0, 1, 2, 3, 5, 7, 9, 10, 11, 12]
            test_tps = [4, 6, 8]
        else:
            raise ValueError("Unknown split type {}!".format(split_type))
    elif data_name == "drosophila":
        if split_type == "interpolation":
            train_tps = [0, 1, 2, 3, 4, 7, 8]
            test_tps = [5, 6, 9, 10]
        elif split_type == "forecasting":
            train_tps = [0, 1, 2, 3, 4, 5]
            test_tps = [6, 7, 8, 9, 10]
        elif split_type == "three_forecasting":
            train_tps = [0, 1, 2, 3, 4, 5, 6, 7]
            test_tps = [8, 9, 10]
        elif split_type == "three_interpolation":
            train_tps = [0, 1, 2, 3, 5, 7, 9, 10]
            test_tps = [4, 6, 8]
        else:
            raise ValueError("Unknown split type {}!".format(split_type))
    elif data_name == "wot":
        if split_type == "interpolation":
            train_tps = np.concatenate([np.arange(17), np.arange(23, 35)]).tolist()  # 0~8.0 + 10.5~16.0
            test_tps = np.concatenate([np.arange(17, 23), np.arange(35, 39)]).tolist()  # 8.25 ~10.0 + 16.5~18.0
        elif split_type == "forecasting":
            train_tps = np.arange(35).tolist()  # 0~16.0
            test_tps = np.arange(35, 39).tolist()  # 16.5~18.0
        elif split_type == "three_forecasting":
            train_tps = np.arange(36).tolist()
            test_tps = np.arange(36, 39).tolist()
        elif split_type == "three_interpolation":
            train_tps = np.arange(39).tolist()
            test_tps = [15, 20, 25]
            train_tps.remove(15)
            train_tps.remove(20)
            train_tps.remove(25)
        else:
            raise ValueError("Unknown split type {}!".format(split_type))
    elif data_name == "pancreatic":
        if split_type == "one_interpolation":
            train_tps = [0, 1, 3]
            test_tps = [2]
        elif split_type == "one_forecasting":
            train_tps = [0, 1, 2]
            test_tps = [3]
        else:
            raise ValueError("Unknown split type {}!".format(split_type))
    elif data_name == "embryoid":
        if split_type == "one_interpolation":
            train_tps = [0, 1, 3, 4]
            test_tps = [2]
        elif split_type == "one_forecasting":
            train_tps = [0, 1, 2, 3]
            test_tps = [4]
        else:
            raise ValueError("Unknown split type {}!".format(split_type))
    else:
        raise ValueError("Unknown data name.")
    return train_tps, test_tps


def splitBySpec(traj_data, train_tps, test_tps):
    '''
    Split timepoints into training and testing sets.
    '''
    train_data = [traj_data[t] for t in train_tps]
    test_data = [traj_data[t] for t in test_tps]
    return train_data, test_data

# --------------------------------

def tunedOurPars(data_name, split_type):
    latent_dim = 50
    if data_name == "zebrafish":
        if split_type == "three_interpolation":
            drift_latent_size = [50, 50]
            enc_latent_list = None
            dec_latent_list = None
        elif split_type == "two_forecasting":
            drift_latent_size = [50, 50]
            enc_latent_list = None
            dec_latent_list = None
        else:
            drift_latent_size = [50, 50]
            enc_latent_list = [50, 50]
            dec_latent_list = None
    elif data_name == "mammalian":
        if split_type == "three_interpolation":
            drift_latent_size = [50]
            enc_latent_list = [50, 50]
            dec_latent_list = [50]
        else:
            drift_latent_size = [50, 50]
            enc_latent_list = None
            dec_latent_list = None
    elif data_name == "drosophila":
        if split_type == "three_interpolation":
            drift_latent_size = [50, 50]
            enc_latent_list = [50, 50]
            dec_latent_list = [50, 50]
        else:
            drift_latent_size = [50, 50]
            enc_latent_list = [50, 50]
            dec_latent_list = [50]
    elif data_name == "wot":
        if split_type == "three_interpolation":
            drift_latent_size = [50, 50]
            enc_latent_list = None
            dec_latent_list = [50]
        else:
            drift_latent_size = [50, 50]
            enc_latent_list = None
            dec_latent_list = [50, 50]
    elif data_name == "pancreatic":
        if split_type == "one_interpolation":
            drift_latent_size = None
            enc_latent_list = [50, 50]
            dec_latent_list = None
        else:
            drift_latent_size = None
            enc_latent_list = [50, 50]
            dec_latent_list = None
    elif data_name == "embryoid":
        if split_type == "one_interpolation":
            drift_latent_size = [50, 50]
            enc_latent_list = None
            dec_latent_list = None
        else:
            drift_latent_size = None
            enc_latent_list = None
            dec_latent_list = None
    else:
        raise ValueError("Unknown data name {}!".format(data_name))
    return latent_dim, drift_latent_size, enc_latent_list, dec_latent_list


def tunedPRESCIENTPars(data_name, split_type):
    latent_dim = 50
    if data_name == "zebrafish":
        if split_type == "three_interpolation":
            layers = 2
            sd = 0.9937806814959224
            tau = 0.043436700281791585
            clip = 0.7069653297916607
        elif split_type == "two_forecasting":
            layers = 2
            sd = 0.9991698103589105
            tau = 0.03710053902853564
            clip = 0.8302477813781769
        else:
            layers = 3
            sd = 0.9879286178559656
            tau = 0.01300206586631907
            clip = 0.5667750361711997
    elif data_name == "mammalian":
        if split_type == "three_interpolation":
            layers = 2
            sd = 0.8305569546486499
            tau = 0.004355429690608932
            clip = 0.936995943477638
        else:
            layers = 3
            sd = 0.9947814602754529
            tau = 0.03517891164226188
            clip = 0.3791403957817134
    elif data_name == "drosophila":
        if split_type == "three_interpolation":
            layers = 3
            sd = 0.9316792895856639
            tau = 0.0016038637683827054
            clip = 0.7111819465979426
        else:
            layers = 3
            sd = 0.9985129735647569
            tau = 0.0009236067525837804
            clip = 0.25063399806101017
    elif data_name == "wot":
        if split_type == "three_interpolation":
            layers = 3
            sd = 0.9393257888909708
            tau = 0.09879122021429537
            clip = 0.09042434894597562
        else:
            layers = 3
            sd = 0.9420156236712828
            tau = 0.09734314429256952
            clip = 0.4719485160958774
    elif data_name == "pancreatic":
        if split_type == "one_interpolation":
            layers = 2
            sd = 0.9140103120642563
            tau = 0.00019380198323672187
            clip = 0.2898396915406
        else:
            layers = 3
            sd = 0.9987447492679734
            tau = 0.05128570753263094
            clip = 0.5302753147230872
    elif data_name == "embryoid":
        if split_type == "one_interpolation":
            layers = 2
            sd = 0.9478977587625725
            tau = 0.004796809332184723
            clip = 0.24055551019393706
        else:
            layers = 2
            sd = 0.9941383677084054
            tau = 0.004691460028394059
            clip = 0.005343627588891897
    else:
        raise ValueError("Unknown data name {}!".format(data_name))
    return latent_dim, layers, sd, tau, clip


def tunedWOTPars(data_name, split_type):
    latent_dim = 50
    if data_name == "zebrafish":
        if split_type == "three_interpolation":
            epsilon = 0.05
            lambda1 = 10
            lambda2 = 50
    elif data_name == "mammalian":
        if split_type == "three_interpolation":
            epsilon = 0.1
            lambda1 = 0.1
            lambda2 = 50
    elif data_name == "drosophila":
        if split_type == "three_interpolation":
            epsilon = 0.1
            lambda1 = 1
            lambda2 = 500
    elif data_name == "wot":
        if split_type == "three_interpolation":
            epsilon = 0.05
            lambda1 = 10
            lambda2 = 50
    elif data_name == "pancreatic":
        if split_type == "one_interpolation":
            epsilon = 0.05
            lambda1 = 1
            lambda2 = 500
    elif data_name == "embryoid":
        if split_type == "one_interpolation":
            epsilon = 0.05
            lambda1 = 0.1
            lambda2 = 5
    else:
        raise ValueError("Unknown data name {}!".format(data_name))
    return latent_dim, epsilon, lambda1, lambda2


def tunedTrjectoryNetPars(data_name, split_type):
    latent_dim = 50
    vecint = None
    if data_name == "pancreatic":
        if split_type == "one_interpolation":
            top_k_reg = 0.0
    elif data_name == "embryoid":
        if split_type == "one_interpolation":
            top_k_reg = 1.0
    else:
        raise ValueError("Unknown data name {}!".format(data_name))
    return latent_dim, top_k_reg, vecint

# --------------------------------

def traj2Ann(traj_data):
    # traj_data: #trajs, #tps, # features
    traj_data_list = [traj_data[:, t, :] for t in range(traj_data.shape[1])]
    time_step = np.concatenate([np.repeat(t, traj_data.shape[0]) for t in range(traj_data.shape[1])])
    ann_data = scanpy.AnnData(X=np.concatenate(traj_data_list, axis=0))
    ann_data.obs["time_point"] = time_step
    return ann_data


def ann2traj(ann_data):
    time_idx = [np.where(ann_data.obs.time_point == t)[0] for t in natsort.natsorted(ann_data.obs.time_point.unique())]
    traj_data_list = [ann_data.X[idx, :] for idx in time_idx]
    traj_data = np.asarray(traj_data_list)
    traj_data = np.moveaxis(traj_data, [0, 1, 2], [1, 0, 2])
    return traj_data

# ---------------------------------

def preprocess(ann_data):
    # adopt recipe_zheng17 w/o HVG selection
    # omit scaling part to avoid information leakage
    scanpy.pp.normalize_per_cell(  # normalize with total UMI count per cell
        ann_data, key_n_counts='n_counts_all', counts_per_cell_after=1e4
    )
    scanpy.pp.log1p(ann_data)  # log transform: adata.X = log(adata.X + 1)
    return ann_data


def postprocess(data):
    # data: cell x gene matrix
    if isinstance(data, np.ndarray):
        norm_data = (data / np.sum(data, axis=1)[:, np.newaxis]) * 1e4
        log_data = np.log(norm_data + 1)
    else:
        norm_data = (data / torch.sum(data, dim=1).unsqueeze(dim=1)) * 1e4
        log_data = torch.log(norm_data + 1)
    return log_data

# ---------------------------------

def sampleOT(true_data, pred_data, sample_n, sample_T):
    ot_list = []
    for _ in range(sample_T):
        true_rand_idx = np.random.choice(np.arange(true_data.shape[0]), sample_n, replace=False)
        pred_rand_idx = np.random.choice(np.arange(pred_data.shape[0]), sample_n, replace=False)
        ot_list.append(_ot(true_data[true_rand_idx,:], pred_data[pred_rand_idx,:]))
    return np.mean(ot_list)


def sampleGaussian(mean, std):
    '''
    Sampling with the re-parametric trick.
    '''
    d = dist.normal.Normal(torch.Tensor([0.]), torch.Tensor([1.]))
    r = d.sample(mean.size()).squeeze(-1)
    x = r * std.float() + mean.float()
    return x