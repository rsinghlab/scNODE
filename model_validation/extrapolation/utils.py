'''
Description:
    Utility functions for extrapolating multiple timepoints.

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
    meta_data = pd.read_csv("{}/{}-meta_data.csv".format(data_dir, split_type), header=0, index_col=0)
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


# --------------------------------
default_zebrafish_data_dir = "../data/single_cell/experimental/zebrafish_embryonic/new_processed"
default_wot_data_dir = "../data/single_cell/experimental/Schiebinger2019/processed/"
default_drosophila_data_dir = "../data/single_cell/experimental/drosophila_embryonic/processed/"


def loadSCData(data_name, split_type, data_dir=None):
    print("[ Data={} | Split={} ] Loading data...".format(data_name, split_type))
    if data_name == "zebrafish":
        zebrafish_data_dir = default_zebrafish_data_dir if data_dir is None else data_dir
        ann_data = loadZebrafishData(zebrafish_data_dir, split_type)
        ann_data.X = ann_data.X.astype(float)
        processed_data = preprocess2(ann_data.copy())
        cell_types =  processed_data.obs["ZF6S-Cluster"].apply(lambda x: "NAN" if pd.isna(x) else x).values
    elif data_name == "drosophila":
        drosophila_data_dir = default_drosophila_data_dir if data_dir is None else data_dir
        ann_data = loadDrosophilaData(drosophila_data_dir, split_type)
        print("Pre-processing...")
        ann_data.X = ann_data.X.astype(float)
        processed_data = preprocess2(ann_data.copy())
        cell_types = processed_data.obs.seurat_clusters.values
    elif data_name == "wot":
        wot_data_dir = default_wot_data_dir if data_dir is None else data_dir
        ann_data = loadWOTData(wot_data_dir, split_type)
        ann_data.X = ann_data.X.astype(float)
        processed_data = ann_data.copy()
        cell_types =  None
    else:
        raise ValueError("Unknown data name.")
    cell_tps = ann_data.obs["tp"]
    n_tps = len(np.unique(cell_tps))
    n_genes = ann_data.shape[1]
    return processed_data, cell_tps, cell_types, n_genes, n_tps

# ---------------------------------

def preprocess(ann_data):
    # recipe_zheng17 w/o HVG selection
    scanpy.pp.filter_genes(ann_data, min_counts=1)  # only consider genes with more than 1 count
    scanpy.pp.normalize_per_cell(  # normalize with total UMI count per cell
        ann_data, key_n_counts='n_counts_all'
    )
    scanpy.pp.normalize_per_cell(ann_data)  # renormalize after filtering
    scanpy.pp.log1p(ann_data)  # log transform: adata.X = log(adata.X + 1)
    scanpy.pp.scale(ann_data)  # scale to unit variance and shift to zero mean
    return ann_data


def preprocess2(ann_data):
    # adopt recipe_zheng17 w/o HVG selection
    # omit scaling part to avoid information leakage
    scanpy.pp.normalize_per_cell(  # normalize with total UMI count per cell
        ann_data, key_n_counts='n_counts_all', counts_per_cell_after=1e4
    )
    scanpy.pp.log1p(ann_data)  # log transform: adata.X = log(adata.X + 1)
    return ann_data


def preprocessFactor(ann_data):
    # adopt recipe_zheng17 w/o HVG selection
    # omit scaling part to avoid information leakage
    scanpy.pp.normalize_per_cell(  # normalize with total UMI count per cell
        ann_data, key_n_counts='n_counts_all', counts_per_cell_after=1e4
    )
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

def tunedOurPars(data_name, forecast_num):
    latent_dim = 50
    int_forecast_num = int(forecast_num)
    if data_name == "zebrafish":
        if int_forecast_num == 1:
            drift_latent_size = [50, 50]
            enc_latent_list = None
            dec_latent_list = None
        elif int_forecast_num == 2:
            drift_latent_size = [50, 50]
            enc_latent_list = None
            dec_latent_list = None
        elif int_forecast_num == 3:
            drift_latent_size = [50, 50]
            enc_latent_list = None
            dec_latent_list = None
        elif int_forecast_num == 4:
            drift_latent_size = [50, 50]
            enc_latent_list = None
            dec_latent_list = None
        elif int_forecast_num == 5:
            drift_latent_size = [50, 50]
            enc_latent_list = None
            dec_latent_list = None
    elif data_name == "drosophila":
        if int_forecast_num == 1:
            drift_latent_size = [50, 50]
            enc_latent_list = [50, 50]
            dec_latent_list = [50]
        elif int_forecast_num == 2:
            drift_latent_size = [50]
            enc_latent_list = [50, 50]
            dec_latent_list = [50]
        elif int_forecast_num == 3:
            drift_latent_size = [50, 50]
            enc_latent_list = None
            dec_latent_list = [50, 50]
        elif int_forecast_num == 4:
            drift_latent_size = None
            enc_latent_list = [50]
            dec_latent_list = [50]
        elif int_forecast_num == 5:
            drift_latent_size = [50, 50]
            enc_latent_list = None
            dec_latent_list = [50]
    elif data_name == "wot":
        if int_forecast_num == 1:
            drift_latent_size = [50, 50]
            enc_latent_list = None
            dec_latent_list = [50]
        elif int_forecast_num == 2:
            drift_latent_size = [50, 50]
            enc_latent_list = None
            dec_latent_list = None
        elif int_forecast_num == 3:
            drift_latent_size = [50, 50]
            enc_latent_list = None
            dec_latent_list = None
        elif int_forecast_num == 4:
            drift_latent_size = [50, 50]
            enc_latent_list = None
            dec_latent_list = None
        elif int_forecast_num == 5:
            drift_latent_size = [50, 50]
            enc_latent_list = None
            dec_latent_list = None
    else:
        raise ValueError("Unknown data name {}!".format(data_name))
    return latent_dim, drift_latent_size, enc_latent_list, dec_latent_list


def tunedPRESCIENTPars(data_name, forecast_num):
    latent_dim = 50
    int_forecast_num = int(forecast_num)
    if data_name == "zebrafish":
        if int_forecast_num == 1:
            layers = 3
            sd = 0.9970791424147027
            tau = 0.07248433046655238
            clip = 0.7338387029572158
        elif int_forecast_num == 2:
            layers = 3
            sd = 0.9840472138501711
            tau = 0.027988650301849
            clip = 0.3166979443459986
        elif int_forecast_num == 3:
            layers = 2
            sd = 0.99983011383432
            tau = 0.033002016807090695
            clip = 0.41156800706268715
        elif int_forecast_num == 4:
            layers = 2
            sd = 0.9831207765278598
            tau = 0.060388026537710235
            clip = 0.6863455582078853
        elif int_forecast_num == 5:
            layers = 3
            sd = 0.9396991323142709
            tau = 0.03711916415928869
            clip = 0.31765646339718157
    elif data_name == "drosophila":
        if int_forecast_num == 1:
            layers = 3
            sd = 0.9488061243799828
            tau = 0.029102780061998616
            clip = 0.5478734656660784
        elif int_forecast_num == 2:
            layers = 3
            sd = 0.9009943617077583
            tau = 0.015318049375944581
            clip = 0.3265838330326015
        elif int_forecast_num == 3:
            layers = 3
            sd = 0.9983864694557134
            tau = 0.037704848889189665
            clip = 0.4134837293153374
        elif int_forecast_num == 4:
            layers = 3
            sd = 0.9975565900992603
            tau = 0.027099372699169993
            clip = 0.34748430115330253
        elif int_forecast_num == 5:
            layers = 3
            sd = 0.9946498381032968
            tau = 0.06333392904253891
            clip = 0.5612788547491271
    elif data_name == "wot":
        if int_forecast_num == 1:
            layers = 2
            sd = 0.947119501628167
            tau = 0.07181961146271176
            clip = 0.772573208138766
        elif int_forecast_num == 2:
            layers = 2
            sd = 0.9911186905514884
            tau = 0.09882635320235982
            clip = 0.7113864275538437
        elif int_forecast_num == 3:
            layers = 2
            sd = 0.994083765527351
            tau = 0.08087861833026336
            clip = 0.6134723513783651
        elif int_forecast_num == 4:
            layers = 2
            sd = 0.9984777307107224
            tau = 0.043226720839861836
            clip = 0.19038231499734223
        elif int_forecast_num == 5:
            layers = 2
            sd = 0.9978677412975496
            tau = 0.09594624861323338
            clip = 0.6878489599579647
    else:
        raise ValueError("Unknown data name {}!".format(data_name))
    return latent_dim, layers, sd, tau, clip


def tunedMIOFlowPars(data_name, forecast_num):
    int_forecast_num = int(forecast_num)
    if data_name == "zebrafish":
        if int_forecast_num == 1:
            gae_embedded_dim = 10
            encoder_layers = [50, 100, 100]
            layers = [16, 32, 16]
            lambda_density = 10.134463796155387
        elif int_forecast_num == 2:
            gae_embedded_dim = 10
            encoder_layers = [50, 100, 100]
            layers = [16, 32, 16]
            lambda_density = 15.566371231575735
        elif int_forecast_num == 3:
            gae_embedded_dim = 10
            encoder_layers = [50, 100, 100]
            layers = [16, 32, 16]
            lambda_density = 28.847718979657923
        elif int_forecast_num == 4:
            gae_embedded_dim = 10
            encoder_layers = [50, 100, 100]
            layers = [16, 32, 16]
            lambda_density = 8.262602476853154
        elif int_forecast_num == 5:
            gae_embedded_dim = 10
            encoder_layers = [50, 100]
            layers = [16, 32, 16]
            lambda_density = 3.031423040384798
    elif data_name == "drosophila":
        if int_forecast_num == 1:
            gae_embedded_dim = 10
            encoder_layers = [50, 100, 100]
            layers = [16, 32, 16]
            lambda_density = 99.80468594998474
        elif int_forecast_num == 2:
            gae_embedded_dim = 10
            encoder_layers = [50, 100, 100]
            layers = [16, 32, 16]
            lambda_density = 17.39580523831568
        elif int_forecast_num == 3:
            gae_embedded_dim = 10
            encoder_layers = [50, 100, 100]
            layers = [16, 32, 16]
            lambda_density = 29.93834796990523
        elif int_forecast_num == 4:
            gae_embedded_dim = 10
            encoder_layers = [50, 100, 100]
            layers = [16, 32, 16]
            lambda_density = 56.268518541348485
        elif int_forecast_num == 5:
            gae_embedded_dim = 10
            encoder_layers = [50, 100, 100]
            layers = [16, 32, 16]
            lambda_density = 35.15713158861297
    elif data_name == "wot":
        if int_forecast_num == 1:
            gae_embedded_dim = 10
            encoder_layers = [50, 100]
            layers = [16, 32, 16]
            lambda_density = 77.67113662252478
        elif int_forecast_num == 2:
            gae_embedded_dim = 10
            encoder_layers = [50, 100]
            layers = [16, 32, 16]
            lambda_density = 3.242683324622826
        elif int_forecast_num == 3:
            gae_embedded_dim = 10
            encoder_layers = [50, 100]
            layers = [16, 32, 16]
            lambda_density = 88.04208924628131
        elif int_forecast_num == 4:
            gae_embedded_dim = 10
            encoder_layers = [50, 100]
            layers = [16, 32, 16]
            lambda_density = 10.84536497202019
        elif int_forecast_num == 5:
            gae_embedded_dim = 10
            encoder_layers = [50, 100]
            layers = [16, 32, 16]
            lambda_density = 94.55651694955769
    else:
        raise ValueError("Unknown data name {}!".format(data_name))
    encoder_layers = encoder_layers + [gae_embedded_dim]
    return gae_embedded_dim, encoder_layers, layers, lambda_density

# ---------------------------------

def sampleOT(true_data, pred_data, sample_n, sample_T):
    ot_list = []
    for _ in range(sample_T):
        true_rand_idx = np.random.choice(np.arange(true_data.shape[0]), sample_n, replace=False)
        pred_rand_idx = np.random.choice(np.arange(pred_data.shape[0]), sample_n, replace=False)
        ot_list.append(_ot(true_data[true_rand_idx,:], pred_data[pred_rand_idx,:]))
    return np.mean(ot_list)

# ---------------------------------

def sampleGaussian(mean, std):
    '''
    Sampling with the re-parametric trick.
    '''
    d = dist.normal.Normal(torch.Tensor([0.]), torch.Tensor([1.]))
    r = d.sample(mean.size()).squeeze(-1)
    x = r * std.float() + mean.float()
    return x


def sampleNB(count, prob):
    nb_dist = dist.negative_binomial.NegativeBinomial(count, prob)
    obs = nb_dist.sample()
    return obs