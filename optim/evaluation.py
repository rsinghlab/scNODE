'''
Description:
    Model evaluations.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''

import numpy as np
import torch
from scipy.spatial.distance import cdist
from geomloss import SamplesLoss
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# =====================================
#                UTILS
# =====================================
def basicStats(data, axis="cell"):
    '''
    Compute mean, variance, and fraction of zeros for each cell or gene.
    '''
    data  = np.asarray(data)
    n_cells, n_genes = data.shape
    if axis == "cell":
        # cell average and var
        expression_avg = np.mean(data, axis=1)
        expression_var = np.var(data, axis=1)
        # fraction of zero
        zero_fraction = np.array([len(np.where(cell==0)[0])/n_genes for cell in data])
    elif axis == "gene":
        # gene average and var
        expression_avg = np.mean(data, axis=0)
        expression_var = np.var(data, axis=0)
        # fraction of zero
        zero_fraction = np.array([len(np.where(data[:,i]==0)[0])/n_cells for i in range(n_genes)])
    else:
        raise ValueError("Undefined axis {}! Should be \"cell\" or \"gene\".".format(axis))
    return expression_avg, expression_var, zero_fraction


# =====================================
#     GLOBAL EVALUATION
# =====================================

def _unbalancedDist(true_data, pred_data):
    '''
    Compute pair-wise distance.

    Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

    :param true_data (numpy.ndarray): True expression data.
    :param pred_data (numpy.ndarray): Predicted expression data.
    :return:
        (float) Pair-wise L2 distance.
        (float) Pair-wise cosine distance.
        (float) Pair-wise correlation distance.
    '''
    l2_dist = cdist(true_data, pred_data, metric="euclidean")
    cos_dist = cdist(true_data, pred_data, metric="cosine")
    corr_dist = cdist(true_data, pred_data, metric="correlation")
    avg_l2 = l2_dist.sum() / np.prod(l2_dist.shape)
    avg_cos = cos_dist.sum() / np.prod(cos_dist.shape)
    avg_corr = corr_dist.sum() / np.prod(corr_dist.shape)
    return avg_l2, avg_cos, avg_corr


def _ot(true_data, pred_data):
    '''
    Compute Wasserstein distance with Sinkhorn algorithm.
    :param true_data (numpy.ndarray): True expression data.
    :param pred_data (numpy.ndarray): Predicted expression data.
    :return: (float) Wasserstein distance.
    '''
    ot_solver = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.5, debias=True, backend="tensorized")
    # ot_solver = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=1.0, debias=True, backend="tensorized")
    if isinstance(true_data, np.ndarray):
        true_data = torch.DoubleTensor(true_data)
    if isinstance(true_data, torch.FloatTensor):
        true_data = torch.DoubleTensor(true_data.detach().numpy())
    if isinstance(pred_data, np.ndarray):
        pred_data = torch.DoubleTensor(pred_data)
    if isinstance(pred_data, torch.FloatTensor):
        pred_data = torch.DoubleTensor(pred_data.detach().numpy())
    ot_loss = ot_solver(true_data, pred_data).item()
    return ot_loss


def globalEvaluation(true_data, pred_data):
    '''Evaluate the difference between true and reconstructed data at a single time point.'''
    assert true_data.shape[1] == pred_data.shape[1]
    l2_dist, cos_dist, corr_dist = _unbalancedDist(true_data, pred_data)
    ot_loss = _ot(true_data, pred_data)
    return {
        "l2": l2_dist, "cos": cos_dist, "corr": corr_dist, "ot": ot_loss,
    }

