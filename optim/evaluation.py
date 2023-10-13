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

# =====================================
#     ALIGNMENT EVALUATION
# =====================================
# Author: Justin Sanders

def _computeSimpson(
    distances: np.ndarray,
    indices: np.ndarray,
    labels: pd.Categorical,
    n_categories: int,
    perplexity: float,
    tol: float=1e-5
):
    n = distances.shape[1]
    P = np.zeros(distances.shape[0])
    simpson = np.zeros(n)
    logU = np.log(perplexity)
    # Loop through each cell.
    for i in range(n):
        beta = 1
        betamin = -np.inf
        betamax = np.inf
        # Compute Hdiff
        P = np.exp(-distances[:,i] * beta)
        P_sum = np.sum(P)
        if P_sum == 0:
            H = 0
            P = np.zeros(distances.shape[0])
        else:
            H = np.log(P_sum) + beta * np.sum(distances[:,i] * P) / P_sum
            P = P / P_sum
        Hdiff = H - logU
        n_tries = 50
        for t in range(n_tries):
            # Stop when we reach the tolerance
            if abs(Hdiff) < tol:
                break
            # Update beta
            if Hdiff > 0:
                betamin = beta
                if not np.isfinite(betamax):
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if not np.isfinite(betamin):
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2
            # Compute Hdiff
            P = np.exp(-distances[:,i] * beta)
            P_sum = np.sum(P)
            if P_sum == 0:
                H = 0
                P = np.zeros(distances.shape[0])
            else:
                H = np.log(P_sum) + beta * np.sum(distances[:,i] * P) / P_sum
                P = P / P_sum
            Hdiff = H - logU
        # distancesefault value
        if H == 0:
            simpson[i] = -1
        # Simpson's index
        for label_category in labels.categories:
            ix = indices[:,i]
            q = labels[ix] == label_category
            if np.any(q):
                P_sum = np.sum(P[q])
                simpson[i] += P_sum * P_sum
    return simpson



def LISIScore(
    X: np.array,
    metadata: pd.DataFrame,
    label_colnames,
    perplexity: float=30
):
    """Compute the Local Inverse Simpson Index (LISI) for each column in metadata.
    LISI is a statistic computed for each item (row) in the data matrix X.
    The following example may help to interpret the LISI values.
    Suppose one of the columns in metadata is a categorical variable with 3 categories.
        - If LISI is approximately equal to 3 for an item in the data matrix,
          that means that the item is surrounded by neighbors from all 3
          categories.
        - If LISI is approximately equal to 1, then the item is surrounded by
          neighbors from 1 category.

    The LISI statistic is useful to evaluate whether multiple datasets are
    well-integrated by algorithms such as Harmony [1].
    [1]: Korsunsky et al. 2019 doi: 10.1038/s41592-019-0619-0
    """
    n_cells = metadata.shape[0]
    n_labels = len(label_colnames)
    # We need at least 3 * n_neigbhors to compute the perplexity
    knn = NearestNeighbors(n_neighbors = perplexity * 3, algorithm = 'kd_tree').fit(X)
    distances, indices = knn.kneighbors(X)
    # Don't count yourself
    indices = indices[:,1:]
    distances = distances[:,1:]
    # Save the result
    lisi_df = np.zeros((n_cells, n_labels))
    for i, label in enumerate(label_colnames):
        labels = pd.Categorical(metadata[label])
        n_categories = len(labels.categories)
        simpson = _computeSimpson(distances.T, indices.T, labels, n_categories, perplexity)
        lisi_df[:,i] = 1 / simpson
    return lisi_df
