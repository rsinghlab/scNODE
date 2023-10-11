'''
Description:
    Model evaluations.
'''

import numpy as np
import scipy.stats
import torch
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, wasserstein_distance, gaussian_kde
from scipy.integrate import quad
from geomloss import SamplesLoss
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_curve, precision_recall_curve, auc
from optim.ndtest import ks2d2s

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


def _ksTest(true, pred):
    ks_res = ks_2samp(true, pred)
    return ks_res


def _ks2DTest(true, pred):
    '''
    Two-dimensional KS test on two samples.

    Reference:
        https://github.com/syrte/ndtest
    '''
    ks_res = ks2d2s(true[:, 0], true[:, 1], pred[:, 0], pred[:, 1], extra=True)
    return ks_res # 0: p-vale; 1: statistics


def _wassDist(true, pred):
    if len(true.shape) == 1:
        true = true[:, np.newaxis]
        pred = pred[:, np.newaxis]
    # dist = wasserstein_distance(true, pred)
    wass_solver = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, scaling=0.5, debias=True, backend="tensorized")
    dist = wass_solver(torch.FloatTensor(true), torch.FloatTensor(pred)).item()
    return dist


def _ise(true, pred):
    try:
        true_kde = gaussian_kde(true)
        pred_kde = gaussian_kde(pred)
        ise = quad(lambda x: np.square(true_kde(x) - pred_kde(x)), -np.inf, np.inf)[0]
    except:
        return None
    return ise

# =====================================
#     ONE DIMENSIONAL EVALUATION
# =====================================
def oneDimEvaluation(true_stats, pred_stats):
    true_avg, true_var, true_zero = true_stats
    pred_avg, pred_var, pred_zero = pred_stats
    # -----
    avg_ks = _ksTest(true_avg, pred_avg)
    var_ks = _ksTest(true_var, pred_var)
    zero_ks = _ksTest(true_zero, pred_zero)
    # -----
    avg_wass = _wassDist(true_avg, pred_avg)
    var_wass = _wassDist(true_var, pred_var)
    zero_wass = _wassDist(true_zero, pred_zero)
    # -----
    avg_ise = _ise(true_avg, pred_avg)
    var_ise = _ise(true_var, pred_var)
    zero_ise = _ise(true_zero, pred_zero)
    # -----
    return {
        "avg": {"ks": avg_ks, "wass": avg_wass, "ise": avg_ise},
        "var": {"ks": var_ks, "wass": var_wass, "ise": var_ise},
        "zero": {"ks": zero_ks, "wass": zero_wass, "ise": zero_ise}
    }


# =====================================
#     TWO DIMENSIONAL EVALUATION
# =====================================
def twoDimEvaluation(true_stats, pred_stats):
    true_avg, true_var, true_zero = true_stats
    pred_avg, pred_var, pred_zero = pred_stats
    # -----
    true_avg_var = np.concatenate([true_avg[:, np.newaxis], true_var[:, np.newaxis]], axis=1)
    true_avg_zero = np.concatenate([true_avg[:, np.newaxis], true_zero[:, np.newaxis]], axis=1)
    true_var_zero = np.concatenate([true_var[:, np.newaxis], true_zero[:, np.newaxis]], axis=1)
    pred_avg_var = np.concatenate([pred_avg[:, np.newaxis], pred_var[:, np.newaxis]], axis=1)
    pred_avg_zero = np.concatenate([pred_avg[:, np.newaxis], pred_zero[:, np.newaxis]], axis=1)
    pred_var_zero = np.concatenate([pred_var[:, np.newaxis], pred_zero[:, np.newaxis]], axis=1)
    # -----
    avg_var_wass = _wassDist(true_avg_var, pred_avg_var)
    avg_zero_wass = _wassDist(true_avg_zero, pred_avg_zero)
    var_zero_wass = _wassDist(true_var_zero, pred_var_zero)
    # -----
    avg_var_ks = _ks2DTest(true_avg_var, pred_avg_var)
    avg_zero_ks = _ks2DTest(true_avg_zero, pred_avg_zero)
    var_zero_ks = _ks2DTest(true_var_zero, pred_var_zero)
    # -----
    return {
        "avg_var": {"ks": avg_var_ks, "wass": avg_var_wass},
        "avg_zero": {"ks": avg_zero_ks, "wass": avg_zero_wass},
        "var_zero": {"ks": var_zero_ks, "wass": var_zero_wass}
    }


# =====================================
#     GLOBAL EVALUATION
# =====================================

def _unbalancedDist(true_data, pred_data):
    l2_dist = cdist(true_data, pred_data, metric="euclidean")
    cos_dist = cdist(true_data, pred_data, metric="cosine")
    corr_dist = cdist(true_data, pred_data, metric="correlation")
    avg_l2 = l2_dist.sum() / np.prod(l2_dist.shape)
    avg_cos = cos_dist.sum() / np.prod(cos_dist.shape)
    avg_corr = corr_dist.sum() / np.prod(corr_dist.shape)
    return avg_l2, avg_cos, avg_corr


def _classification(true_data, pred_data):
    true_cell_y = np.zeros((true_data.shape[0], ))
    pred_cell_y = np.ones((pred_data.shape[0], ))
    y = np.concatenate([true_cell_y, pred_cell_y])
    x = np.concatenate([true_data, pred_data], axis=0)
    # -----
    classifier = RandomForestClassifier(n_estimators=20, max_depth=3)
    classifier.fit(x, y)
    pred_y = classifier.predict(x)
    pred_prob = classifier.predict_proba(x)
    # -----
    f1 = f1_score(y, pred_y)
    precision_list, recall_list, _ = precision_recall_curve(y, pred_prob[:, 1])
    pr_sort_idx = np.argsort(recall_list)
    precision_list = precision_list[pr_sort_idx]
    recall_list = recall_list[pr_sort_idx]
    auprc = auc(recall_list, precision_list)

    fpr_list, tpr_list, _ = roc_curve(y, pred_prob[:, 1])
    roc_sort_idx = np.argsort(fpr_list)
    fpr_list = fpr_list[roc_sort_idx]
    tpr_list = tpr_list[roc_sort_idx]
    auroc = auc(fpr_list, tpr_list)
    return f1, {"precision": precision_list, "recall": recall_list, "auprc": auprc}, {"fpr": fpr_list, "tpr": tpr_list, "auroc": auroc}


def _ot(true_data, pred_data):
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
    # f1, pr_curve, ro_curve = _classification(true_data, pred_data)
    ot_loss = _ot(true_data, pred_data)
    return {
        "l2": l2_dist, "cos": cos_dist, "corr": corr_dist, "ot": ot_loss,
        # "classification": {"f1": f1, "pr_curve": pr_curve, "ro_curve": ro_curve}
    }

# =====================================
#     ALIGNMENT EVALUATION
# =====================================
import pandas as pd
from sklearn.neighbors import NearestNeighbors

#TODO: check this function
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



# =====================================

# (DEPRECATED)
# def evalSingleTP(true_data, pred_data):
#     '''Evaluate the difference between true and reconstructed data at a single time point.'''
#     assert true_data.shape[1] == pred_data.shape[1]
#     l2_dist, cos_dist, corr_dist = _unbalancedDist(true_data, pred_data)
#     f1, pr_curve, ro_curve = _classification(true_data, pred_data)
#     ot_loss = _ot(true_data, pred_data)
#     true_cell_avg, true_cell_var, _ = _basicStats(true_data, axis="cell")
#     true_gene_avg, true_gene_var, _ = _basicStats(true_data, axis="gene")
#     pred_cell_avg, pred_cell_var, _ = _basicStats(pred_data, axis="cell")
#     pred_gene_avg, pred_gene_var, _ = _basicStats(pred_data, axis="gene")
#     return (
#         l2_dist, cos_dist, corr_dist,
#         f1, pr_curve, ro_curve, ot_loss,
#         true_cell_avg, true_cell_var, true_gene_avg, true_gene_var,
#         pred_cell_avg, pred_cell_var, pred_gene_avg, pred_gene_var
#     )





if __name__ == '__main__':
    print("Loading VAE res...")
    vae_filename = "../res/single_cell/experimental/zebrafish/zebrafish-two_forecasting-latent_ODE_OT_pretrain-res.npy"
    res = np.load(vae_filename, allow_pickle=True).item()
    true_data = res["true"]
    pred_data = res["pred"]
    tps = res["tps"]["all"]
    test_tps = res["tps"]["test"]
    print("Num of TPs: ", len(tps))
    print("Testing TPs: ", test_tps)
    print("Construct cell tp list...")
    true_cell_tps = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(true_data)])
    pred_cell_tps = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(pred_data)])
    print("Reorder prediction time points...")
    reorder_pred_data = pred_data
    # # -----
    # cell_1D_metric = oneDimEvaluation(basicStats(true_data[0], axis="cell"), basicStats(reorder_pred_data[0], axis="cell"))
    # gene_1D_metric = oneDimEvaluation(basicStats(true_data[0], axis="gene"), basicStats(reorder_pred_data[0], axis="gene"))
    # # -----
    # cell_2D_metric = twoDimEvaluation(basicStats(true_data[0], axis="cell"), basicStats(reorder_pred_data[0], axis="cell"))
    # gene_2D_metric = twoDimEvaluation(basicStats(true_data[0], axis="gene"), basicStats(reorder_pred_data[0], axis="gene"))
    # # -----
    # global_metric = globalEvaluation(true_data[0], reorder_pred_data[0])
    # -----
    # true_data =  np.random.normal(0.0, 1.0, (50, 20))
    # generated_data =  np.random.normal(0.0, 1.0, (100, 20))
    print("LISI...")
    true_data =  true_data[10]
    generated_data =  reorder_pred_data[10]
    combined = np.concatenate([true_data, generated_data])
    d_labels = np.concatenate([np.zeros(true_data.shape[0], ), np.ones(generated_data.shape[0])])
    lisi = LISIScore(combined, pd.DataFrame(data=d_labels, columns=['Type']), ['Type'])
    print("miLISI={}".format(np.median(lisi)))
    print("avgLISI={}".format(np.mean(lisi)))