'''
Description:
    Prepare data for PRESCIENT model.
    Codes are adopted from PRESCIENT source codes.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>

Reference:
    https://github.com/gifford-lab/prescient/blob/master/prescient/commands/process_data.py
'''
import numpy as np
import sklearn
import umap
import torch


def prepareData(expr_mat, tps, num_pcs, num_neighbors_umap):
    # transformations
    scaler = sklearn.preprocessing.StandardScaler()
    pca = sklearn.decomposition.PCA(n_components = num_pcs)
    um = umap.UMAP(n_components = 2, metric = 'euclidean', n_neighbors = num_neighbors_umap)

    x = scaler.fit_transform(expr_mat)
    xp = pca.fit_transform(x)
    xu = um.fit_transform(xp)

    y = list(np.sort(np.unique(tps)))

    x_ = [torch.from_numpy(x[(tps == d),:]).float() for d in y]
    xp_ = [torch.from_numpy(xp[(tps == d),:]).float() for d in y]
    xu_ = [torch.from_numpy(xu[(tps == d),:]).float() for d in y]

    return expr_mat, x_, xp_, xu_, y, pca, um, tps, scaler, pca, um


def main(expr_mat, tps, cell_types, genes, num_pcs=50, num_neighbors_umap=10):
    """
    Outputs:
    --------
    Saves a PRESCIENT file to out_path. Does not output file.
    data.pt:
        |- x: scaled expression
        |- xp: n PC space
        |- xu: UMAP space
        |- pca: sklearn pca object for pca tranformation
        |- um: umap object for umap transformation
        |- y: timepoints
        |- genes: features
        |- w: growth weights
        |- celltype: vector of celltype labels
    """
    expr, x, xp, xu, y, pca, um, tps, scaler, pca, um = prepareData(expr_mat, tps, num_pcs, num_neighbors_umap)

    growth_rate = [np.ones((each.shape[0],)) for each in x]
    w = growth_rate

    return {
     "data": expr,
     "genes": genes,
     "celltype": cell_types,
     "tps": tps,
     "x":x,
     "xp":xp,
     "xu": xu,
     "y": y,
     "pca": pca,
     "um":um,
     "w":w
     }, scaler, pca, um