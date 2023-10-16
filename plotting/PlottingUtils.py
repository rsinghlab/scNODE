'''
Description:
    Utility functions for figure plotting.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import numpy as np
from umap import UMAP
from sklearn.decomposition import PCA

# ======================================

def computeVisEmbedding(true_data, model_pred_data, embed_name):
    if embed_name == "umap":
        true_umap_traj, umap_model = umapWithoutPCA(np.concatenate(true_data, axis=0), n_neighbors=50, min_dist=0.2)
        model_pred_umap_traj = [umap_model.transform(np.concatenate(m_pred, axis=0)) for m_pred in model_pred_data]
    elif embed_name == "pca_umap":
        true_umap_traj, umap_model, pca_model = umapWithPCA(
            np.concatenate(true_data, axis=0), n_neighbors=50, min_dist=0.1, pca_pcs=50
        )
        model_pred_umap_traj = [
            umap_model.transform(pca_model.transform(np.concatenate(m_pred, axis=0)))
            for m_pred in model_pred_data
        ]
    else:
        raise ValueError("Unknown embedding type {}!".format(embed_name))
    return true_umap_traj, model_pred_umap_traj


def umapWithPCA(traj_data, n_neighbors, min_dist, pca_pcs):
    pca_model = PCA(n_components=pca_pcs, svd_solver="arpack")
    umap_model = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist)
    umap_traj_data = umap_model.fit_transform(pca_model.fit_transform(traj_data))
    return umap_traj_data, umap_model, pca_model


def umapWithoutPCA(traj_data, n_neighbors, min_dist):
    umap_model = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist)
    umap_traj_data = umap_model.fit_transform(traj_data)
    return umap_traj_data, umap_model


def onlyPCA(traj_data, pca_pcs):
    pca_model = PCA(n_components=pca_pcs, svd_solver="arpack")
    pca_traj_data = pca_model.fit_transform(traj_data)
    return pca_traj_data, pca_model

# ======================================

def computeLatentEmbedding(latent_seq, next_seq, n_neighbors, min_dist):
    latent_tp_list = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(latent_seq)])
    umap_latent_data, umap_model = umapWithoutPCA(np.concatenate(latent_seq, axis=0), n_neighbors=n_neighbors, min_dist=min_dist)
    # umap_latent_data, umap_model = onlyPCA(np.concatenate(latent_seq, axis=0), pca_pcs=2)
    umap_latent_data = [umap_latent_data[np.where(latent_tp_list == t)[0], :] for t in range(len(latent_seq))]
    umap_next_data = [umap_model.transform(each) for each in next_seq]
    return umap_latent_data, umap_next_data, umap_model, latent_tp_list
