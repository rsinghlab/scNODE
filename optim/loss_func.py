'''
Description:
    Loss functions.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist

# ===========================================
# Loss I: MSE reconstruction loss + KL divergence for the first latent
def MSEAndKL(true_obs, est_obs, prior_latent_dist, est_latent_dist, tradeoff):
    true_obs = true_obs.unsqueeze(0)
    true_obs = torch.repeat_interleave(true_obs, est_obs.shape[0], dim=0)
    recon_loss_func = nn.MSELoss(reduction="none")
    recon_loss = recon_loss_func(est_obs, true_obs)
    recon_loss = torch.sum(recon_loss) / np.prod(true_obs.shape)
    kl_loss = dist.kl.kl_divergence(est_latent_dist, prior_latent_dist)
    kl_loss = torch.mean(kl_loss)
    loss = tradeoff * recon_loss + (1 - tradeoff) * kl_loss
    return loss, kl_loss, recon_loss


def MSELoss(true_obs, est_obs):
    loss_func = nn.MSELoss(reduction="mean")
    return loss_func(est_obs, true_obs)

# ===========================================
# Loss II: Gaussian likelihood + KL divergence for the first latent
def GLLAndKL(true_obs, est_obs, prior_latent_dist, est_latent_dist, tradeoff, observed_var):
    true_obs = true_obs.unsqueeze(0)
    true_obs = torch.repeat_interleave(true_obs, est_obs.shape[0], dim=0)
    observed_var = observed_var * torch.ones_like(true_obs)
    nll_loss_func = nn.GaussianNLLLoss(reduction="none")
    nll_loss = nll_loss_func(est_obs, true_obs, var=observed_var)
    nll_loss = torch.sum(nll_loss) / np.prod(true_obs.shape)
    kl_loss = dist.kl.kl_divergence(est_latent_dist, prior_latent_dist)
    kl_loss = torch.mean(kl_loss)
    loss = tradeoff * nll_loss + (1 - tradeoff) * kl_loss
    return loss, kl_loss, nll_loss


# ===========================================
# Loss III: IWAE Gaussian likelihood + KL divergence for the first latent
def IWAEAndKL(true_obs, est_obs, prior_latent_dist, est_latent_dist, tradeoff, observed_var):
    true_obs = true_obs.unsqueeze(0)
    true_obs = torch.repeat_interleave(true_obs, est_obs.shape[0], dim=0)
    observed_var = observed_var * torch.ones_like(true_obs)
    nll_loss_func = nn.GaussianNLLLoss(reduction="none")
    nll_loss = nll_loss_func(est_obs, true_obs, var=observed_var)
    nll_loss = torch.mean(nll_loss, dim=[1, 2, 3])
    kl_loss = dist.kl.kl_divergence(est_latent_dist, prior_latent_dist)
    kl_loss = torch.mean(kl_loss)
    loss = torch.logsumexp(tradeoff*nll_loss + (1-tradeoff)*kl_loss, dim=0)
    return loss, kl_loss, nll_loss.mean()


# ===========================================
# Loss IV: Sinkhorn loss

# geomloss implementation
import geomloss
def SinkhornLoss(true_obs, est_obs, blur=0.05, scaling=0.5, batch_size=None):
    # est_obs = est_obs.squeeze()
    n_tps = len(true_obs)
    loss = 0.0
    ot_solver = geomloss.SamplesLoss("sinkhorn", p=2, blur=blur, scaling=scaling, debias=True, backend="tensorized")
    for t in range(n_tps):
        t_est = est_obs[:, t, :]
        t_true = true_obs[t]
        if batch_size is not None:
            cell_idx = np.random.choice(np.arange(t_true.shape[0]), size = batch_size, replace = (t_true.shape[0] < batch_size))
            t_true = t_true[cell_idx, :]
        t_loss = ot_solver(t_true, t_est)
        loss += t_loss
    loss = loss / n_tps
    return loss


def SinkhornLossWithWeight(true_obs, est_obs, true_weight, blur=0.05, scaling=0.5, batch_size=None):
    # est_obs = est_obs.squeeze()
    n_tps = len(true_obs)
    loss = 0.0
    ot_solver = geomloss.SamplesLoss("sinkhorn", p=2, blur=blur, scaling=scaling, debias=True, backend="tensorized")
    for t in range(n_tps):
        t_est = est_obs[:, t, :]
        t_true = true_obs[t]
        t_weight = true_weight[t]
        est_weight = torch.ones(t_est.shape[0]).type_as(t_est) / t_est.shape[0]
        if batch_size is not None:
            cell_idx = np.random.choice(np.arange(t_true.shape[0]), size = batch_size, replace = (t_true.shape[0] < batch_size))
            t_true = t_true[cell_idx, :]
            t_weight = t_weight[cell_idx]
        t_loss = ot_solver(t_weight, t_true, est_weight, t_est)
        loss += t_loss
    loss = loss / n_tps
    return loss


def SinkhornLoss4FNN(true_obs, est_obs, blur=0.05, scaling=0.5, batch_size=None):
    # Sonkhorn loss for FNN training
    n_tps = len(true_obs)
    loss = 0.0
    ot_solver = geomloss.SamplesLoss("sinkhorn", p=2, blur=blur, scaling=scaling, debias=True, backend="tensorized")
    for t in range(n_tps):
        t_est = est_obs[t]
        t_true = true_obs[t]
        if batch_size is not None:
            cell_idx = np.random.choice(np.arange(t_true.shape[0]), size = batch_size, replace = (t_true.shape[0] < batch_size))
            t_true = t_true[cell_idx, :]
        t_loss = ot_solver(t_true, t_est)
        loss += t_loss
    loss = loss / n_tps
    return loss

# ===========================================
# Loss V: Fused loss
def fusedLoss(est_obs):
    loss_func = nn.MSELoss(reduction="mean")
    n_tps = est_obs.shape[1]
    loss = 0.0
    for t in range(n_tps-1):
        t_loss = loss_func(est_obs[:, t, :], est_obs[:, t+1, :])
        loss += t_loss
    loss = loss / (n_tps-1)
    return loss


# ===========================================
# Loss VI: More compact loss
# Loss = OT + MSE for the first time + OT(enc latent, DE latent)
def OTAndLatentMSE(
        true_obs, est_obs,
        first_time_true_batch, first_time_pred_batch,
        encoder_latent_seq, de_latent_seq,
        mse_coeff, latent_coeff,
        blur=0.05, scaling=0.5, batch_size=None
):
    # OT loss between true and reconstructed cell sets at each time point
    ot_loss = SinkhornLoss(true_obs, est_obs, blur, scaling, batch_size)
    # MSE between the true and reconstructed data at first time point
    first_mse = nn.MSELoss(reduction="mean")(first_time_true_batch, first_time_pred_batch)
    # Difference between encoder latent and DE latent
    latent_diff = SinkhornLoss(encoder_latent_seq, de_latent_seq, blur, scaling, batch_size=None)
    #TODO: KL(latent, Gaussian)
    loss = ot_loss + mse_coeff * first_mse + latent_coeff * latent_diff
    return loss

# ===========================================
# Loss VII: UMAP loss
from umap.umap_ import find_ab_params
def umapLoss(graph, latent, spread=1.0, min_dist=0.1):
    tol = 1e-7
    p = graph + tol  # node pair weight in input space
    a, b = find_ab_params(spread=spread, min_dist=min_dist)
    pair_dist = torch.cdist(latent, latent, p=2.0)
    q = 1 / (1 + a * torch.pow(pair_dist, 2 * b)) + tol  # node pair weight in latent space
    pos_loss = p * torch.log(p / q)
    neg_loss = (1 - p) * torch.log((1 - p) / (1 - q))
    if torch.any(torch.isnan(neg_loss)):
        neg_loss[torch.isnan(neg_loss)] = 1 - p[torch.isnan(neg_loss)].double()  # avoid the case where q=1
    pair_loss = pos_loss + neg_loss
    cluster_loss = pair_loss.mean()
    return cluster_loss

# ===========================================
# Loss VIII: NB negative log-likelihood loss
def NBLikelihood(true_X, est_total_cnt, est_prob, n_samples=None):
    est_dist = dist.negative_binomial.NegativeBinomial(est_total_cnt, est_prob)
    likelihood = est_dist.log_prob(true_X)
    nll = torch.mean(-likelihood)
    return nll


def NBSeqLikelihood(true_obs, est_total_cnt, est_prob, n_samples=1):
    n_tps = len(true_obs)
    batch_size = est_total_cnt.shape[0]
    nll = 0.0
    for t in range(n_tps):
        t_est_dist = dist.negative_binomial.NegativeBinomial(est_total_cnt[:, t, :], est_prob[:, t, :])
        t_true = true_obs[t]
        t_nll = 0
        for s in range(n_samples):
            cell_idx = np.random.choice(np.arange(t_true.shape[0]), size=batch_size, replace=False)
            l = torch.mean(-t_est_dist.log_prob(t_true[cell_idx, :]))
            t_nll += l
        t_nll /= n_samples
        nll += t_nll
    loss = nll / n_tps
    return loss