'''
Description:
    Model model_wrapper of scNODE models.
'''
import copy

import torch
from tqdm import tqdm
import numpy as np
import itertools

from optim.loss_func import SinkhornLoss, MSELoss
from benchmark.BenchmarkUtils import sampleGaussian

# =============================================

def scNODETrain4Adjustment(
        train_data, train_tps, latent_ode_model, latent_coeff, epochs, iters, batch_size, lr,
        pretrain_iters=200, pretrain_lr=1e-3, only_train_de=False, true_batch_size=200, save_iter=50
):
    latent_encoder = latent_ode_model.latent_encoder
    obs_decoder = latent_ode_model.obs_decoder
    # Parametric dimensionality reduction training with all training data
    pretrained_model = None
    if pretrain_iters > 0:
        pretrain_data = train_data
        all_pretrain_data = torch.cat(pretrain_data, dim=0)
        # all_pretrain_tps = np.concatenate([np.repeat(t, pretrain_data[i].shape[0]) for i, t in enumerate(train_tps)])
        # -----
        dim_reduction_params = itertools.chain(*[latent_encoder.parameters(), obs_decoder.parameters()])
        dim_reduction_optimizer = torch.optim.Adam(params=dim_reduction_params, lr=pretrain_lr, betas=(0.95, 0.99))
        # dim_reduction_optimizer = torch.optim.RMSprop(params=dim_reduction_params, lr=pretrain_lr)
        dim_reduction_pbar = tqdm(range(pretrain_iters), desc="[ Pre-Training ]")
        latent_encoder.train()
        obs_decoder.train()
        dim_reduction_loss_list = []
        for t in dim_reduction_pbar:
            dim_reduction_optimizer.zero_grad()
            latent_mu, latent_std = latent_encoder(all_pretrain_data)
            latent_sample = sampleGaussian(latent_mu, latent_std)
            recon_obs = obs_decoder(latent_sample)
            dim_reduction_loss = MSELoss(all_pretrain_data, recon_obs)
            # KL div between latent dist and N(0, 1)
            # kl_coeff = 0.0
            # kl_div = (latent_std**2 + latent_mu**2 - 2*torch.log(latent_std + 1e-5)).mean()
            # vae_loss = kl_coeff * kl_div + dim_reduction_loss
            vae_loss = dim_reduction_loss
            # Backward
            dim_reduction_pbar.set_postfix({"Loss": "{:.3f}".format(vae_loss)})
            # dim_reduction_pbar.set_postfix({"Loss": "{:.3f} | MSE={:.3f}, KL={:.3f}".format(vae_loss, dim_reduction_loss, kl_div)})
            # dim_reduction_loss_list.append(dim_reduction_loss.item())
            # dim_reduction_loss.backward()
            dim_reduction_loss_list.append(vae_loss.item())
            vae_loss.backward()
            dim_reduction_optimizer.step()
        pretrained_model = {"enc": copy.deepcopy(latent_encoder.state_dict()), "dec": copy.deepcopy(obs_decoder.state_dict())}
    # -----
    # Dynamic learning
    scNODE_model_list = []
    latent_ode_model.latent_encoder = latent_encoder
    latent_ode_model.obs_decoder = obs_decoder
    num_IWAE_sample = 1
    blur = 0.05
    scaling = 0.5
    cnt = 0
    loss_list = []
    if only_train_de:
        print("Only train DE in dynamic learning")
        optimizer = torch.optim.Adam(params=latent_ode_model.diffeq_decoder.net.parameters(), lr=lr, betas=(0.95, 0.99))
    else:
        optimizer = torch.optim.Adam(params=latent_ode_model.parameters(), lr=lr, betas=(0.95, 0.99))
    latent_ode_model.train()
    scNODE_model_list.append(copy.deepcopy(latent_ode_model.state_dict()))
    for e in range(epochs):
        epoch_pbar = tqdm(range(iters), desc="[ Epoch {} ]".format(e + 1))
        for t in epoch_pbar:
            optimizer.zero_grad()
            recon_obs, first_latent_dist, first_time_true_batch, latent_seq = latent_ode_model(
                train_data, train_tps, num_IWAE_sample, batch_size=batch_size)
            encoder_latent_seq = [
                latent_ode_model.singleReconstruct(
                    each[np.random.choice(np.arange(each.shape[0]), size=batch_size, replace=(each.shape[0] < batch_size)), :]
                )[1]
                for each in train_data
            ]
            # -----
            # OT loss between true and reconstructed cell sets at each time point
            # ot_loss = SinkhornLoss(train_data, recon_obs, blur=blur, scaling=scaling, batch_size=200)
            ot_loss = SinkhornLoss(train_data, recon_obs, blur=blur, scaling=scaling, batch_size=true_batch_size)
            # Difference between encoder latent and DE latent
            latent_diff = SinkhornLoss(encoder_latent_seq, latent_seq, blur=blur, scaling=scaling, batch_size=None)
            loss = ot_loss + latent_coeff * latent_diff
            epoch_pbar.set_postfix(
                {"Loss": "{:.3f} | OT={:.3f}, Latent_Diff={:.3f}".format(loss, ot_loss, latent_diff)})
            loss.backward()
            optimizer.step()
            loss_list.append([loss.item(), ot_loss.item(), latent_diff.item()])
            # -----
            if cnt % save_iter == 0:
                scNODE_model_list.append(copy.deepcopy(latent_ode_model.state_dict()))
            cnt += 1
    scNODE_model_list.append(copy.deepcopy(latent_ode_model.state_dict()))
    # latent_ODE model prediction
    latent_ode_model.eval()
    recon_obs, first_latent_dist, _, latent_seq = latent_ode_model(train_data, train_tps, num_IWAE_sample, batch_size=None)
    return latent_ode_model, loss_list, recon_obs, first_latent_dist, latent_seq, pretrained_model, scNODE_model_list
