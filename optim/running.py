'''
Description:
    Model model_wrapper of Latent_ODE models.
'''
import torch
from tqdm import tqdm
import numpy as np
import itertools

from model.layer import LinearNet, LinearVAENet, LinearNBNet
from model.diff_solver import ODE, SDE
from model.dynamic_model import scNODE, Latent_ODE_OT_VAE_Aug, Latent_ODE_OT_VAE_NB
from model.dummy_model import DummyModel
from model.FNN_model import FNN
from optim.loss_func import SinkhornLoss, fusedLoss, OTAndLatentMSE, MSELoss, SinkhornLoss4FNN
from benchmark.BenchmarkUtils import sampleGaussian



# =============================================

def constructLatentODEModel(
        n_genes, latent_dim,
        enc_latent_list=None, dec_latent_list=None, drift_latent_size=[64],
        latent_enc_act="none", latent_dec_act="relu", drift_act="relu",
        ode_method="euler"
):
    # Construct latent_ODE_VAE model
    latent_encoder = LinearVAENet(input_dim=n_genes, latent_size_list=enc_latent_list, output_dim=latent_dim,
                                  act_name=latent_enc_act)  # encoder
    obs_decoder = LinearNet(input_dim=latent_dim, latent_size_list=dec_latent_list, output_dim=n_genes,
                            act_name=latent_dec_act)  # decoder
    diffeq_drift_net = LinearNet(input_dim=latent_dim, latent_size_list=drift_latent_size, output_dim=latent_dim,
                                 act_name=drift_act)  # drift network
    diffeq_decoder = ODE(input_dim=latent_dim, drift_net=diffeq_drift_net,
                         ode_method=ode_method)  # differential equation
    latent_ode_model = scNODE(
        input_dim=n_genes,
        latent_dim=latent_dim,
        output_dim=n_genes,
        latent_encoder=latent_encoder,
        diffeq_decoder=diffeq_decoder,
        obs_decoder=obs_decoder
    )
    return latent_ode_model


def constructLatentSDEModel(
        n_genes, latent_dim,
        enc_latent_list=None, dec_latent_list=None, drift_latent_size=[64], diffusion_latent_size=[64],
        latent_enc_act="none", latent_dec_act="relu", drift_act="relu", diffusion_act="relu",
        sde_type="ito", noise_type="diagonal", brownian_size=2
):
    # Construct latent_ODE_VAE model (using SDE for dynamic learning)
    latent_encoder = LinearVAENet(input_dim=n_genes, latent_size_list=enc_latent_list, output_dim=latent_dim,
                                  act_name=latent_enc_act)  # encoder
    obs_decoder = LinearNet(input_dim=latent_dim, latent_size_list=dec_latent_list, output_dim=n_genes,
                            act_name=latent_dec_act)  # decoder
    diffeq_drift_net = LinearNet(input_dim=latent_dim, latent_size_list=drift_latent_size, output_dim=latent_dim,
                                 act_name=drift_act)  # drift network
    diffeq_diffusion_net = LinearNet(
        input_dim=latent_dim, latent_size_list=diffusion_latent_size,
        output_dim=latent_dim if noise_type == "diagonal" else latent_dim * brownian_size,
        act_name=diffusion_act
    )  # diffusion network
    diffeq_decoder = SDE(input_dim=latent_dim, drift_net=diffeq_drift_net, diffusion_net=diffeq_diffusion_net,
                         noise_type=noise_type, sde_type=sde_type, brownian_size=brownian_size)  # differential equation

    latent_ode_model = scNODE(
        input_dim=n_genes,
        latent_dim=latent_dim,
        output_dim=n_genes,
        latent_encoder=latent_encoder,
        diffeq_decoder=diffeq_decoder,
        obs_decoder=obs_decoder
    )
    return latent_ode_model


def constructLatentODEModelWithAug(
        n_genes, latent_dim, augmentation_dim,
        enc_latent_list=None, dec_latent_list=None, drift_latent_size=[64],
        latent_enc_act="none", latent_dec_act="relu", drift_act="relu",
        ode_method="euler"
):
    # Construct latent_ODE_VAE model with augmentation trick
    assert augmentation_dim > 0
    total_latent_dim = latent_dim + augmentation_dim
    latent_encoder = LinearVAENet(input_dim=n_genes, latent_size_list=enc_latent_list, output_dim=latent_dim,
                                  act_name=latent_enc_act)  # encoder
    obs_decoder = LinearNet(input_dim=latent_dim, latent_size_list=dec_latent_list, output_dim=n_genes,
                            act_name=latent_dec_act)  # decoder
    diffeq_drift_net = LinearNet(input_dim=total_latent_dim, latent_size_list=drift_latent_size,
                                 output_dim=total_latent_dim, act_name=drift_act)  # drift network
    diffeq_decoder = ODE(input_dim=total_latent_dim, drift_net=diffeq_drift_net,
                         ode_method=ode_method)  # differential equation
    latent_ode_model = Latent_ODE_OT_VAE_Aug(
        input_dim=n_genes,
        latent_dim=latent_dim,
        augmentation_dim=augmentation_dim,
        output_dim=n_genes,
        latent_encoder=latent_encoder,
        diffeq_decoder=diffeq_decoder,
        obs_decoder=obs_decoder
    )
    return latent_ode_model


def constructLatentODENBModel(
        n_genes, latent_dim,
        enc_latent_list=None, dec_latent_list=None, drift_latent_size=[64],
        latent_enc_act="none", latent_dec_act="relu", drift_act="relu",
        ode_method="euler"
):
    # Construct latent_ODE_VAE model with augmentation trick
    # latent_encoder = LinearVAENet(input_dim=n_genes, latent_size_list=enc_latent_list, output_dim=latent_dim, act_name=latent_enc_act)  # encoder
    latent_encoder = LinearNet(input_dim=n_genes, latent_size_list=enc_latent_list, output_dim=latent_dim, act_name=latent_enc_act)  # encoder
    obs_decoder = LinearNBNet(input_dim=latent_dim, latent_size_list=dec_latent_list, output_dim=n_genes, act_name=latent_dec_act)  # decoder
    diffeq_drift_net = LinearNet(input_dim=latent_dim, latent_size_list=drift_latent_size,
                                 output_dim=latent_dim, act_name=drift_act)  # drift network
    diffeq_decoder = ODE(input_dim=latent_dim, drift_net=diffeq_drift_net,
                         ode_method=ode_method)  # differential equation
    latent_ode_model = Latent_ODE_OT_VAE_NB(
        input_dim=n_genes,
        latent_dim=latent_dim,
        output_dim=n_genes,
        latent_encoder=latent_encoder,
        diffeq_decoder=diffeq_decoder,
        obs_decoder=obs_decoder
    )
    return latent_ode_model


# =============================================

def latentODETrain(train_data, train_tps, latent_ode_model, epochs, iters, batch_size, lr, loss_name):
    lambda_coeff = 0.1 if loss_name == "OT_fused" else 0.0
    if "IWAE" in loss_name:
        num_IWAE_sample = 5
    else:
        num_IWAE_sample = 1
    loss_list = []
    # optimizer = torch.optim.Adam(params=latent_ode_model.parameters(), lr=lr, betas=(0.95, 0.99))
    # optimizer = torch.optim.SGD(params=latent_ode_model.parameters(), lr=lr)
    optimizer = torch.optim.RMSprop(params=latent_ode_model.parameters(), lr=lr)
    latent_ode_model.train()
    for e in range(epochs):
        epoch_pbar = tqdm(range(iters), desc="[ Epoch {} ]".format(e + 1))
        for t in epoch_pbar:
            optimizer.zero_grad()
            recon_obs, first_latent_dist, _, latent_seq = latent_ode_model(train_data, train_tps, num_IWAE_sample, batch_size=batch_size)
            if loss_name == "OT":
                # loss = SinkhornLoss(train_data, recon_obs, blur=0.05, scaling=0.5, batch_size=batch_size)
                loss = SinkhornLoss(train_data, recon_obs, blur=0.05, scaling=0.5, batch_size=200)
                loss_list.append([loss.item()])
            elif loss_name == "OT_fused":
                # ot_loss = SinkhornLoss(train_data, recon_obs, blur=0.05, scaling=0.5, batch_size=batch_size)
                ot_loss = SinkhornLoss(train_data, recon_obs, blur=0.05, scaling=0.5, batch_size=200)
                fused_loss = fusedLoss(recon_obs)
                loss = ot_loss + lambda_coeff * fused_loss
                loss_list.append([loss.item(), ot_loss.item(), fused_loss.item()])
            else:
                raise ValueError("Unknown loss fun name {}!".format(loss_name))
            epoch_pbar.set_postfix({"Loss": "{:.3f}".format(loss)})
            loss.backward()
            optimizer.step()
    # -----
    latent_ode_model.eval()
    recon_obs, first_latent_dist, _, latent_seq = latent_ode_model(train_data, train_tps, num_IWAE_sample, batch_size=None)
    return latent_ode_model, loss_list, recon_obs, first_latent_dist, latent_seq


def latentODETrainWithPreTrain(
        train_data, train_tps, latent_ode_model, latent_coeff, epochs, iters, batch_size, lr,
        pretrain_iters=200, pretrain_lr=1e-3, only_train_de=False
):
    latent_encoder = latent_ode_model.latent_encoder
    obs_decoder = latent_ode_model.obs_decoder
    # Parametric dimensionality reduction training with all training data
    all_train_data = torch.cat(train_data, dim=0)
    all_train_tps = np.concatenate([np.repeat(t, train_data[i].shape[0]) for i, t in enumerate(train_tps)])
    if pretrain_iters > 0:
        dim_reduction_params = itertools.chain(*[latent_encoder.parameters(), obs_decoder.parameters()])
        dim_reduction_optimizer = torch.optim.Adam(params=dim_reduction_params, lr=pretrain_lr, betas=(0.95, 0.99))
        # dim_reduction_optimizer = torch.optim.RMSprop(params=dim_reduction_params, lr=pretrain_lr)
        dim_reduction_pbar = tqdm(range(pretrain_iters), desc="[ Pre-Training ]")
        latent_encoder.train()
        obs_decoder.train()
        dim_reduction_loss_list = []
        for t in dim_reduction_pbar:
            dim_reduction_optimizer.zero_grad()
            latent_mu, latent_std = latent_encoder(all_train_data)
            latent_sample = sampleGaussian(latent_mu, latent_std)
            recon_obs = obs_decoder(latent_sample)
            dim_reduction_loss = MSELoss(all_train_data, recon_obs)
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
        #####################################
        # # VAE reconstruction visualization
        # latent_encoder.eval()
        # obs_decoder.eval()
        # latent_mu, latent_std = latent_encoder(all_train_data)
        # latent_sample = sampleGaussian(latent_mu, latent_std)
        # recon_obs = obs_decoder(latent_sample)
        # from plotting.visualization import umapWithPCA, plotUMAPTimePoint
        # true_umap, umap_model, pca_model = umapWithPCA(all_train_data.detach().numpy(), n_neighbors=50, min_dist=0.1, pca_pcs=50)
        # pred_umap = umap_model.transform(pca_model.transform(recon_obs.detach().numpy()))
        # plotUMAPTimePoint(true_umap, pred_umap, all_train_tps, all_train_tps)
        #####################################
    # Dynamic learning
    latent_ode_model.latent_encoder = latent_encoder
    latent_ode_model.obs_decoder = obs_decoder
    num_IWAE_sample = 1
    blur = 0.05
    scaling = 0.5
    loss_list = []
    if only_train_de:
        print("Only train DE in dynamic learning")
        optimizer = torch.optim.Adam(params=latent_ode_model.diffeq_decoder.net.parameters(), lr=lr, betas=(0.95, 0.99))
    else:
        optimizer = torch.optim.Adam(params=latent_ode_model.parameters(), lr=lr, betas=(0.95, 0.99))
    latent_ode_model.train()
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
            ot_loss = SinkhornLoss(train_data, recon_obs, blur=blur, scaling=scaling, batch_size=200)
            # Difference between encoder latent and DE latent
            latent_diff = SinkhornLoss(encoder_latent_seq, latent_seq, blur=blur, scaling=scaling, batch_size=None)
            loss = ot_loss + latent_coeff * latent_diff
            epoch_pbar.set_postfix(
                {"Loss": "{:.3f} | OT={:.3f}, Latent_Diff={:.3f}".format(loss, ot_loss, latent_diff)})
            loss.backward()
            optimizer.step()
            loss_list.append([loss.item(), ot_loss.item(), latent_diff.item()])
    # latent_ODE model prediction
    latent_ode_model.eval()
    recon_obs, first_latent_dist, _, latent_seq = latent_ode_model(train_data, train_tps, num_IWAE_sample, batch_size=None)
    return latent_ode_model, loss_list, recon_obs, first_latent_dist, latent_seq


def latentODETrainWithLatentMSE(train_data, train_tps, latent_ode_model, epochs, iters, batch_size, lr, mse_coeff, latent_coeff):
    num_IWAE_sample = 1
    loss_list = []
    optimizer = torch.optim.Adam(params=latent_ode_model.parameters(), lr=lr, betas=(0.95, 0.99))
    latent_ode_model.train()
    for e in range(epochs):
        epoch_pbar = tqdm(range(iters), desc="[ Epoch {} ]".format(e + 1))
        for t in epoch_pbar:
            optimizer.zero_grad()
            #
            recon_obs, first_latent_dist, first_time_true_batch, latent_seq = latent_ode_model(train_data, train_tps, num_IWAE_sample, batch_size=batch_size)
            first_time_pred_batch, _ = latent_ode_model.singleReconstruct(first_time_true_batch)
            encoder_latent_seq = [
                latent_ode_model.singleReconstruct(
                    each[np.random.choice(np.arange(each.shape[0]), size = batch_size, replace = (each.shape[0] < batch_size)), :]
                )[1]
                for each in train_data
            ]
            #
            loss = OTAndLatentMSE(
                train_data, recon_obs,
                first_time_true_batch, first_time_pred_batch,
                encoder_latent_seq, latent_seq,
                mse_coeff, latent_coeff,
                blur=0.05, scaling=0.5, batch_size=200)
            loss_list.append([loss.item()])

            epoch_pbar.set_postfix({"Loss": "{:.3f}".format(loss)})
            loss.backward()
            optimizer.step()
    # -----
    latent_ode_model.eval()
    recon_obs, first_latent_dist, _, latent_seq = latent_ode_model(train_data, train_tps, num_IWAE_sample, batch_size=None)
    return latent_ode_model, loss_list, recon_obs, first_latent_dist, latent_seq


def latentODESimulate(latent_ode_model, first_latent_dist, tps, n_cells, batch_size=None):
    # Make predictions
    latent_ode_model.eval()
    if batch_size is None:
        _, all_pred_data = latent_ode_model.predict(first_latent_dist, tps, n_cells=n_cells)
    else:
        _, all_pred_data = latent_ode_model.batchPredict(first_latent_dist, tps, n_cells=n_cells, batch_size=batch_size)
    all_pred_data = all_pred_data.detach().numpy()  # (# trajs, # tps, # genes)
    return all_pred_data


# =============================================

def constructDummyModel():
    dummy_model = DummyModel()
    return dummy_model


def dummySimulate(dummy_model, all_traj_data, train_tps, know_all=False):
    # Make predictions
    all_pred_data = dummy_model.predict(all_traj_data, train_tps, know_all=know_all) # (#tps, #cells, #genes)
    all_pred_data[0] = all_traj_data[0]
    all_pred_data = [each.detach().numpy() for each in all_pred_data]
    return all_pred_data


# =============================================

def constructFNNModel(gene_dim, latent_size=None, act_name="relu"):
    fnn_model = FNN(input_dim=gene_dim, latent_size_list=latent_size, output_dim=gene_dim, act_name=act_name)
    return fnn_model


def FNNTrain(all_traj_data, train_tps, fnn_model, iters, lr, batch_size):
    # Prepare input data and output labels
    n_tps = len(all_traj_data)
    train_x = []
    train_y = []
    for t in range(n_tps-1):
        if (t in train_tps) and ((t+1) in train_tps):
            train_x.append(all_traj_data[t])
            train_y.append(all_traj_data[t+1])
    # Model training
    optimizer = torch.optim.Adam(params=fnn_model.parameters(), lr=lr, betas=(0.95, 0.99))
    fnn_model.train()
    epoch_pbar = tqdm(range(iters), desc="[ FNN Training ]")
    loss_list = []
    for t in epoch_pbar:
        optimizer.zero_grad()
        batch_x = [each[np.random.choice(np.arange(each.shape[0]), batch_size, replace=False),:] for each in train_x]
        pred_y = [fnn_model(x) for x in batch_x]
        loss = SinkhornLoss4FNN(train_y, pred_y, blur=0.05, scaling=0.5, batch_size=200)
        loss_list.append([loss.item()])
        epoch_pbar.set_postfix({"Loss": "{:.3f}".format(loss)})
        loss.backward()
        optimizer.step()
    # -----
    # Model prediction: assume only training data are available
    fnn_model.eval()
    all_recon = [None]
    last_true = all_traj_data[0]
    for t in range(1, n_tps):
        if t in train_tps:
            all_recon.append(fnn_model(last_true))
            last_true = all_traj_data[t]
        else:
            all_recon.append(fnn_model(last_true))
    return fnn_model, loss_list, all_recon


def FNNSimulate(all_traj_data, train_tps, fnn_model):
    n_tps = len(all_traj_data)
    # Model prediction: assume only training data are available
    fnn_model.eval()
    all_recon = [None]
    last_true = all_traj_data[0]
    for t in range(1, n_tps):
        if t in train_tps:
            all_recon.append(fnn_model(last_true))
            last_true = all_traj_data[t]
        else:
            all_recon.append(fnn_model(last_true))
    all_recon[0] = all_traj_data[0]
    all_recon = [each.detach().numpy() for each in all_recon]
    return all_recon