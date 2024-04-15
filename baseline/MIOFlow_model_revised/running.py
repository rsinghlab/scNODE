'''
Description:
    Run MIOFlow model.

Reference:
    [1] https://github.com/KrishnaswamyLab/MIOFlow/tree/main
'''
from MIOFlow.utils import generate_steps, set_seeds, config_criterion
from MIOFlow.models import make_model, Autoencoder
from MIOFlow.plots import plot_comparision, plot_losses
from MIOFlow.train import train_ae, training_regimen, training_regimen_with_timer
from MIOFlow.constants import ROOT_DIR, DATA_DIR, NTBK_DIR, IMGS_DIR, RES_DIR
from MIOFlow.datasets import (
    make_diamonds, make_swiss_roll, make_tree, make_eb_data,
    make_dyngen_data
)
from MIOFlow.geo import setup_distance
from MIOFlow.exp import setup_exp
from MIOFlow.eval import generate_plot_data

import os, pandas as pd, numpy as np, \
    seaborn as sns, matplotlib as mpl, matplotlib.pyplot as plt, \
    torch, torch.nn as nn

import time


def trainModel(
        df, train_tps, test_tps, n_genes, n_epochs_emb=1000, samples_size_emb = (30,), gae_embedded_dim = 50,
        encoder_layers = [50, 50, 50], layers = [50, 50, 50],
        batch_size=32, n_local_epochs=40, n_global_epochs=40, n_post_local_epochs=0,
        lambda_density=35, pca_dims=50
):
    use_cuda = False
    hold_out = test_tps
    groups = train_tps
    # =================================
    # GAE hyperparameter
    distance_type = 'alpha_decay' # gaussian, alpha_decay
    rbf_length_scale = 0.001 # 0.1
    knn = 5
    t_max = 5
    dist = setup_distance(distance_type, rbf_length_scale=rbf_length_scale, knn=knn, t_max=t_max)
    # Construct GAE
    gae = Autoencoder(
        encoder_layers=encoder_layers,
        decoder_layers=encoder_layers[::-1],
        activation='ReLU', use_cuda=use_cuda
    ) # [model_features, hidden layer, gae_embedded_dim]
    optimizer = torch.optim.AdamW(gae.parameters())
    # GAE training
    recon = True # use reconstruction loss
    gae_losses = train_ae(
        gae, df, train_tps, optimizer,
        n_epochs=n_epochs_emb, sample_size=samples_size_emb,
        noise_min_scale=0.09, noise_max_scale=0.15,
        dist=dist, recon=recon, use_cuda=use_cuda
    )
    autoencoder = gae
    # plt.title("GAE Loss Curve")
    # plt.plot(gae_losses)
    # plt.show()
    # =================================
    # ODE hyperparameters
    use_density_loss = True
    activation = 'CELU' # LeakyReLU, ReLU, CELU
    sde_scales = (len(train_tps) + len(test_tps)) * [0.2]
    # Construct ODE model
    model_features = gae_embedded_dim
    model = make_model(
        model_features, layers,
        activation=activation, scales=sde_scales, use_cuda=use_cuda
    )
    # ODE training
    sample_size = (batch_size,)
    n_local_epochs = n_local_epochs
    n_epochs = n_global_epochs
    n_post_local_epochs = n_post_local_epochs
    reverse_schema = True
    reverse_n = 2 # each reverse_n epoch
    criterion_name = 'ot'
    criterion = config_criterion(criterion_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    local_losses, batch_losses, globe_losses = training_regimen(
        # local, global, local train structure
        n_local_epochs=n_local_epochs,
        n_epochs=n_epochs,
        n_post_local_epochs=n_post_local_epochs,
        # BEGIN: train params
        model=model, df=df, train_tps=train_tps, optimizer=optimizer,
        criterion=criterion, use_cuda=use_cuda,
        use_density_loss=use_density_loss,
        lambda_density=lambda_density,
        autoencoder=autoencoder,
        sample_size=sample_size,
        reverse_schema=reverse_schema, reverse_n=reverse_n,
        # END: train params
    )
    opts = {"use_cuda": use_cuda, "autoencoder": autoencoder, "recon": recon}
    return model, gae_losses, local_losses, batch_losses, globe_losses, opts


def trainModelWithTimer(
        df, true_data, train_tps, test_tps, tps, pca_model, n_genes, n_epochs_emb=1000, samples_size_emb = (30,), gae_embedded_dim = 50,
        encoder_layers = [50, 50, 50], layers = [50, 50, 50],
        batch_size=32, n_local_epochs=40, n_global_epochs=40, n_post_local_epochs=0,
        lambda_density=35, pca_dims=50
):
    use_cuda = False
    hold_out = test_tps
    groups = train_tps
    # =================================
    # GAE hyperparameter
    distance_type = 'alpha_decay' # gaussian, alpha_decay
    rbf_length_scale = 0.001 # 0.1
    knn = 5
    t_max = 5
    dist = setup_distance(distance_type, rbf_length_scale=rbf_length_scale, knn=knn, t_max=t_max)
    # Construct GAE
    gae = Autoencoder(
        encoder_layers=encoder_layers,
        decoder_layers=encoder_layers[::-1],
        activation='ReLU', use_cuda=use_cuda
    ) # [model_features, hidden layer, gae_embedded_dim]
    optimizer = torch.optim.AdamW(gae.parameters())
    # GAE training
    recon = True # use reconstruction loss
    pretrain_time = 0.0
    pretrain_start = time.perf_counter()
    gae_losses = train_ae(
        gae, df, train_tps, optimizer,
        n_epochs=n_epochs_emb, sample_size=samples_size_emb,
        noise_min_scale=0.09, noise_max_scale=0.15,
        dist=dist, recon=recon, use_cuda=use_cuda
    )
    pretrain_end = time.perf_counter()
    pretrain_time = pretrain_end - pretrain_start
    autoencoder = gae
    # plt.title("GAE Loss Curve")
    # plt.plot(gae_losses)
    # plt.show()
    # =================================
    # ODE hyperparameters
    use_density_loss = True
    activation = 'CELU' # LeakyReLU, ReLU, CELU
    sde_scales = (len(train_tps) + len(test_tps)) * [0.2] # TODO: SDE scales
    # Construct ODE model
    model_features = gae_embedded_dim
    model = make_model(
        model_features, layers,
        activation=activation, scales=sde_scales, use_cuda=use_cuda
    )
    # ODE training
    sample_size = (batch_size,)
    n_local_epochs = n_local_epochs
    n_epochs = n_global_epochs
    n_post_local_epochs = n_post_local_epochs
    reverse_schema = True
    reverse_n = 2 # each reverse_n epoch
    criterion_name = 'ot'
    criterion = config_criterion(criterion_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    local_losses, batch_losses, globe_losses, epoch_time, epoch_metric = training_regimen_with_timer(
        # local, global, local train structure
        n_local_epochs=n_local_epochs,
        n_epochs=n_epochs,
        n_post_local_epochs=n_post_local_epochs,
        # BEGIN: train params
        model=model, df=df, true_data=true_data, train_tps=train_tps, test_tps=test_tps, tps=tps, optimizer=optimizer, pca_model=pca_model,
        criterion=criterion, use_cuda=use_cuda,
        use_density_loss=use_density_loss,
        lambda_density=lambda_density,
        autoencoder=autoencoder,
        sample_size=sample_size,
        reverse_schema=reverse_schema, reverse_n=reverse_n,
        # END: train params
    )
    opts = {"use_cuda": use_cuda, "autoencoder": autoencoder, "recon": recon}
    return model, gae_losses, local_losses, batch_losses, globe_losses, opts, pretrain_time, epoch_time, epoch_metric



def makeSimulation(df, model, tps, opts, n_sim_cells, n_trajectories=100, n_bins=100):
    use_cuda = opts["use_cuda"]
    autoencoder = opts["autoencoder"]
    recon = opts["recon"]
    n_points = n_sim_cells
    generated = generate_plot_data(
        model, df, tps, n_points, n_trajectories, n_bins, use_cuda=use_cuda,
        autoencoder=autoencoder, recon=recon
    )
    return generated


