# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/05_train.ipynb.

# %% auto 0
__all__ = ['train', 'train_ae', 'training_regimen']

import copy
import os, sys, json, math, itertools
import pandas as pd, numpy as np
import warnings
from tqdm import tqdm
import torch
from .utils import sample, generate_steps
from .losses import MMD_loss, OT_loss, Density_loss, Local_density_loss

def train(
    model, df, train_tps, optimizer, n_batches=20,
    criterion=MMD_loss(),
    use_cuda=False,

    sample_size=(100, ),
    sample_with_replacement=False,

    local_loss=True,
    global_loss=False,

    apply_losses_in_time=True,

    top_k = 5,
    hinge_value = 0.01,
    use_density_loss=True,
    # use_local_density=False,

    lambda_density = 1.0,

    autoencoder=None,

    use_gaussian:bool=True, 
    add_noise:bool=False, 
    noise_scale:float=0.1,
    
    logger=None,

    use_penalty=False,
    lambda_energy=1.0,

    reverse:bool = False
):
    '''
    MIOFlow training loop
    
    Notes:
        - The argument `model` must have a method `forward` that accepts two arguments
            in its function signature:
                ```python
                model.forward(x, t)
                ```
            where, `x` is the input tensor and `t` is a `torch.Tensor` of time points (float).
        - The training loop is divided in two parts; local (predict t+1 from t), and global (predict the entire trajectory).
                        
    Arguments:
        model (nn.Module): the initialized pytorch ODE model.
        
        df (pd.DataFrame): the DataFrame from which to extract batch data.
        
        groups (list): the list of the numerical groups in the data, e.g. 
            `[1.0, 2.0, 3.0, 4.0, 5.0]`, if the data has five groups.
    
        optimizer (torch.optim): an optimizer initilized with the model's parameters.
        
        n_batches (int): Default to '20', the number of batches from which to randomly sample each consecutive pair
            of groups.
            
        criterion (Callable | nn.Loss): a loss function.
        
        use_cuda (bool): Defaults to `False`. Whether or not to send the model and data to cuda. 

        sample_size (tuple): Defaults to `(100, )`

        sample_with_replacement (bool): Defaults to `False`. Whether or not to sample data points with replacement.
        
        local_loss (bool): Defaults to `True`. Whether or not to use a local loss in the model.
            See notes for more detail.
            
        global_loss (bool): Defaults to `False`. Whether or not to use a global loss in the model.
        
        hold_one_out (bool): Defaults to `False`. Whether or not to randomly hold one time pair
            e.g. t_1 to t_2 out when computing the global loss.

        hold_out (str | int): Defaults to `"random"`. Which time point to hold out when calculating the
            global loss.
            
        apply_losses_in_time (bool): Defaults to `True`. Applies the losses and does back propegation
            as soon as a loss is calculated. See notes for more detail.

        top_k (int): Default to '5'. The k for the k-NN used in the density loss.

        hinge_value (float): Defaults to `0.01`. The hinge value for density loss.

        use_density_loss (bool): Defaults to `True`. Whether or not to add density regularization.

        lambda_density (float): Defaults to `1.0`. The weight for density loss.

        autoencoder (NoneType|nn.Module): Default to 'None'. The full geodesic Autoencoder.

        use_emb (bool): Defaults to `True`. Whether or not to use the embedding model.
        
        use_gae (bool): Defaults to `False`. Whether or not to use the full Geodesic AutoEncoder.

        use_gaussian (bool): Defaults to `True`. Whether to use random or gaussian noise.

        add_noise (bool): Defaults to `False`. Whether or not to add noise.

        noise_scale (float): Defaults to `0.30`. How much to scale the noise by.
        
        logger (NoneType|Logger): Default to 'None'. The logger to record information.

        use_penalty (bool): Defaults to `False`. Whether or not to use $L_e$ during training (norm of the derivative).
        
        lambda_energy (float): Default to '1.0'. The weight of the energy penalty.

        reverse (bool): Whether to train time backwards.
    '''
    noise_fn = torch.randn if use_gaussian else torch.rand
    def noise(data):
        return noise_fn(*data.shape).cuda() if use_cuda else noise_fn(*data.shape)
    # Create the indicies for the steps that should be used
    groups = train_tps
    steps = generate_steps(train_tps)
    if reverse:
        groups = train_tps[::-1]
        steps = generate_steps(groups)
    # Storage variables for losses
    batch_losses = []
    globe_losses = []
    local_losses = {f'{t0}:{t1}':[] for (t0, t1) in generate_steps(groups)}
    density_fn = Density_loss(hinge_value) # if not use_local_density else Local_density_loss()
    if use_cuda:
        model = model.cuda()
    model.train()
    train_bar = tqdm(range(n_batches))
    for batch in train_bar:
        # apply local loss
        if local_loss and not global_loss:
            # for storing the local loss with calling `.item()` so `loss.backward()` can still be used
            batch_loss = []
            steps = generate_steps(groups)
            for step_idx, (t0, t1) in enumerate(steps):
                optimizer.zero_grad()
                #sampling, predicting, and evaluating the loss.
                # sample data
                data_t0 = sample(df, t0, size=sample_size, replace=sample_with_replacement, to_torch=True, use_cuda=use_cuda)
                data_t1 = sample(df, t1, size=sample_size, replace=sample_with_replacement, to_torch=True, use_cuda=use_cuda)
                time = torch.Tensor([t0, t1]).cuda() if use_cuda else torch.Tensor([t0, t1])
                if add_noise:
                    data_t0 += noise(data_t0) * noise_scale
                    data_t1 += noise(data_t1) * noise_scale
                data_t0 = autoencoder.encoder(data_t0)
                data_t1 = autoencoder.encoder(data_t1)
                # prediction
                data_tp = model(data_t0, time)
                # loss between prediction and sample t1
                loss = criterion(data_tp, data_t1)
                recon_loss = loss
                if use_density_loss:                
                    density_loss = density_fn(data_tp, data_t1, top_k=top_k)
                    density_loss = density_loss.to(loss.device)
                    loss += lambda_density * density_loss
                if use_penalty:
                    penalty = sum(model.norm)
                    loss += lambda_energy * penalty
                # apply local loss as we calculate it
                if apply_losses_in_time and local_loss:
                    loss.backward()
                    optimizer.step()
                    model.norm=[]
                # save loss in storage variables 
                local_losses[f'{t0}:{t1}'].append(loss.item())
                batch_loss.append(loss)
            # convert the local losses into a tensor of len(steps)
            batch_loss = torch.Tensor(batch_loss).float()
            if use_cuda:
                batch_loss = batch_loss.cuda()
            if not apply_losses_in_time:
                batch_loss.backward()
                optimizer.step()
            # store average / sum of local losses for training
            ave_local_loss = torch.mean(batch_loss)
            sum_local_loss = torch.sum(batch_loss)            
            batch_losses.append(ave_local_loss.item())
            train_bar.set_postfix({
                "Loss": "{:.3f}".format(loss),
                "Recon": "{:.3f}".format(recon_loss),
                "Density": "{:.3f}".format(density_loss),
            })
        # -----
        # apply global loss
        elif global_loss and not local_loss:
            optimizer.zero_grad()
            #sampling, predicting, and evaluating the loss.
            # sample data
            data_ti = [
                sample(
                    df, group, size=sample_size, replace=sample_with_replacement,
                    to_torch=True, use_cuda=use_cuda
                )
                for group in groups
            ]
            time = torch.Tensor(groups).cuda() if use_cuda else torch.Tensor(groups)

            if add_noise:
                data_ti = [
                    data + noise(data) * noise_scale for data in data_ti
                ]
            data_ti = [autoencoder.encoder(data) for data in data_ti]
            # prediction
            data_tp = model(data_ti[0], time, return_whole_sequence=True)
            loss = sum([
                criterion(data_tp[i], data_ti[i]) 
                for i in range(len(data_tp))
            ])
            recon_loss = loss
            if use_density_loss:
                density_loss = density_fn(data_tp, data_ti, groups, None, top_k)
                density_loss = density_loss.to(loss.device)
                loss += lambda_density * density_loss
            if use_penalty: #TODO
                penalty = sum([model.norm[-(i+1)] for i in range(len(data_tp))])
                loss += lambda_energy * penalty
            loss.backward()
            optimizer.step()
            model.norm=[]
            globe_losses.append(loss.item())
            train_bar.set_postfix({
                "Loss": "{:.3f}".format(loss),
                "Recon": "{:.3f}".format(recon_loss),
                "Density": "{:.3f}".format(density_loss),
            })
        else:
            raise ValueError('A form of loss must be specified.')
    return local_losses, batch_losses, globe_losses



from benchmark.utils import sampleOT
import time as time_timer
def _makeSimulation(df, model, tps, opts, n_sim_cells, n_trajectories=100, n_bins=100):
    use_cuda = opts["use_cuda"]
    autoencoder = opts["autoencoder"]
    recon = opts["recon"]
    n_points = n_sim_cells
    generated = generate_plot_data(
        model, df, tps, n_points, n_trajectories, n_bins, use_cuda=use_cuda,
        autoencoder=autoencoder, recon=recon
    )
    return generated


def train_timer(
        model, df, true_data, train_tps, test_tps, optimizer, recon, tps, pca_model,
        n_batches=20,
        criterion=MMD_loss(),
        use_cuda=False,

        sample_size=(100,),
        sample_with_replacement=False,

        local_loss=True,
        global_loss=False,

        apply_losses_in_time=True,

        top_k=5,
        hinge_value=0.01,
        use_density_loss=True,
        # use_local_density=False,

        lambda_density=1.0,

        autoencoder=None,

        use_gaussian: bool = True,
        add_noise: bool = False,
        noise_scale: float = 0.1,

        logger=None,

        use_penalty=False,
        lambda_energy=1.0,

        reverse: bool = False
):
    noise_fn = torch.randn if use_gaussian else torch.rand

    def noise(data):
        return noise_fn(*data.shape).cuda() if use_cuda else noise_fn(*data.shape)

    # Create the indicies for the steps that should be used
    groups = train_tps
    steps = generate_steps(train_tps)
    if reverse:
        groups = train_tps[::-1]
        steps = generate_steps(groups)
    # Storage variables for losses
    batch_losses = []
    globe_losses = []
    local_losses = {f'{t0}:{t1}': [] for (t0, t1) in generate_steps(groups)}
    density_fn = Density_loss(hinge_value)  # if not use_local_density else Local_density_loss()
    if use_cuda:
        model = model.cuda()

    train_bar = tqdm(range(n_batches))
    iter_time_list = []
    iter_metric_list = []
    for batch in train_bar:
        # apply local loss
        model.train()
        start_time = time_timer.perf_counter()
        if local_loss and not global_loss:
            # for storing the local loss with calling `.item()` so `loss.backward()` can still be used
            batch_loss = []
            steps = generate_steps(groups)
            for step_idx, (t0, t1) in enumerate(steps):
                optimizer.zero_grad()
                # sampling, predicting, and evaluating the loss.
                # sample data
                data_t0 = sample(df, t0, size=sample_size, replace=sample_with_replacement, to_torch=True,
                                 use_cuda=use_cuda)
                data_t1 = sample(df, t1, size=sample_size, replace=sample_with_replacement, to_torch=True,
                                 use_cuda=use_cuda)
                time = torch.Tensor([t0, t1]).cuda() if use_cuda else torch.Tensor([t0, t1])
                if add_noise:
                    data_t0 += noise(data_t0) * noise_scale
                    data_t1 += noise(data_t1) * noise_scale
                data_t0 = autoencoder.encoder(data_t0)
                data_t1 = autoencoder.encoder(data_t1)
                # prediction
                data_tp = model(data_t0, time)
                # loss between prediction and sample t1
                loss = criterion(data_tp, data_t1)
                recon_loss = loss
                if use_density_loss:
                    density_loss = density_fn(data_tp, data_t1, top_k=top_k)
                    density_loss = density_loss.to(loss.device)
                    loss += lambda_density * density_loss
                if use_penalty:
                    penalty = sum(model.norm)
                    loss += lambda_energy * penalty
                # apply local loss as we calculate it
                if apply_losses_in_time and local_loss:
                    loss.backward()
                    optimizer.step()
                    model.norm = []
                # save loss in storage variables
                local_losses[f'{t0}:{t1}'].append(loss.item())
                batch_loss.append(loss)
            # convert the local losses into a tensor of len(steps)
            batch_loss = torch.Tensor(batch_loss).float()
            if use_cuda:
                batch_loss = batch_loss.cuda()
            if not apply_losses_in_time:
                batch_loss.backward()
                optimizer.step()
            # store average / sum of local losses for training
            ave_local_loss = torch.mean(batch_loss)
            sum_local_loss = torch.sum(batch_loss)
            batch_losses.append(ave_local_loss.item())
            train_bar.set_postfix({
                "Loss": "{:.3f}".format(loss),
                "Recon": "{:.3f}".format(recon_loss),
                "Density": "{:.3f}".format(density_loss),
            })
        # -----
        # apply global loss
        elif global_loss and not local_loss:
            optimizer.zero_grad()
            # sampling, predicting, and evaluating the loss.
            # sample data
            data_ti = [
                sample(
                    df, group, size=sample_size, replace=sample_with_replacement,
                    to_torch=True, use_cuda=use_cuda
                )
                for group in groups
            ]
            time = torch.Tensor(groups).cuda() if use_cuda else torch.Tensor(groups)

            if add_noise:
                data_ti = [
                    data + noise(data) * noise_scale for data in data_ti
                ]
            data_ti = [autoencoder.encoder(data) for data in data_ti]
            # prediction
            data_tp = model(data_ti[0], time, return_whole_sequence=True)
            loss = sum([
                criterion(data_tp[i], data_ti[i])
                for i in range(len(data_tp))
            ])
            recon_loss = loss
            if use_density_loss:
                density_loss = density_fn(data_tp, data_ti, groups, None, top_k)
                density_loss = density_loss.to(loss.device)
                loss += lambda_density * density_loss
            if use_penalty:  # TODO
                penalty = sum([model.norm[-(i + 1)] for i in range(len(data_tp))])
                loss += lambda_energy * penalty
            loss.backward()
            optimizer.step()
            model.norm = []
            globe_losses.append(loss.item())
            train_bar.set_postfix({
                "Loss": "{:.3f}".format(loss),
                "Recon": "{:.3f}".format(recon_loss),
                "Density": "{:.3f}".format(density_loss),
            })
        else:
            raise ValueError('A form of loss must be specified.')
        end_time = time_timer.perf_counter()
        iter_time = end_time - start_time
        iter_time_list.append(iter_time)
        # -----
        # Evaluation
        model.eval()
        opts = {"use_cuda": use_cuda, "autoencoder": autoencoder, "recon": recon}
        generated = _makeSimulation(df, model, tps, opts, n_sim_cells=2000, n_trajectories=100, n_bins=100)
        forward_recon_traj = [pca_model.inverse_transform(generated[i, :, :]) for i in range(generated.shape[0])]
        t_global_metric = [
            sampleOT(true_data[t], forward_recon_traj[t], sample_n=100, sample_T=10)
            for t in test_tps]
        iter_metric_list.append(t_global_metric)
    return local_losses, batch_losses, globe_losses, iter_time_list, iter_metric_list


from .utils import generate_steps
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def train_ae(
    model, df, train_tps, optimizer,
    n_epochs=60, criterion=nn.MSELoss(), dist=None, recon = True,
    use_cuda=False, sample_size=(100, ),
    sample_with_replacement=False,
    noise_min_scale=0.09,
    noise_max_scale=0.15,
    
):
    """
    Geodesic Autoencoder training loop.
    
    Notes:
        - We can train only the encoder the fit the geodesic distance (recon=False), or the full geodesic Autoencoder (recon=True),
            i.e. matching the distance and reconstruction of the inputs.
            
    Arguments:
    
        model (nn.Module): the initialized pytorch Geodesic Autoencoder model.
        df (pd.DataFrame): the DataFrame from which to extract batch data.
        train_tps (list): training timepoints
        optimizer (torch.optim): an optimizer initilized with the model's parameters.
        n_epochs (int): Default to '60'. The number of training epochs.
        criterion (torch.nn). Default to 'nn.MSELoss()'. The criterion to minimize.
        dist (NoneType|Class). Default to 'None'. The distance Class with a 'fit(X)' method for a dataset 'X'. Computes the pairwise distances in 'X'.
        recon (bool): Default to 'True'. Whether or not the apply the reconstruction loss.
        use_cuda (bool): Defaults to `False`. Whether or not to send the model and data to cuda.
        sample_size (tuple): Defaults to `(100, )`.
        sample_with_replacement (bool): Defaults to `False`. Whether or not to sample data points with replacement.
        noise_min_scale (float): Default to '0.0'. The minimum noise scale.
        noise_max_scale (float): Default to '1.0'. The maximum noise scale. The true scale is sampled between these two bounds for each epoch.
    
    """
    losses = []
    model.train()
    gae_pbar = tqdm(range(n_epochs), desc="[ GAE Training ]")
    for epoch in gae_pbar:
        # data preparation
        optimizer.zero_grad()
        noise_scale = torch.FloatTensor(1).uniform_(noise_min_scale, noise_max_scale)
        data_ti = torch.vstack([sample(df, group, size=sample_size, replace=sample_with_replacement, to_torch=True, use_cuda=use_cuda) for group in train_tps])
        noise = (noise_scale*torch.randn(data_ti.size())).cuda() if use_cuda else noise_scale*torch.randn(data_ti.size())
        # computation
        encode_dt = model.encoder(data_ti + noise)
        recon_dt = model.decoder(encode_dt) if recon else None
        # reconstruction loss
        loss_recon = criterion(recon_dt,data_ti)
        # distance loss
        dist_geo = dist.fit(data_ti.cpu().numpy())
        dist_geo = torch.from_numpy(dist_geo).float().cuda() if use_cuda else torch.from_numpy(dist_geo).float()
        dist_emb = torch.cdist(encode_dt, encode_dt) ** 2
        loss_dist = criterion(dist_emb, dist_geo)
        loss = loss_recon + loss_dist
        gae_pbar.set_postfix({
            "Loss": "{:.3f}".format(loss),
            "Recon": "{:.3f}".format(loss_recon),
            "Dist": "{:.3f}".format(loss_dist),
        })
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


from .plots import plot_comparision, plot_losses
from .eval import generate_plot_data

def training_regimen(
    n_local_epochs, n_epochs, n_post_local_epochs,
    # BEGIN: train params
    model, df, train_tps, optimizer, n_batches=20,
    criterion=MMD_loss(), use_cuda=False,

    hinge_value=0.01, use_density_loss=True, 

    top_k = 5, lambda_density = 1.0, 
    autoencoder=None,
    sample_size=(100, ), 
    sample_with_replacement=False,
    add_noise=False, noise_scale=0.1, use_gaussian=True,  
    use_penalty=False, lambda_energy=1.0,
    # END: train params
    reverse_schema=True, reverse_n=4
):
    # recon = use_gae and not use_emb
    recon = True
    steps = generate_steps(train_tps)

    local_losses = {f'{t0}:{t1}': [] for (t0, t1) in generate_steps(train_tps)}
    if reverse_schema:
        local_losses = {
            **local_losses,
            **{f'{t0}:{t1}': [] for (t0, t1) in generate_steps(train_tps[::-1])}
        }
    batch_losses = []
    globe_losses = []
    # -----
    local_pbar = tqdm(range(n_local_epochs), desc="[ Local Training ]")
    for epoch in local_pbar:
        reverse = True if reverse_schema and epoch % reverse_n == 0 else False
        l_loss, b_loss, g_loss = train(
            model, df, train_tps, optimizer, n_batches,
            criterion = criterion, use_cuda = use_cuda,
            local_loss=True, global_loss=False, apply_losses_in_time=True,
            hinge_value=hinge_value,
            use_density_loss = use_density_loss,    
            top_k = top_k, lambda_density = lambda_density, 
            autoencoder = autoencoder, sample_size=sample_size,
            sample_with_replacement=sample_with_replacement,
            add_noise=add_noise, noise_scale=noise_scale, use_gaussian=use_gaussian, 
            use_penalty=use_penalty, lambda_energy=lambda_energy, reverse=reverse
        )
        for k, v in l_loss.items():  
            local_losses[k].extend(v)
        batch_losses.extend(b_loss)
        globe_losses.extend(g_loss)
    # -----
    global_pbar = tqdm(range(n_epochs), desc="[ Global Training ]")
    for epoch in global_pbar:
        reverse = True if reverse_schema and epoch % reverse_n == 0 else False
        l_loss, b_loss, g_loss = train(
            model, df, train_tps, optimizer, n_batches,
            criterion = criterion, use_cuda = use_cuda,
            local_loss=False, global_loss=True, apply_losses_in_time=True,
            hinge_value=hinge_value,
            use_density_loss = use_density_loss,       
            top_k = top_k, lambda_density = lambda_density, 
            autoencoder = autoencoder, sample_size=sample_size,
            sample_with_replacement=sample_with_replacement,
            add_noise=add_noise, noise_scale=noise_scale, use_gaussian=use_gaussian,
            use_penalty=use_penalty, lambda_energy=lambda_energy, reverse=reverse
        )
        for k, v in l_loss.items():  
            local_losses[k].extend(v)
        batch_losses.extend(b_loss)
        globe_losses.extend(g_loss)
    # -----
    post_pbar = tqdm(range(n_post_local_epochs), desc="[ Postlocal Training ]")
    for epoch in post_pbar:
        reverse = True if reverse_schema and epoch % reverse_n == 0 else False
        l_loss, b_loss, g_loss = train(
            model, df, train_tps, optimizer, n_batches,
            criterion = criterion, use_cuda = use_cuda,
            local_loss=True, global_loss=False, apply_losses_in_time=True,
            hinge_value=hinge_value,
            use_density_loss = use_density_loss,       
            top_k = top_k, lambda_density = lambda_density,  
            autoencoder = autoencoder, sample_size=sample_size,
            sample_with_replacement=sample_with_replacement,
            add_noise=add_noise, noise_scale=noise_scale, use_gaussian=use_gaussian,
            use_penalty=use_penalty, lambda_energy=lambda_energy, reverse=reverse
        )
        for k, v in l_loss.items():  
            local_losses[k].extend(v)
        batch_losses.extend(b_loss)
        globe_losses.extend(g_loss)
    # -----
    if reverse_schema:
        _temp = {}
        for (t0, t1) in generate_steps([g for g in train_tps]):
            a = f'{t0}:{t1}'
            b = f'{t1}:{t0}'
            _temp[a] = []
            for i, value in enumerate(local_losses[a]):

                if i % reverse_n == 0:
                    _temp[a].append(local_losses[b].pop(0))
                    _temp[a].append(value)
                else:
                    _temp[a].append(value)
        local_losses = _temp

    return local_losses, batch_losses, globe_losses


# =================================================================

def training_regimen_with_timer(
    n_local_epochs, n_epochs, n_post_local_epochs,
    # BEGIN: train params
    model, df, true_data, train_tps, test_tps, tps, optimizer, pca_model, n_batches=20,
    criterion=MMD_loss(), use_cuda=False,

    hinge_value=0.01, use_density_loss=True,

    top_k = 5, lambda_density = 1.0,
    autoencoder=None,
    sample_size=(100, ),
    sample_with_replacement=False,
    add_noise=False, noise_scale=0.1, use_gaussian=True,
    use_penalty=False, lambda_energy=1.0,
    # END: train params
    reverse_schema=True, reverse_n=4
):
    # recon = use_gae and not use_emb
    recon = True
    steps = generate_steps(train_tps)

    local_losses = {f'{t0}:{t1}': [] for (t0, t1) in generate_steps(train_tps)}
    if reverse_schema:
        local_losses = {
            **local_losses,
            **{f'{t0}:{t1}': [] for (t0, t1) in generate_steps(train_tps[::-1])}
        }
    batch_losses = []
    globe_losses = []

    epoch_time = []
    epoch_metric = []
    # -----
    local_pbar = tqdm(range(n_local_epochs), desc="[ Local Training ]")
    for epoch in local_pbar:
        reverse = True if reverse_schema and epoch % reverse_n == 0 else False
        l_loss, b_loss, g_loss, iter_time_list, iter_metric_list = train_timer(
            # model, df, train_tps, optimizer, n_batches,
            model, df, true_data, train_tps, test_tps, optimizer, recon, tps, pca_model, n_batches,
            criterion = criterion, use_cuda = use_cuda,
            local_loss=True, global_loss=False, apply_losses_in_time=True,
            hinge_value=hinge_value,
            use_density_loss = use_density_loss,
            top_k = top_k, lambda_density = lambda_density,
            autoencoder = autoencoder, sample_size=sample_size,
            sample_with_replacement=sample_with_replacement,
            add_noise=add_noise, noise_scale=noise_scale, use_gaussian=use_gaussian,
            use_penalty=use_penalty, lambda_energy=lambda_energy, reverse=reverse
        )
        epoch_time.append(iter_time_list)
        epoch_metric.append(iter_metric_list)
        for k, v in l_loss.items():
            local_losses[k].extend(v)
        batch_losses.extend(b_loss)
        globe_losses.extend(g_loss)
    # -----
    global_pbar = tqdm(range(n_epochs), desc="[ Global Training ]")
    for epoch in global_pbar:
        reverse = True if reverse_schema and epoch % reverse_n == 0 else False
        l_loss, b_loss, g_loss, iter_time_list, iter_metric_list = train_timer(
            # model, df, train_tps, optimizer, n_batches,
            model, df, true_data, train_tps, test_tps, optimizer, recon, tps, pca_model, n_batches,
            criterion = criterion, use_cuda = use_cuda,
            local_loss=False, global_loss=True, apply_losses_in_time=True,
            hinge_value=hinge_value,
            use_density_loss = use_density_loss,
            top_k = top_k, lambda_density = lambda_density,
            autoencoder = autoencoder, sample_size=sample_size,
            sample_with_replacement=sample_with_replacement,
            add_noise=add_noise, noise_scale=noise_scale, use_gaussian=use_gaussian,
            use_penalty=use_penalty, lambda_energy=lambda_energy, reverse=reverse
        )
        epoch_time.append(iter_time_list)
        epoch_metric.append(iter_metric_list)
        for k, v in l_loss.items():
            local_losses[k].extend(v)
        batch_losses.extend(b_loss)
        globe_losses.extend(g_loss)
    # -----
    post_pbar = tqdm(range(n_post_local_epochs), desc="[ Postlocal Training ]")
    for epoch in post_pbar:
        reverse = True if reverse_schema and epoch % reverse_n == 0 else False
        l_loss, b_loss, g_loss, iter_time_list, iter_metric_list = train_timer(
            # model, df, train_tps, optimizer, n_batches,
            model, df, true_data, train_tps, test_tps, optimizer, recon, tps, pca_model, n_batches,
            criterion = criterion, use_cuda = use_cuda,
            local_loss=True, global_loss=False, apply_losses_in_time=True,
            hinge_value=hinge_value,
            use_density_loss = use_density_loss,
            top_k = top_k, lambda_density = lambda_density,
            autoencoder = autoencoder, sample_size=sample_size,
            sample_with_replacement=sample_with_replacement,
            add_noise=add_noise, noise_scale=noise_scale, use_gaussian=use_gaussian,
            use_penalty=use_penalty, lambda_energy=lambda_energy, reverse=reverse
        )
        epoch_time.append(iter_time_list)
        epoch_metric.append(iter_metric_list)
        for k, v in l_loss.items():
            local_losses[k].extend(v)
        batch_losses.extend(b_loss)
        globe_losses.extend(g_loss)
    # -----
    if reverse_schema:
        _temp = {}
        for (t0, t1) in generate_steps([g for g in train_tps]):
            a = f'{t0}:{t1}'
            b = f'{t1}:{t0}'
            _temp[a] = []
            for i, value in enumerate(local_losses[a]):

                if i % reverse_n == 0:
                    _temp[a].append(local_losses[b].pop(0))
                    _temp[a].append(value)
                else:
                    _temp[a].append(value)
        local_losses = _temp

    return local_losses, batch_losses, globe_losses, epoch_time, epoch_metric

# =================================================================

def training_regimen_save_model(
    n_local_epochs, n_epochs, n_post_local_epochs,
    # BEGIN: train params
    model, df, train_tps, optimizer, n_batches=20,
    criterion=MMD_loss(), use_cuda=False,

    hinge_value=0.01, use_density_loss=True,

    top_k = 5, lambda_density = 1.0,
    autoencoder=None,
    sample_size=(100, ),
    sample_with_replacement=False,
    add_noise=False, noise_scale=0.1, use_gaussian=True,
    use_penalty=False, lambda_energy=1.0,
    # END: train params
    reverse_schema=True, reverse_n=4
):
    # recon = use_gae and not use_emb
    recon = True
    steps = generate_steps(train_tps)

    local_losses = {f'{t0}:{t1}': [] for (t0, t1) in generate_steps(train_tps)}
    if reverse_schema:
        local_losses = {
            **local_losses,
            **{f'{t0}:{t1}': [] for (t0, t1) in generate_steps(train_tps[::-1])}
        }
    batch_losses = []
    globe_losses = []
    # [Jiaqi] Save model during training
    local_model_list = []
    global_model_list = []
    # -----
    local_pbar = tqdm(range(n_local_epochs), desc="[ Local Training ]")
    for epoch in local_pbar:
        reverse = True if reverse_schema and epoch % reverse_n == 0 else False
        l_loss, b_loss, g_loss = train(
            model, df, train_tps, optimizer, n_batches,
            criterion = criterion, use_cuda = use_cuda,
            local_loss=True, global_loss=False, apply_losses_in_time=True,
            hinge_value=hinge_value,
            use_density_loss = use_density_loss,
            top_k = top_k, lambda_density = lambda_density,
            autoencoder = autoencoder, sample_size=sample_size,
            sample_with_replacement=sample_with_replacement,
            add_noise=add_noise, noise_scale=noise_scale, use_gaussian=use_gaussian,
            use_penalty=use_penalty, lambda_energy=lambda_energy, reverse=reverse
        )
        for k, v in l_loss.items():
            local_losses[k].extend(v)
        batch_losses.extend(b_loss)
        globe_losses.extend(g_loss)
        # [Jiaqi]
        if epoch % 10 == 0:
            local_model_list.append(copy.deepcopy(model))
    # -----
    global_pbar = tqdm(range(n_epochs), desc="[ Global Training ]")
    for epoch in global_pbar:
        reverse = True if reverse_schema and epoch % reverse_n == 0 else False
        l_loss, b_loss, g_loss = train(
            model, df, train_tps, optimizer, n_batches,
            criterion = criterion, use_cuda = use_cuda,
            local_loss=False, global_loss=True, apply_losses_in_time=True,
            hinge_value=hinge_value,
            use_density_loss = use_density_loss,
            top_k = top_k, lambda_density = lambda_density,
            autoencoder = autoencoder, sample_size=sample_size,
            sample_with_replacement=sample_with_replacement,
            add_noise=add_noise, noise_scale=noise_scale, use_gaussian=use_gaussian,
            use_penalty=use_penalty, lambda_energy=lambda_energy, reverse=reverse
        )
        for k, v in l_loss.items():
            local_losses[k].extend(v)
        batch_losses.extend(b_loss)
        globe_losses.extend(g_loss)
        if epoch % 10 == 0:
            global_model_list.append(copy.deepcopy(model))
    # -----
    post_pbar = tqdm(range(n_post_local_epochs), desc="[ Postlocal Training ]")
    for epoch in post_pbar:
        reverse = True if reverse_schema and epoch % reverse_n == 0 else False
        l_loss, b_loss, g_loss = train(
            model, df, train_tps, optimizer, n_batches,
            criterion = criterion, use_cuda = use_cuda,
            local_loss=True, global_loss=False, apply_losses_in_time=True,
            hinge_value=hinge_value,
            use_density_loss = use_density_loss,
            top_k = top_k, lambda_density = lambda_density,
            autoencoder = autoencoder, sample_size=sample_size,
            sample_with_replacement=sample_with_replacement,
            add_noise=add_noise, noise_scale=noise_scale, use_gaussian=use_gaussian,
            use_penalty=use_penalty, lambda_energy=lambda_energy, reverse=reverse
        )
        for k, v in l_loss.items():
            local_losses[k].extend(v)
        batch_losses.extend(b_loss)
        globe_losses.extend(g_loss)
    # -----
    if reverse_schema:
        _temp = {}
        for (t0, t1) in generate_steps([g for g in train_tps]):
            a = f'{t0}:{t1}'
            b = f'{t1}:{t0}'
            _temp[a] = []
            for i, value in enumerate(local_losses[a]):

                if i % reverse_n == 0:
                    _temp[a].append(local_losses[b].pop(0))
                    _temp[a].append(value)
                else:
                    _temp[a].append(value)
        local_losses = _temp

    return local_losses, batch_losses, globe_losses, local_model_list, global_model_list