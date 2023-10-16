'''
Description:
    The main class of our scNODE model.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
from model.layer import LinearNet, LinearVAENet

# ===========================================

class scNODE(nn.Module):
    '''
    scNODE model.
    '''
    def __init__(self, input_dim, latent_dim, output_dim, latent_encoder, diffeq_decoder, obs_decoder):
        '''
        Initialize scNODE model.
        :param input_dim (int): Input space size.
        :param latent_dim (int): Latent space size.
        :param output_dim (int): Output space size.
        :param latent_encoder (LinearVAENet): VAE encoder.
        :param diffeq_decoder (ODE): Differential equation solver.
        :param obs_decoder (LinearNet): VAE decoder.
        '''
        super(scNODE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        # -----
        assert isinstance(latent_encoder, LinearVAENet)
        self.latent_encoder = latent_encoder
        self.diffeq_decoder = diffeq_decoder
        self.obs_decoder = obs_decoder


    def forward(self, data, tps, batch_size=None):
        '''
        scNODE generative process.
        :param data (list of torch.FloatTensor): A list of cell-by-gene expression matrices at every timepoint.
        :param tps (torch.FloatTensor): A list of timepoints to predict.
        :param batch_size (int or None): The batch size (default is None).
        :return: (torch.FloatTensor) Predicted expression at all given timepoints.
                 It has the shape of (batch size, # tps, # genes)
        '''
        first_tp_data = data[0]
        if batch_size is not None:
            cell_idx = np.random.choice(np.arange(first_tp_data.shape[0]), size = batch_size, replace = (first_tp_data.shape[0] < batch_size))
            first_tp_data = first_tp_data[cell_idx, :]
        # Map data at the first timepoint to the VAE latent space
        first_latent_mu, first_latent_std = self.latent_encoder(first_tp_data)
        first_latent_dist = dist.Normal(first_latent_mu, first_latent_std)
        first_latent_sample = self._sampleGaussian(first_latent_mu, first_latent_std)
        # Predict forward with ODE solver in the latent space
        latent_seq = self.diffeq_decoder(first_latent_sample, tps)
        # Convert latent variables (at all timepoints) back to the gene space
        recon_obs = self.obs_decoder(latent_seq) # (batch size, # tps, # genes)
        return recon_obs, first_latent_dist, first_tp_data, latent_seq


    def predict(self, first_tp_data, tps_to_predict, n_cells): # NOTE: use first_tp_data instead of first_tp_dist
        '''
        Predicts at given timepoints.
        :param first_tp_data (torch.FloatTensor): Expression at the first timepoint.
        :param tps_to_predict (torch.FloatTensor): A list of timepoints to predict.
        :param n_cells (int): The number of cells to predict.
        :return: (torch.FloatTensor) Predicted expression at all given timepoints.
                 It has the shape of (# cells, # tps, # genes)
        '''
        first_latent_mean, first_latent_std = self.latent_encoder(first_tp_data)
        repeat_times = (n_cells // first_latent_mean.shape[0]) + 1
        repeat_mean = torch.repeat_interleave(first_latent_mean, repeat_times, dim=0)[:n_cells, :]
        repeat_std = torch.repeat_interleave(first_latent_std, repeat_times, dim=0)[:n_cells, :]
        first_latent_sample = self._sampleGaussian(repeat_mean, repeat_std)
        latent_seq = self.diffeq_decoder(first_latent_sample, tps_to_predict)
        recon_obs = self.obs_decoder(latent_seq)  # (# cells, # tps, # genes)
        return first_latent_sample, latent_seq, recon_obs # NOTE: also return the list of latent variables


    def computeDrift(self, latent_var):
        '''
        Compute drifts corresponding to a list of latent variables.
        :param latent_var (torch.FloatTensor): Latent variable.
        :return: (list of torch.FloatTensor) Drift.
        '''
        # drift_seq = []
        # for i in range(latent_seq.shape[1]):
        #     drift_seq.append()
        # drift_seq = torch.moveaxis(torch.concatenate(drift_seq, dim=0), [0, 1, 2], [1, 0, 2])  # (# cells, # tps, latent size)
        drift = self.diffeq_decoder.net.forwardWithTime(None, latent_var)
        return drift


    def vaeReconstruct(self, data):
        '''
        Compute latent variables corresponding to a list of expression matrices.
        :param data (list of torch.FloatTensor): A list of cell-by-gene expression matrices at every timepoint.
        :return:
            (list of np.ndarray) A list of latent variables obtained from VAE encoding.
            (list of np.ndarray) A list of VAE reconstructed data.
        '''
        latent_list = []
        recons_list = []
        for t, t_data in enumerate(data):
            latent_mean, latent_std = self.latent_encoder(torch.FloatTensor(t_data))
            latent_sample = self._sampleGaussian(latent_mean, latent_std)
            recon_obs = self.obs_decoder(latent_sample)
            latent_list.append(latent_sample)
            recons_list.append(recon_obs)
        return latent_list, recons_list


    def _sampleGaussian(self, mean, std):
        '''
        Sampling with the re-parametric trick.
        '''
        d = dist.normal.Normal(torch.Tensor([0.]), torch.Tensor([1.]))
        r = d.sample(mean.size()).squeeze(-1)
        x = r * std.float() + mean.float()
        return x
