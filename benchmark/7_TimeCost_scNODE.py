'''
Description:
    Compute time cost for our model on the zebrafish (interpolation) data.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''

import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
import itertools
import time

from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, tunedOurPars, sampleOT, splitBySpec
from optim.running import constructscNODEModel, scNODEPredict
from benchmark.BenchmarkUtils import sampleGaussian
from optim.loss_func import SinkhornLoss, MSELoss

# ======================================================
# Load data and pre-processing
print("=" * 70)
data_name = "zebrafish"
print("[ {} ]".format(data_name).center(60))
split_type = "three_interpolation"
print("Split type: {}".format(split_type))
ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name, split_type)
train_tps, test_tps = tpSplitInd(data_name, split_type)
data = ann_data.X

# Convert to torch project
traj_data = [torch.FloatTensor(data[np.where(cell_tps == t)[0], :]) for t in
             range(1, n_tps + 1)]  # (# tps, # cells, # genes)
if cell_types is not None:
    traj_cell_types = [cell_types[np.where(cell_tps == t)[0]] for t in range(1, n_tps + 1)]

all_tps = list(range(n_tps))
train_data, test_data = splitBySpec(traj_data, train_tps, test_tps)
tps = torch.FloatTensor(all_tps)
train_tps = torch.FloatTensor(train_tps)
test_tps = torch.FloatTensor(test_tps)
test_tps_list = [int(t) for t in test_tps]
n_cells = [each.shape[0] for each in traj_data]
print("# tps={}, # genes={}".format(n_tps, n_genes))
print("# cells={}".format(n_cells))
print("Train tps={}".format(train_tps))
print("Test tps={}".format(test_tps))

# ======================================================
# Construct model
pretrain_iters = 200
pretrain_lr = 1e-3
latent_coeff = 1.0
epochs = 1 # 10
iters = 1000
batch_size = 32
lr = 1e-3
act_name = "relu"
n_sim_cells = 2000

latent_dim, drift_latent_size, enc_latent_list, dec_latent_list = tunedOurPars(data_name, split_type)
latent_ode_model = constructscNODEModel(
    n_genes, latent_dim=latent_dim,
    enc_latent_list=enc_latent_list, dec_latent_list=dec_latent_list, drift_latent_size=drift_latent_size,
    latent_enc_act="none", latent_dec_act=act_name, drift_act=act_name,
    ode_method="euler"
)

# Pre-training
latent_encoder = latent_ode_model.latent_encoder
obs_decoder = latent_ode_model.obs_decoder
all_train_data = torch.cat(train_data, dim=0)
all_train_tps = np.concatenate([np.repeat(t, train_data[i].shape[0]) for i, t in enumerate(train_tps)])
if pretrain_iters > 0:
    dim_reduction_params = itertools.chain(*[latent_encoder.parameters(), obs_decoder.parameters()])
    dim_reduction_optimizer = torch.optim.Adam(params=dim_reduction_params, lr=pretrain_lr, betas=(0.95, 0.99))
    dim_reduction_pbar = tqdm(range(pretrain_iters), desc="[ Pre-Training ]")
    latent_encoder.train()
    obs_decoder.train()
    dim_reduction_loss_list = []
    pretrain_start = time.perf_counter()
    for t in dim_reduction_pbar:
        dim_reduction_optimizer.zero_grad()
        latent_mu, latent_std = latent_encoder(all_train_data)
        latent_sample = sampleGaussian(latent_mu, latent_std)
        recon_obs = obs_decoder(latent_sample)
        dim_reduction_loss = MSELoss(all_train_data, recon_obs)
        dim_reduction_pbar.set_postfix({"Loss": "{:.3f}".format(dim_reduction_loss)})
        dim_reduction_loss_list.append(dim_reduction_loss.item())
        dim_reduction_loss.backward()
        dim_reduction_optimizer.step()
    pretrain_end = time.perf_counter()
pretrain_time = pretrain_end - pretrain_start

# Model training
latent_ode_model.latent_encoder = latent_encoder
latent_ode_model.obs_decoder = obs_decoder
num_IWAE_sample = 1
blur = 0.05
scaling = 0.5
loss_list = []
optimizer = torch.optim.Adam(params=latent_ode_model.parameters(), lr=lr, betas=(0.95, 0.99))
latent_ode_model.train()

iter_time = []
iter_metric = []
for e in range(epochs):
    epoch_pbar = tqdm(range(iters), desc="[ Epoch {} ]".format(e + 1))
    for t in epoch_pbar:
        latent_ode_model.train()
        iter_start = time.perf_counter()
        optimizer.zero_grad()
        recon_obs, first_latent_dist, first_time_true_batch, latent_seq = latent_ode_model(
            train_data, train_tps, batch_size=batch_size)
        encoder_latent_seq = [
            latent_ode_model.vaeReconstruct(
                [each[np.random.choice(np.arange(each.shape[0]), size=batch_size, replace=(each.shape[0] < batch_size)),
                 :]]
            )[0][0]
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
        iter_end = time.perf_counter()
        # -----
        latent_ode_model.eval()
        recon_obs, first_latent_dist, _, latent_seq = latent_ode_model(train_data, train_tps, batch_size=None)
        all_recon_obs = scNODEPredict(latent_ode_model, train_data[0], tps, n_cells=n_sim_cells)
        all_recon_obs = [all_recon_obs[:, t, :] for t in range(all_recon_obs.shape[1])]
        t_global_metric = [sampleOT(traj_data[t].detach().numpy(), all_recon_obs[t], sample_n=100, sample_T=10) for t in test_tps_list]
        iter_metric.append(t_global_metric)
        iter_time.append(iter_end - iter_start)

# -----
time_list = np.cumsum(iter_time) + pretrain_time
time_ot = np.asarray(iter_metric).mean(axis=1)
plt.plot(time_list, time_ot)
plt.xlabel("Iteration")
plt.ylabel("Wasserstein")
plt.tight_layout()
plt.show()

# # -----
# print("Saving results...")
# np.save(
#     "../res/time_cost/{}-{}-latent_ODE_OT_pretrain-time_cost.npy".format(data_name, split_type),
#     {
#         "pretrain_time": pretrain_time,
#         "iter_time": iter_time,
#         "iter_metric": iter_metric,
#     },
#     allow_pickle=True
# )
