'''
Description:
    Comapre time costs.
'''

import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
import itertools
import time

from plotting.__init__ import *

# ======================================================
# Load data and pre-processing
print("=" * 70)
data_name = "zebrafish"  # zebrafish, mammalian, drosophila, wot, pancreatic, embryoid
print("[ {} ]".format(data_name).center(60))
split_type = "three_forecasting"  # three_interpolation, three_forecasting, one_interpolation, one_forecasting
print("Split type: {}".format(split_type))

our_res = np.load("../res/time_cost/{}-{}-latent_ODE_OT_pretrain-time_cost.npy".format(data_name, split_type), allow_pickle=True).item()
prescient_res = np.load("../res/time_cost/{}-{}-PRESCIENT-time_cost.npy".format(data_name, split_type), allow_pickle=True).item()

our_pretrain_time = our_res["pretrain_time"]
our_iter_time = our_res["iter_time"]
our_iter_metric = our_res["iter_metric"]

prescient_pretrain_time = prescient_res["pretrain_time"]
prescient_iter_time = prescient_res["iter_time"]
prescient_iter_metric = prescient_res["iter_metric"]

our_time = np.cumsum(our_iter_time) + our_pretrain_time
our_ot = np.asarray(our_iter_metric).mean(axis=1)
prescient_time = np.cumsum(prescient_iter_time) + prescient_pretrain_time
prescient_ot = np.asarray(prescient_iter_metric).mean(axis=1)

# -----
plt.figure(figsize=(8, 4))
plt.plot(our_time, our_ot, lw=2, label="our", color=Vivid_10.mpl_colors[0])
plt.plot(prescient_time, prescient_ot, lw=2, label="PRESCIENT", color=Vivid_10.mpl_colors[1])
plt.xlim(np.round(our_pretrain_time), np.round(our_time[-1] + 10))
plt.xlabel("Time Cost (sec)")
plt.ylabel("OT (testing)")
plt.legend()
plt.tight_layout()
plt.show()
