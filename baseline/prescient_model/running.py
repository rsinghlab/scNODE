'''
Description:
    Wrapper for running PRESCIENT model in our benchmarks
    Codes are adopted from PRESCIENT source codes.

Reference:
    https://github.com/gifford-lab/prescient/blob/master/prescient/commands/train_model.py
'''

import torch
import numpy as np
import pandas as pd
import pickle as pkl
import sklearn
import umap

import os
import copy
import argparse
import baseline.prescient_model.prescient.train as train
import baseline.prescient_model.prescient.simulate as traj
from baseline.prescient_model.prescient.train.model import *


# --------------------------------------------------

def init_config(args):
    config = SimpleNamespace(
        seed=args.seed,
        timestamp=strftime("%a, %d %b %Y %H:%M:%S", localtime()),

        # data parameters
        data_path=args.data_path,
        weight=args.weight,

        # model parameters
        activation=args.activation,
        layers=args.layers,
        k_dim=args.k_dim,

        # pretraining parameters
        pretrain_burnin=50,
        pretrain_sd=0.1,
        pretrain_lr=1e-9,
        pretrain_epochs=args.pretrain_epochs,

        # training parameters
        train_dt=args.train_dt,
        train_sd=args.train_sd,
        train_batch_size=args.train_batch,
        ns=2000,
        train_burnin=100,
        train_tau=args.train_tau,
        train_epochs=args.train_epochs,
        train_lr=args.train_lr,
        train_clip=args.train_clip,
        save=args.save,

        # loss parameters
        sinkhorn_scaling=0.7,
        sinkhorn_blur=0.1,

        # file parameters
        out_dir=args.out_dir,
        out_name=args.out_dir.split('/')[-1],
        pretrain_pt=os.path.join(args.out_dir, 'pretrain.pt'),
        train_pt=os.path.join(args.out_dir, 'train.{}.pt'),
        train_log=os.path.join(args.out_dir, 'train.log'),
        done_log=os.path.join(args.out_dir, 'done.log'),
        config_pt=os.path.join(args.out_dir, 'config.pt'),
    )

    config.train_t = []
    config.test_t = []

    # if not os.path.exists(args.out_dir):
    #     print('Making directory at {}'.format(args.out_dir))
    #     os.makedirs(args.out_dir)
    # else:
    #     print('Directory exists at {}'.format(args.out_dir))
    return config


def load_data(args):
    return torch.load(args.data_path)


def train_init(args):
    a = copy.copy(args)
    data_pt = args.data_pt
    x = data_pt["xp"]
    y = data_pt["y"]
    weight = data_pt["w"]
    if args.weight_name != None:
        a.weight = args.weight_name

    # out directory
    a.train_sd = args.train_sd
    a.train_lr = args.train_lr
    a.train_clip = args.train_clip
    a.train_batch = args.train_batch
    name = (
        "{weight}-"
        "{activation}_{layers}_{k_dim}-"
        "{train_tau}-"
        "{train_sd}-"
        "{train_lr}-"
        "{train_clip}-"
        "{train_batch}"
    ).format(**a.__dict__)
    name = name + "-{}".format(a.timestamp)

    a.out_dir = os.path.join(args.out_dir, name, 'seed_{}'.format(a.seed))
    config = init_config(a)

    config.x_dim = x[0].shape[-1]
    config.t = y[-1] - y[0]

    # config.start_t = y[0]
    # config.train_t = y[1:]
    config.start_t = 0
    config.train_t = a.train_t
    y_start = y[config.start_t]
    y_ = [y_ for y_ in y if y_ > y_start]

    w_ = weight[config.start_t]
    w = {(y_start, yy): torch.from_numpy(np.exp((yy - y_start) * w_)) for yy in y_}

    return x, y, w, config


def trainModel(args):
    model, best_state_dict, config, loss_list = train.run(args, train_init)
    return model, best_state_dict, config, loss_list

def trainModelWithTimer(args):
    pretrain_time, iter_time, iter_metric = train.run_with_timer(args, train_init)
    return pretrain_time, iter_time, iter_metric


def makeSimulation(args, config):
    # load data
    # data_pt = torch.load(args.data_path)
    data_pt = args.data_pt
    expr = data_pt["data"]
    pca = data_pt["pca"]
    xp = pca.transform(expr)
    # xp = expr

    # torch device
    if args.gpu != None:
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')

    # load model
    net = AutoGenerator(config)
    net.load_state_dict(args.best_model_state['model_state_dict'])
    net.to(device)

    # Either use assigned number of steps or calculate steps, both using the stepsize used for training
    if args.num_steps == None:
        num_steps = int(np.round(data_pt["y"] / config.train_dt))
    else:
        num_steps = int(args.num_steps)

    # simulate forward
    num_cells = min(args.num_cells, xp.shape[0])
    out = traj.simulate(
        xp, data_pt["tps"], data_pt["celltype"], data_pt["w"], net, config, args.num_sims, num_cells,
        num_steps, device, args.tp_subset, args.celltype_subset)
    return out[0]


# --------------------------------------------------

def create_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--gpu', default=-1, type=int,
                        help="Designate GPU number as an integer (compatible with CUDA).")
    parser.add_argument('--out_dir', default='./mammalian/', help="Directory for storing training output.")
    parser.add_argument('--seed', type=int, default=2, help="Set seed for training process.")
    # -- data options
    parser.add_argument('-i', '--data_path', default="mammalian", help="Input PRESCIENT data torch file.")
    parser.add_argument('--weight_name', default="default",
                        help="Designate descriptive name of growth parameters for filename.")
    # -- model options
    parser.add_argument('--loss', default='euclidean', help="Designate distance function for loss.")
    parser.add_argument('--k_dim', default=100, type=int, help="Designate hidden units of NN.")
    parser.add_argument('--activation', default='softplus', help="Designate activation function for layers of NN.")
    parser.add_argument('--layers', default=2, type=int,
                        help="Choose number of layers for neural network parameterizing the potential function.")
    # -- pretrain options
    parser.add_argument('--pretrain_epochs', default=5, type=int,
                        help="Number of epochs for pretraining with contrastive divergence.")
    # -- train options
    parser.add_argument('--train_epochs', default=100, type=int, help="Number of epochs for training.")
    parser.add_argument('--train_lr', default=0.1, type=float, help="Learning rate for Adam optimizer during training.")
    parser.add_argument('--train_dt', default=0.1, type=float, help="Timestep for simulations during training.")
    parser.add_argument('--train_sd', default=0.5, type=float,
                        help="Standard deviation of Gaussian noise for simulation steps.")
    parser.add_argument('--train_tau', default=1e-6, type=float, help="Tau hyperparameter of PRESCIENT.")
    parser.add_argument('--train_batch', default=0.1, type=float, help="Batch size for training.")
    parser.add_argument('--train_clip', default=0.25, type=float, help="Gradient clipping threshold for training.")
    parser.add_argument('--save', default=100, type=int, help="Save model every n epochs.")
    # -- run options
    parser.add_argument('--pretrain', type=bool, default=True, help="If True, pretraining will run.")
    parser.add_argument('--train', type=bool, default=True,
                        help="If True, training will run with existing pretraining torch file.")
    parser.add_argument('--config')
    return parser


def create_simulate_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data_path", default="mammalian", required=False,
                        help="Path to PRESCIENT data file stored as a torch pt.")
    parser.add_argument("--model_path",
                        default="./mammalian/default-softplus_2_100-1e-06-0.5-0.1-0.25-0.1-20230612215839/",
                        required=False, help="Path to directory containing PRESCIENT model for simulation.")
    parser.add_argument("--seed", default=2, required=False,
                        help="Choose the seed of the trained model to use for simulations.")
    parser.add_argument("--epoch", type=str, required=False,
                        help="Choose which epoch of the model to use for simulations.")
    parser.add_argument("--num_sims", default=1, help="Number of simulations to run.")
    parser.add_argument("--num_cells", default=2000, help="Number of cells per simulation.")
    parser.add_argument("--num_steps", default=120, required=False,
                        help="Define number of forward steps of size dt to take.")
    # parser.add_argument("--num_steps", default=None, required=False,
    #                     help="Define number of forward steps of size dt to take.")
    parser.add_argument("--gpu", default=None, required=False, help="If available, assign GPU device number.")
    parser.add_argument("--celltype_subset", default=None, required=False,
                        help="Randomly sample initial cells from a particular celltype defined in metadata.")
    parser.add_argument("--tp_subset", default=None, required=False,
                        help="Randomly sample initial cells from a particular timepoint.")
    parser.add_argument("-o", "--out_path", required=False, default="../../Prediction/PRESCIENT/",
                        help="Path to output directory.")
    return parser

# --------------------------------------------------

def prescientTrain(data_dict, data_name, out_dir, k_dim, layers, train_epochs, train_lr, train_sd, train_tau, train_clip, train_t, timestamp):
    print("PRESCIENT model training...")
    train_parser = create_train_parser()
    args = train_parser.parse_args()
    args.data_path = data_name
    args.out_dir = out_dir
    args.k_dim = k_dim
    args.data_pt = data_dict
    args.train_t = train_t
    args.timestamp = timestamp
    args.layers = layers # num of hidden layers
    args.train_epochs = train_epochs
    args.train_lr = train_lr
    args.train_sd = train_sd # Gaussian sd for simulation
    args.train_tau = train_tau
    args.train_clip = train_clip # gradient clipping
    model, best_state_dict, config, loss_list = trainModel(args)
    return model, best_state_dict, config, loss_list


def prescientSimulate(data_dict, data_name, best_model_state, num_cells, num_steps, config):
    sim_parser = create_simulate_parser()
    args = sim_parser.parse_args()
    args.data_pt = data_dict
    args.data_path = data_name
    args.best_model_state = best_model_state
    args.num_cells = num_cells
    args.num_steps = num_steps
    # -----
    sim_data = makeSimulation(args, config)
    return sim_data


def prescientTrainWithTimer(
        data_dict, data_name, out_dir, k_dim, layers, train_epochs, train_lr, train_sd, train_tau, train_clip,
        train_t, timestamp, num_cells, num_steps, test_tps):
    print("PRESCIENT model training...")
    train_parser = create_train_parser()
    args = train_parser.parse_args()
    args.data_path = data_name
    args.out_dir = out_dir
    args.k_dim = k_dim
    args.data_pt = data_dict
    args.train_t = train_t
    args.timestamp = timestamp
    args.layers = layers # num of hidden layers
    args.train_epochs = train_epochs
    args.train_lr = train_lr
    args.train_sd = train_sd # Gaussian sd for simulation
    args.train_tau = train_tau
    args.train_clip = train_clip # gradient clipping
    args.num_cells = num_cells
    args.num_steps = num_steps
    args.test_tps = test_tps
    pretrain_time, iter_time, iter_metric = trainModelWithTimer(args)
    return pretrain_time, iter_time, iter_metric

