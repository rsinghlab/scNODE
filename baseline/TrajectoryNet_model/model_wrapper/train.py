'''
Description:
    A wrapper of TrajectoryNet training function.

Reference:
    [1] https://github.com/KrishnaswamyLab/TrajectoryNet/blob/master/TrajectoryNet/main.py
'''

import os
import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from baseline.TrajectoryNet_model.TrajectoryNet.lib import utils
from baseline.TrajectoryNet_model.TrajectoryNet.train_misc import (
    count_nfe,
    count_parameters,
    count_total_time,
    create_regularization_fns,
    get_regularization,
    append_regularization_to_log,
    build_model_tabular,
)

from baseline.TrajectoryNet_model.TrajectoryNet import dataset
from baseline.TrajectoryNet_model.TrajectoryNet.parse import parser


# ==================================

def compute_loss(device, args, model, growth_model, full_data):
    """
    Compute loss by integrating backwards from the last time step
    At each time step integrate back one time step, and concatenate that
    to samples of the empirical distribution at that previous timestep
    repeating over and over to calculate the likelihood of samples in
    later timepoints iteratively, making sure that the ODE is evaluated
    at every time step to calculate those later points.

    The growth model is a single model of time independent cell growth /
    death rate defined as a variation from uniform.
    """

    # Backward pass accumulating losses, previous state and deltas
    deltas = []
    zs = []
    z = None
    interp_loss = 0.0
    for i, (itp, tp) in enumerate(zip(args.int_tps[::-1], args.timepoints[::-1])):
        # tp counts down from last
        integration_times = torch.tensor([itp - args.time_scale, itp])
        integration_times = integration_times.type(torch.float32).to(device)
        # integration_times.requires_grad = True

        # load data and add noise
        idx = args.data.sample_index(args.batch_size, tp)
        x = args.data.get_data()[idx]
        if args.training_noise > 0.0:
            x += np.random.randn(*x.shape) * args.training_noise
        x = torch.from_numpy(x).type(torch.float32).to(device)

        if i > 0:
            x = torch.cat((z, x))
            zs.append(z)
        zero = torch.zeros(x.shape[0], 1).to(x)

        # transform to previous timepoint
        z, delta_logp = model(x, zero, integration_times=integration_times)
        deltas.append(delta_logp)

        # Straightline regularization
        # Integrate to random point at time t and assert close to (1 - t) * end + t * start
        if args.interp_reg:
            t = np.random.rand()
            int_t = torch.tensor([itp - t * args.time_scale, itp])
            int_t = int_t.type(torch.float32).to(device)
            int_x = model(x, integration_times=int_t)
            int_x = int_x.detach()
            actual_int_x = x * (1 - t) + z * t
            interp_loss += F.mse_loss(int_x, actual_int_x)
    if args.interp_reg:
        print("interp_loss", interp_loss)

    logpz = args.data.base_density()(z)

    # build growth rates
    if args.use_growth:
        growthrates = [torch.ones_like(logpz)]
        for z_state, tp in zip(zs[::-1], args.timepoints[:-1]):
            # Full state includes time parameter to growth_model
            time_state = tp * torch.ones(z_state.shape[0], 1).to(z_state)
            full_state = torch.cat([z_state, time_state], 1)
            growthrates.append(growth_model(full_state))

    # Accumulate losses
    losses = []
    logps = [logpz]
    for i, delta_logp in enumerate(deltas[::-1]):
        logpx = logps[-1] - delta_logp
        if args.use_growth:
            logpx += torch.log(torch.clamp(growthrates[i], 1e-4, 1e4))
        logps.append(logpx[: -args.batch_size])
        losses.append(-torch.mean(logpx[-args.batch_size :]))
    losses = torch.stack(losses)
    weights = torch.ones_like(losses).to(logpx)
    if args.leaveout_timepoint >= 0:
        weights[args.leaveout_timepoint] = 0
    losses = torch.mean(losses * weights)

    # Direction regularization
    if args.vecint:
        similarity_loss = 0
        for i, (itp, tp) in enumerate(zip(args.int_tps, args.timepoints)):
            itp = torch.tensor(itp).type(torch.float32).to(device)
            idx = args.data.sample_index(args.batch_size, tp)
            x = args.data.get_data()[idx]
            v = args.data.get_velocity()[idx]
            x = torch.from_numpy(x).type(torch.float32).to(device)
            v = torch.from_numpy(v).type(torch.float32).to(device)
            x += torch.randn_like(x) * 0.1
            # Only penalizes at the time / place of visible samples
            direction = -model.chain[0].odefunc.odefunc.diffeq(itp, x)
            if args.use_magnitude:
                similarity_loss += torch.mean(F.mse_loss(direction, v))
            else:
                similarity_loss -= torch.mean(F.cosine_similarity(direction, v))
        # logger.info(similarity_loss)
        losses += similarity_loss * args.vecint

    # Density regularization
    if args.top_k_reg > 0:
        density_loss = 0
        tp_z_map = dict(zip(args.timepoints[:-1], zs[::-1]))
        if args.leaveout_timepoint not in tp_z_map:
            idx = args.data.sample_index(args.batch_size, tp)
            x = args.data.get_data()[idx]
            if args.training_noise > 0.0:
                x += np.random.randn(*x.shape) * args.training_noise
            x = torch.from_numpy(x).type(torch.float32).to(device)
            t = np.random.rand()
            int_t = torch.tensor([itp - t * args.time_scale, itp])
            int_t = int_t.type(torch.float32).to(device)
            int_x = model(x, integration_times=int_t)
            samples_05 = int_x
        else:
            # If we are leaving out a timepoint the regularize there
            samples_05 = tp_z_map[args.leaveout_timepoint]

        # Calculate distance to 5 closest neighbors
        # WARNING: This currently fails in the backward pass with cuda on pytorch < 1.4.0
        #          works on CPU. Fixed in pytorch 1.5.0
        # RuntimeError: CUDA error: invalid configuration argument
        # The workaround is to run on cpu on pytorch <= 1.4.0 or upgrade
        cdist = torch.cdist(samples_05, full_data)
        values, _ = torch.topk(cdist, 5, dim=1, largest=False, sorted=False)
        # Hinge loss
        hinge_value = 0.1
        values -= hinge_value
        values[values < 0] = 0
        density_loss = torch.mean(values)
        # print("Density Loss", density_loss.item())
        losses += density_loss * args.top_k_reg
    losses += interp_loss
    return losses


def train_func(
    device, args, model, growth_model, regularization_coeffs, regularization_fns
):
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    time_meter = utils.RunningAverageMeter(0.93)
    loss_meter = utils.RunningAverageMeter(0.93)
    nfef_meter = utils.RunningAverageMeter(0.93)
    nfeb_meter = utils.RunningAverageMeter(0.93)
    tt_meter = utils.RunningAverageMeter(0.93)

    full_data = (
        torch.from_numpy(
            args.data.get_data()[args.data.get_times() != args.leaveout_timepoint]
        )
        .type(torch.float32)
        .to(device)
    )

    best_loss = float("inf")
    # if args.use_growth:
    #     growth_model.eval()
    end = time.time()
    for itr in range(1, args.niters + 1):
        model.train()
        optimizer.zero_grad()

        # Train
        # if args.spectral_norm:
        #     spectral_norm_power_iteration(model, 1)

        loss = compute_loss(device, args, model, growth_model, full_data)
        loss_meter.update(loss.item())

        if len(regularization_coeffs) > 0:
            # Only regularize on the last timepoint
            reg_states = get_regularization(model, regularization_coeffs)
            reg_loss = sum(
                reg_state * coeff
                for reg_state, coeff in zip(reg_states, regularization_coeffs)
                if coeff != 0
            )
            loss = loss + reg_loss
        total_time = count_total_time(model)
        nfe_forward = count_nfe(model)

        loss.backward()
        optimizer.step()

        # Eval
        nfe_total = count_nfe(model)
        nfe_backward = nfe_total - nfe_forward
        nfef_meter.update(nfe_forward)
        nfeb_meter.update(nfe_backward)
        time_meter.update(time.time() - end)
        tt_meter.update(total_time)

        log_message = (
            "Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f}) |"
            " NFE Forward {:.0f}({:.1f})"
            " | NFE Backward {:.0f}({:.1f})".format(
                itr,
                time_meter.val,
                time_meter.avg,
                loss_meter.val,
                loss_meter.avg,
                nfef_meter.val,
                nfef_meter.avg,
                nfeb_meter.val,
                nfeb_meter.avg,
            )
        )
        print(log_message)
        end = time.time()
    return args, model



def train(args):
    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    if args.use_cpu:
        device = torch.device("cpu")

    # if args.dataset == "zebrafish":
    #     args.data = ZebrafishData(
    #         dir_path="../data/single_cell/experimental/zebrafish_embryonic/new_processed/",
    #         split_type=args.split_type,
    #         n_pcs=50, use_velocity=False
    #     )
    # else:
    #     args.data = dataset.SCData.factory(args.dataset, args)


    args.timepoints = args.data.get_unique_times()
    # Use maximum timepoint to establish integration_times
    # as some timepoints may be left out for validation etc.
    args.int_tps = (np.arange(max(args.timepoints) + 1) + 1.0) * args.time_scale

    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = build_model_tabular(args, args.data.get_shape()[0], regularization_fns).to(device)
    growth_model = None
    print(model)

    args, model = train_func(
        device,
        args,
        model,
        growth_model,
        regularization_coeffs,
        regularization_fns
    )
    return args, model


if __name__ == '__main__':
    args = parser.parse_args()
    # args.dataset = "EB-PCA"
    args.dataset = "zebrafish"
    args.split_type = "interpolation"
    args.save = "../results/tmp/"
    args.top_k_reg = 0.1
    args.training_noise = 0.0
    # args.max_dim = 5
    args.niters = 5
    args.batch_size = 32
    args.use_growth = False
    args.leaveout_timepoint = 5
    train(args)