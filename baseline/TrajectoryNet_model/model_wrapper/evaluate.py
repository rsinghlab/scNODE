'''
Description:
    A wrapper of TrajectoryNet predicting function.
'''
import os
import numpy as np
import torch

from baseline.TrajectoryNet_model.TrajectoryNet import dataset, eval_utils
from baseline.TrajectoryNet_model.TrajectoryNet.parse import parser

from baseline.TrajectoryNet_model.TrajectoryNet.train_misc import (
    create_regularization_fns,
    build_model_tabular,
)


# ==================================

def integrate_backwards(
    args, end_samples, model, savedir, ntimes=100, memory=0.1, device="cpu"
):
    """Integrate some samples backwards and save the results."""
    with torch.no_grad():
        z = torch.from_numpy(end_samples).type(torch.float32).to(device)
        zero = torch.zeros(z.shape[0], 1).to(z)
        cnf = model.chain[0]

        zs = [z]
        deltas = []
        int_tps = np.linspace(args.int_tps[0], args.int_tps[-1], ntimes)
        for i, itp in enumerate(int_tps[::-1][:-1]):
            # tp counts down from last
            timescale = int_tps[1] - int_tps[0]
            integration_times = torch.tensor([itp - timescale, itp])
            # integration_times = torch.tensor([np.linspace(itp - args.time_scale, itp, ntimes)])
            integration_times = integration_times.type(torch.float32).to(device)

            # transform to previous timepoint
            z, delta_logp = cnf(zs[-1], zero, integration_times=integration_times)
            zs.append(z)
            deltas.append(delta_logp)
        forward_traj = [each.cpu().numpy() for each in zs[::-1]]
        return forward_traj


def eval(args, model):
    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    # Use maximum timepoint to establish integration_times
    # as some timepoints may be left out for validation etc.
    args.int_tps = (np.arange(max(args.timepoints) + 1) + 1.0) * args.time_scale

    regularization_fns, regularization_coeffs = create_regularization_fns(args)

    data = args.data
    args.timepoints = args.data.get_unique_times()
    args.int_tps = (np.arange(max(args.timepoints) + 1) + 1.0) * args.time_scale

    print("integrating backwards")
    # end_time_data = data.data_dict[args.embedding_name]
    end_time_data = data.get_data()[
        args.data.get_times() == np.max(args.data.get_times())
    ]
    # integrate_backwards(args, end_time_data, model, args.save, ntimes=100, device=device)
    zs = integrate_backwards(args, end_time_data, model, args.save, ntimes=args.ntimes, device=device)
    return zs
