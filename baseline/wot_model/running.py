'''
Description:
    Wrapper for running WOT model in our benchmarks
    Codes are adopted from WOT source codes.

Reference:
    [1] https://github.com/broadinstitute/wot
    [2] https://broadinstitute.github.io/wot/tutorial/
'''

import sys
sys.path.append("./")
sys.path.append("../")
from tqdm import tqdm

from baseline.wot_model.wot.ot import OTModel
from baseline.wot_model.wot.ot.util import interpolate_with_ot



def wotSimulate(
        traj_data, ann_data, n_tps, train_tps, test_tps, num_cells, tp_field="tp",
        epsilon=0.05, lambda1=1, lambda2=50, tau=10000, growth_iters=3
):
    ot_model = OTModel(
        ann_data, day_field=tp_field,
        epsilon=epsilon, lambda1=lambda1, lambda2=lambda2, tau=tau, growth_iters=growth_iters
    )
    interpolate_data = []
    epoch_pbar = tqdm(test_tps, desc="[ WOT Interpolation ]")
    for t in epoch_pbar:
        source_t = int(t - 1)
        source_idx = train_tps.index(source_t)
        dest_t = int(t + 1)
        dest_idx = train_tps.index(dest_t)
        tmap = ot_model.compute_transport_map(source_t, dest_t)
        source_x = traj_data[source_idx]
        dest_x = traj_data[dest_idx]
        mid_x = interpolate_with_ot(source_x, dest_x, tmap.X, interp_frac=0.5, size=num_cells)
        interpolate_data.append(mid_x)
    all_recon_data = []
    for t in range(n_tps):
        if t in train_tps:
            all_recon_data.append(traj_data[train_tps.index(t)])
        else:
            all_recon_data.append(interpolate_data[test_tps.index(t)])
    return all_recon_data


def wotSimulate4Tuning(
        traj_data, ann_data, n_tps, train_tps, num_cells, tp_field="tp",
        epsilon=0.05, lambda1=1, lambda2=50, tau=10000, growth_iters=3
):
    ot_model = OTModel(
        ann_data, day_field=tp_field,
        epsilon=epsilon, lambda1=lambda1, lambda2=lambda2, tau=tau, growth_iters=growth_iters
    )
    interpolate_train_data = []
    epoch_pbar = tqdm(train_tps, desc="[ WOT for Parameter Tuning ]")
    for t in epoch_pbar:
        source_t = int(t - 1)
        dest_t = int(t + 1)
        if source_t not in train_tps or dest_t not in train_tps:
            interpolate_train_data.append(None)
            continue
        source_idx = train_tps.index(source_t)
        dest_idx = train_tps.index(dest_t)
        tmap = ot_model.compute_transport_map(source_t, dest_t)
        source_x = traj_data[source_idx]
        dest_x = traj_data[dest_idx]
        mid_x = interpolate_with_ot(source_x, dest_x, tmap.X, interp_frac=0.5, size=num_cells)
        interpolate_train_data.append(mid_x)
    return interpolate_train_data