'''
Description:
    Utility functions for testing scNODE pre-training phase.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import numpy as np

# ======================================================

def selectTPs(tp_list, n_tps, strategy):
    assert n_tps <= len(tp_list)
    if n_tps == 0:
        return None
    else:
        if strategy == "random":
            selected_idx = np.random.choice(np.arange(len(tp_list)), n_tps, replace=False)
        elif strategy == "first":
            selected_idx = np.arange(n_tps)
        else:
            raise ValueError("Unknown strategy {}!".format(strategy))
        return selected_idx
