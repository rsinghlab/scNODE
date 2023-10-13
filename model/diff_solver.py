'''
Description:
    Differential equation solver.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import torch
import torchdiffeq
import torch.nn as nn

# ===========================================

class ODE(nn.Module):
    '''
    Ordinary differential equation (ODE) solver.
    '''
    def __init__(self, input_dim, drift_net, ode_method):
        super(ODE, self).__init__()
        self.input_dim = input_dim
        self.net = drift_net
        self.ode_method = ode_method
        self.rtol = 1e-5 # upper bound on relative error
        self.atol = 1e-5 # upper bound on absolute error

    def forward(self, first_data, tp_to_pred):
        pred_data = torchdiffeq.odeint(self.net.forwardWithTime, first_data, tp_to_pred, method=self.ode_method, rtol=self.rtol, atol=self.atol)
        pred_data = torch.moveaxis(pred_data, [0, 1, 2], [1, 0, 2]) # (# cells, # tps, # genes) after flipping axes
        return pred_data