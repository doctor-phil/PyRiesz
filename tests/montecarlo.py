import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))) # sometimes i hate python so much

import pyriesz as pr


#####################
# test the RieszNet estimators


#first, we need to generate some data

def dgp(N, dimx, Eyx=lambda x: 1 / (1 + torch.exp(-torch.sum(x))), noise_std=lambda x: 0.1):
    # Define the multivariate normal distribution
    Fx_mean = torch.zeros(dimx)
    Fx_cov = torch.diag(torch.ones(dimx))
    Fx = torch.distributions.MultivariateNormal(Fx_mean, Fx_cov)

    # Generate x samples
    x = Fx.sample((N,))  # Shape: (N, dimx)

    # Generate y samples
    y = Eyx(x.T).detach().numpy() + torch.randn(N).numpy() * noise_std(x.T)

    # Compute gradients for adjustment factors
    alpha = []
    for xi in x: # this could be done with only torch structures...
        xi = xi.requires_grad_()  # Enable gradient tracking
        logpdf = Fx.log_prob(xi)
        logpdf.backward()  # Compute gradients
        alpha.append(-xi.grad[0].item())  # Extract gradient and negate
        xi.grad.zero_()  # Clear gradients for the next iteration

    # Create the DataFrame
    data = {
        'y': y,
        'id': range(1, N + 1),
        'alpha': alpha
    }
    for i in range(dimx):
        data[f'x{i+1}'] = x[:, i].numpy()

    return pd.DataFrame(data)

def fit_full_rn(data, dimx=5):
    rn = pr.RieszNet(d=dimx, num_layers_representation=3, num_layers_output=3, activation=nn.ReLU())
    optimizer = optim.Adam(rn.parameters(), lr=0.01)
    objective = lambda net: pr.riesz_net_loss(torch.tensor(data['y'].values), torch.tensor(data[[f'x{i+1}' for i in range(dimx)]].values), 1., net, pr.m_deriv_first_arg)

    for i in range(1000):
        optimizer.zero_grad()
        obj = objective(rn)
        obj.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f'iteration {i}, objective {obj.item()}')

        # return predicted values for y and alpha
        y_pred = rn(data[[f'x{i+1}' for i in range(dimx)]].values)[0]
        alpha_pred = rn(data[[f'x{i+1}' for i in range(dimx)]].values)[1]
        return y_pred, alpha_pred
    
print(dgp(10, 5))
print(fit_full_rn(dgp(10, 5)))

# work in progress