import torch
import torch.nn as nn

#####################
# define the riesznet loss functions

def riesz_net_loss(Y, X, eps, riesznet, m, rr_loss_weight=0.1, tmle_loss_weight=1., l2_weight=1e-3, l1_weight=1e-3):
    """
    Returns the loss function from RieszNet (Chernozhukov et al. 2022)

    Parameters:
        Y: output data to be fit by g
        X: input data
        eps: tuning parameter for targeted maximum likelihood estimator
        riesznet: the RieszNet model
        m: moment function m(Y,X,alpha) where alpha is the output of the RieszNet model
    
    Optional arguments:
        rr_loss_weight: regularization parameter for the Riesz representer loss (defaults to 0.1)
        tmle_loss_weight: regularization parameter for the TML estimator loss (defaults to 1)
        l2_weight: regularization parameter for the L2 norm of the RieszNet parameters, except for epsilon (defaults to 0.001
        l1_weight: regularization parameter for the L1 norm of the RieszNet parameters, except for epsilon (defaults to 0.001)
    """
    mse = nn.MSELoss()
    alpha = lambda X: riesznet(X)[0]
    g = lambda X: riesznet(X)[1]
    RRLoss = (1/len(Y)) * torch.sum((alpha(X)**2) - (2*m(Y,X,alpha)))
    REGLoss = mse(Y, g(X))
    TMLELoss = mse(Y - g(X), eps*alpha(X))
    L2Loss = sum(p.pow(2).sum() for p in riesznet.parameters())
    L1Loss = sum(p.abs().sum() for p in riesznet.parameters())
    return REGLoss + rr_loss_weight*RRLoss + tmle_loss_weight*TMLELoss + l2_weight*L2Loss + l1_weight*L1Loss

