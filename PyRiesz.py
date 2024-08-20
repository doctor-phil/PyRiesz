import torch
from torch import nn
import torch.optim as optim


class RieszNet(nn.Module): # following the specification from Chernozhukov et al. (2022) PMLR
    def __init__(self, d, k1 = 200, k2 = 100, num_layers1 = 2, num_layers2 = 2, hidden_layers1 = None, hidden_layers2 = None, activation=torch.relu):
        super(RieszNet, self).__init__()
        self.first_layer = nn.Linear(d, k1)
        if hidden_layers1 is None:
            self.hidden_layers1 = [nn.Linear(k1, k1) for _ in range(num_layers1)]
        else:
            self.hidden_layers1 = hidden_layers1
        self.transition_layer = nn.Linear(k1, k2)
        if hidden_layers2 is None:
            self.hidden_layers2 = [nn.Linear(k2, k2) for _ in range(num_layers2)]
        else:
            self.hidden_layers2 = hidden_layers2
        self.output_alpha = nn.Linear(k1, 1)
        self.output_g = nn.Linear(k2, 1)
        self.l1 = num_layers1 # number of layers for both alpha and g
        self.l2 = num_layers2 # number of layers for just g, after the split
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.first_layer(x))
        for i in range(self.l1):
            x = self.activation(self.hidden_layers1[i](x))
        alpha = self.output_alpha(x)

        x = self.activation(self.transition_layer(x))
        for i in range(self.l2):
            x = self.activation(self.hidden_layers2[i](x))
        g = self.output_g(x)

        return alpha, g


#####################
# define the loss functions

# example moment function for average derivative estimator of exp(g(X)) with respect to input X_i
# from Schrimpf & Solimine working paper
def m_avg_derivative(Y, X, g, i):
    """
    Returns the moment function m(Y,X,alpha) = dY / dX_i where E[Y|X] = g(X)
    """
    output = torch.exp(g(X))
    n = len(Y)
    output.backward((1/n)*torch.ones_like(output), retain_graph=True) # derivative of the mean wrt X is 1/n
    return X.grad[:,i].view(-1,1)

# define the riesznet loss functions

m = lambda Y, X, g: m_avg_derivative(Y,X,g,0) # specific moment function for derivative wrt profit

def riesz_net_loss(Y, X, eps, riesznet, m, lambda1=0.1, lambda2=1., lambda3=1e-3):
    """
    Returns the loss function from RieszNet (Chernozhukov et al. 2022)
    """
    mse = nn.MSELoss()
    alpha = lambda X: riesznet(X)[0]
    g = lambda X: riesznet(X)[1]
    RRLoss = (1/len(Y)) * torch.sum((alpha(X)**2) - (2*m(Y,X,alpha))) # need to double check that m(Y,X,alpha) is right, if alpha=alpha(X)
    REGLoss = mse(Y, g(X))
    TMLELoss = mse(Y - g(X), eps*alpha(X))
    RLoss = sum(p.pow(2).sum() for p in riesznet.parameters())
    return REGLoss + lambda1*RRLoss + lambda2*TMLELoss + lambda3*RLoss

