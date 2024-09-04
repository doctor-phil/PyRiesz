import numpy as np
import torch
from torch import nn
import torch.optim as optim


class RieszNet(nn.Module): # following the specification from Chernozhukov et al. (2022) PMLR
    def __init__(self, d, k1 = 2*d, k2 = d//2, num_layers_representation = 1, num_layers_output = 1, hidden_layers1 = None, hidden_layers2 = None, activation=torch.relu):
        """
        Initializes the RieszNet model.

        Parameters:
            d: number of input features
            
        Optional arguments:
            k1: number of nodes in each layer before the split, with the last layer being the latent representation used to estimate alpha (defaults to k1=2d)
            k2: number of units in the second layer, after the split, used to estimate the nuisance function g (defaults to k2= d//2)
            num_layers_representation: number of layers before the split (defaults to 1)
            num_layers_output: number of layers after the split, used to estimate g (defaults to 1)
            hidden_layers1: list of hidden layers for the latent representation, allowing for more flexible descriptions of the representation layers
                Defaults to None, in which case the hidden layers are all linear transformations using k1 nodes.
                If specified, num_layers_representation and k1 will be ignored
            hidden_layers2: list of hidden layers for the nuisance function g, allowing for more flexible descriptions of the output layers
                Defaults to None, in which case the hidden layers are all linear transformations using k2 nodes.
                If specified, num_layers_output and k2 will be ignored
            activation: activation function used for all hidden layers (defaults to ReLU)
        """
        super(RieszNet, self).__init__()
        self.first_layer = nn.Linear(d, k1)
        if hidden_layers1 is None:
            self.hidden_layers1 = nn.ModuleList([nn.Linear(k1, k1) for _ in range(num_layers_representation)])
        else:
            self.hidden_layers1 = hidden_layers1
        self.transition_layer = nn.Linear(k1, k2)
        if hidden_layers2 is None:
            self.hidden_layers2 = nn.ModuleList([nn.Linear(k2, k2) for _ in range(num_layers_output)])
        else:
            self.hidden_layers2 = hidden_layers2
        self.output_alpha = nn.Linear(k1, 1)
        self.output_g = nn.Linear(k2, 1)
        self.layer1 = num_layers_representation # number of layers for both alpha and g
        self.layer2 = num_layers_output # number of layers for just g, after the split
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
# define some moment functions

# example moment function for average derivative estimator
def m_derivative(Y, X, g, i):
    """
    Returns the moment function m(Y,X,alpha,i) = dY / dX_i where E[Y|X] = g(X)
    In order to use this, will need to fix i in a lambda function

    Parameters:
        Y: output data
        X: input data
        g: nuisance function g(X) = E[Y|X]
        i: index of the input variable to take the derivative with respect to
    """
    output = torch.sum(g(X))
    gr = torch.autograd.grad(output, X, create_graph=True, retain_graph=True)[0]
    return gr[:,i].view(-1,1)

m_deriv_first_arg = lambda Y, X, g: m_derivative(Y,X,g,0) # specific moment function for derivative wrt profit

#####################
# define the riesznet loss functions

def riesz_net_loss(Y, X, eps, riesznet, m, lambda1=0.1, lambda2=1., lambda3=1e-3, lambda4=1e-3):
    """
    Returns the loss function from RieszNet (Chernozhukov et al. 2022)

    Parameters:
        Y: output data to be fit by g
        X: input data
        eps: tuning parameter for targeted maximum likelihood estimator
        riesznet: the RieszNet model
        m: moment function m(Y,X,alpha) where alpha is the output of the RieszNet model
    
    Optional arguments:
        lambda1: regularization parameter for the Riesz representer loss (defaults to 0.1)
        lambda2: regularization parameter for the TML estimator loss (defaults to 1)
        lambda3: regularization parameter for the L2 norm of the RieszNet parameters, except for epsilon (defaults to 0.001
        lambda4: regularization parameter for the L1 norm of the RieszNet parameters, except for epsilon (defaults to 0.001)
    """
    mse = nn.MSELoss()
    alpha = lambda X: riesznet(X)[0]
    g = lambda X: riesznet(X)[1]
    RRLoss = (1/len(Y)) * torch.sum((alpha(X)**2) - (2*m(Y,X,alpha)))
    REGLoss = mse(Y, g(X))
    TMLELoss = mse(Y - g(X), eps*alpha(X))
    L2Loss = sum(p.pow(2).sum() for p in riesznet.parameters())
    L1Loss = sum(p.abs().sum() for p in riesznet.parameters())
    return REGLoss + lambda1*RRLoss + lambda2*TMLELoss + lambda3*L2Loss + lambda4*L1Loss


#####################
# alternative, estimate Riesz representer using LASSO method from Chernozhukov et al. (2022) ECMA

def e(i,k):
    """
    Returns the i-th standard basis vector in R^k.
    """
    return torch.zeros(k,dtype=torch.float32).scatter(0, torch.tensor(i), 1.).view(-1,1)

def lasso_Rr(m,b,data,r=1.,maxit=100000,tol=1e-20, printevery=1000,dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    """
    Estimates the Riesz representer of the moment function m using LASSO with penalty parameter r.

    Parameters:
        m: moment function m(W,b) where W is the input and output data
        b: list of basis functions b_i(X) where X is only the input data
        data: input and output data

    Optional arguments:
        r: LASSO penalty parameter
        maxit: maximum number of iterations
        tol: convergence tolerance
        dev: torch device (will default to CUDA if available)
        printevery: print the objective value every printevery iterations

    Returns:
        rho: the estimated parameters of the Riesz representer
            appoximated on the basis b
    """
    k = len(b) # b is a list of lambda functions, which should be functions of the input data
    n = len(data)
    Mhat = sum([data.apply(lambda x: m(x,b[i]), axis=1).sum() * e(i,k) for i in range(k)]) / n
    Gsub = torch.tensor(data.apply(b, axis=1).to_numpy(), dtype=torch.float32, device=dev)
    Ghat = ((Gsub.t() @ Gsub) / n).to(dev)
    
    objective = lambda rho: (-2 * Mhat.t() @ rho) + (rho.t() @ Ghat @ rho) + 2*r*torch.norm(rho,1)
    rho = torch.tensor(np.random.normal(size=(k,1)), dtype=torch.float32, device=dev, requires_grad=True)
    rho_old = torch.tensor(np.zeros((k,1)), dtype=torch.float32, device=dev)
    optimizer = optim.LBFGS([rho], line_search_fn="strong_wolfe")
    objective_values = np.zeros(maxit)
    
    def closure():
        optimizer.zero_grad()
        obj = objective(rho)
        obj.backward()
        return obj
    
    for i in range(maxit):
        rho_old = rho.detach().clone()
        optimizer.step(closure)
        objective_values[i] = objective(rho).item()
        if i % printevery == 0:
            print(f'iteration {i}, objective {objective_values[i]}')
        if torch.norm(rho - rho_old,1) < tol:
            print(f'LASSO converged in {i} iterations')
            break
        if i == maxit - 1:
            print(f'LASSO did not converge in {maxit} iterations')

    return rho

