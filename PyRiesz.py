import numpy as np
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
# define some moment functions

# example moment function for average derivative estimator of exp(g(X)) with respect to input X_i
# from Schrimpf & Solimine working paper
def m_avg_derivative(Y, X, g, i):
    """
    Returns the moment function m(Y,X,alpha,i) = dY / dX_i where E[Y|X] = g(X)
    In order to use this, will need to fix i in a lambda function
    """
    output = torch.exp(g(X))
    n = len(Y)
    output.backward((1/n)*torch.ones_like(output), retain_graph=True) # derivative of the mean wrt X is 1/n
    return X.grad[:,i].view(-1,1)

m = lambda Y, X, g: m_avg_derivative(Y,X,g,0) # specific moment function for derivative wrt profit

#####################
# define the riesznet loss functions

def riesz_net_loss(Y, X, eps, riesznet, m, lambda1=0.1, lambda2=1., lambda3=1e-3):
    """
    Returns the loss function from RieszNet (Chernozhukov et al. 2022)
    """
    mse = nn.MSELoss()
    alpha = lambda X: riesznet(X)[0]
    g = lambda X: riesznet(X)[1]
    RRLoss = (1/len(Y)) * torch.sum((alpha(X)**2) - (2*m(Y,X,alpha)))
    REGLoss = mse(Y, g(X))
    TMLELoss = mse(Y - g(X), eps*alpha(X))
    RLoss = sum(p.pow(2).sum() for p in riesznet.parameters())
    return REGLoss + lambda1*RRLoss + lambda2*TMLELoss + lambda3*RLoss


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

