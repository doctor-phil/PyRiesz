import numpy as np
import torch
import torch.optim as optim


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

