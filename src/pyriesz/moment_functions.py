import torch

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