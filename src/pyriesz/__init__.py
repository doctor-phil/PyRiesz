import numpy as np
import torch
from torch import nn
import torch.optim as optim

import pyriesz.moments as moments
from pyriesz.loss import riesz_net_loss
import pyriesz.lasso as lasso


class RieszNet(nn.Module): # following the specification from Chernozhukov et al. (2022) PMLR
    def __init__(self, d, k1 = None, k2 = None, num_layers_representation = 1, num_layers_output = 1, hidden_layers1 = None, hidden_layers2 = None, activation=torch.relu, l1reg = 1e-5, l2reg=1e-5, rr_loss_weight=0.1, tmle_loss_weight=1.):
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
            activation: activation function used for all hidden layers (defaults to ReLU) OR an iterable collection of activation functions for each layer
        """
        super(RieszNet, self).__init__()
        if k1 is None and hidden_layers1 is None:
            k1 = 2*d
        elif k1 is None and hidden_layers1 is not None:
            k1 = hidden_layers1[0].in_features

        if k2 is None and hidden_layers2 is None:
            k2 = d//2
        elif k2 is None and hidden_layers2 is not None:
            k2 = hidden_layers2[0].in_features

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
        self.l1reg = l1reg
        self.l2reg = l2reg
        self.tmle_loss_weight = tmle_loss_weight
        self.rr_loss_weight = rr_loss_weight
        self.eps = torch.tensor(1.0, dtype=torch.float32) # what should it be initialized to?
        self.output_alpha = nn.Linear(k1, 1)
        self.output_g = nn.Linear(k2, 1)
        self.layer1 = num_layers_representation # number of layers for both alpha and g
        self.layer2 = num_layers_output # number of layers for just g, after the split
        if hasattr(activation, '__iter__'):
            self.activation = nn.ModuleList(activation)
        else:
            self.activation = nn.ModuleList([activation for _ in range(num_layers_representation + num_layers_output)])

    def forward(self, x):
        x = self.activation(self.first_layer(x))
        for i in range(self.layer1):
            x = self.activation[i](self.hidden_layers1[i](x))
        alpha = self.output_alpha(x)

        x = self.activation[self.layer1](self.transition_layer(x))
        for i in range(self.layer2):
            x = self.activation[self.layer1 + i + 1](self.hidden_layers2[i](x))
        g = self.output_g(x)

        return alpha, g
    
    def fit(self, X, y, m, eps, epochs=1000, lr=1e-3, printevery=100):
        """
        Fits the RieszNet model to the data.

        Parameters:
            X: input data
            y: output data

        Optional arguments:
            epochs: number of epochs to train the model (defaults to 1000)
            lr: learning rate for the optimizer (defaults to 1e-3)
            printevery: print the loss every printevery epochs
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = riesz_net_loss(y, X, self.eps, self, m, self.rr_loss_weight, self.tmle_loss_weight, self.l1reg, self.l2reg)
            loss.backward()
            optimizer.step()
            if epoch % printevery == 0:
                print(f'epoch {epoch}, loss {loss.item()}')
