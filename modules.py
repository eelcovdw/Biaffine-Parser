"""
Modules used for the biaffine dependency parser
- BiAffineAttn Module
- MLP Module
- TimeDistributed wrapper
"""


import torch
from torch import nn
from torch.nn import functional
from torch.autograd import Variable
import numpy as np


class BiAffineAttn(nn.Module):
    """
    BiAffine Attention layer from https://arxiv.org/abs/1611.01734
    Expects inputs as batch-first sequences [batch_size, seq_length, dim].

    Returns score matrices as [batch_size, dim, dim] for arc attention
    (out_channels=1), and score as [batch_size, out_channels, dim, dim]
    for label attention (where out_channels=#labels).
    """

    def __init__(self, in_dim, out_channels, bias_head=True, bias_dep=True):
        super(BiAffineAttn, self).__init__()
        self.bias_head = bias_head
        self.bias_dep = bias_dep
        self.U = nn.Parameter(torch.Tensor(out_channels,
                                           in_dim + int(bias_head),
                                           in_dim + int(bias_dep)))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.U.size(1))
        self.U.data.uniform_(-stdv, stdv)

    def forward(self, Rh, Rd):
        """
        Returns S = (Rh @ U @ Rd.T) with dims [batchsize, n_channels, t, t]
        S[b, c, i, j] = Score sample b Label c Head i Dep j
        """

        if self.bias_head:
            Rh = self.add_ones_col(Rh)
        if self.bias_dep:
            Rd = self.add_ones_col(Rd)

        # Add dimension to Rh and Rd for batch matrix products,
        # shape [batch, t, d] -> [batch, 1, t, d]
        Rh = Rh.unsqueeze(1)
        Rd = Rd.unsqueeze(1)

        S = Rh @ self.U @ torch.transpose(Rd, -1, -2)

        # If out_channels == 1, squeeze [batch, 1, t, t] -> [batch, t, t]
        return S.squeeze(1)

    @staticmethod
    def add_ones_col(X):
        """
        Add column of ones to each matrix in batch.
        """
        b = Variable(torch.ones(X.data.shape[:-1]).type(type(X.data)))
        return torch.cat([X, b], -1)

    def __repr__(self):
        tmpstr = self.__class__.__name__
        tmpstr += '(\n  (U): {}\n)'.format(self.U.size())
        return tmpstr


class MLP(nn.Module):
    """
    Module for an MLP with dropout.
    """

    def __init__(self, input_size, layer_size, depth, activation, dropout):
        super(MLP, self).__init__()
        self.layers = nn.Sequential()
        act_fn = getattr(nn, activation)
        for i in range(depth):
            self.layers.add_module('fc_{}'.format(i),
                                   nn.Linear(input_size, layer_size))
            if activation:
                self.layers.add_module('{}_{}'.format(activation, i),
                                       act_fn())
            if dropout:
                self.layers.add_module('dropout_{}'.format(i),
                                       nn.Dropout(dropout))
            input_size = layer_size

    def forward(self, x):
        return self.layers(x)


class TimeDistributed(nn.Module):
    """
    Module that mimics Keras TimeDistributed
    source: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    """

    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        # (samples * timesteps, input_size)
        x_reshape = x.contiguous().view(-1, x.size(-1))

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            # (samples, timesteps, output_size)
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            # (timesteps, samples, output_size)
            y = y.contiguous().view(-1, x.size(1), y.size(-1))

        return y