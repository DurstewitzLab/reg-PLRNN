# Written by Max Beutelspacher and Dominik Schmidt
# Copyright: Dept. of Theoretical Neuroscience, CIMH, Heidelberg University
#
# Please cite:
#   Dominik Schmidt, Georgia Koppe, Zahra Monfared, Max Beutelspacher,
#   Daniel Durstewitz, Identifying nonlinear dynamical systems with multiple
#   time scales and long-range dependencies, ICLR (2021)
import torch.nn as nn
import torch
import math
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from collections import namedtuple
import numpy as np
import copy


class DiagLinear(nn.Module):
    def __init__(self, dim_features, bias=True):
        super(DiagLinear, self).__init__()
        self.in_features = dim_features
        self.out_features = dim_features
        self.weight = nn.Parameter(torch.Tensor(dim_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(dim_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        full_tensor = torch.diag(self.weight)
        return nn.functional.linear(input, full_tensor, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
        + 'in_features=' + str(self.in_features) \
        + ', out_features=' + str(self.out_features) \
        + ', bias=' + str(self.bias is not None) + ')'


class OffDiagLinear(nn.Linear):
    def __init__(self, dim_features, bias=True):
        super().__init__(dim_features, dim_features)

    def forward(self, input):
        full_tensor = self.weight - torch.diag(torch.diag(self.weight))
        return nn.functional.linear(input, full_tensor, self.bias)


class PLRNNCell(nn.Module):
    def __init__(self, initial_parameters, optimize_C=False):
        super(PLRNNCell, self).__init__()
        input_size = initial_parameters.C.shape[1]
        hidden_size = initial_parameters.A.shape[0]
        self.d_hidden = hidden_size
        output_size = initial_parameters.B.shape[0]
        self.A = DiagLinear(hidden_size, bias=False)
        self.A.weight.data = torch.from_numpy(np.diag(
            initial_parameters.A)).float()
        self.B = nn.Linear(hidden_size, output_size, bias=False)
        self.B.weight.data = torch.from_numpy(initial_parameters.B).float()
        self.W = OffDiagLinear(hidden_size)
        self.W.weight.data = torch.from_numpy(initial_parameters.W).float()
        self.W.bias.data = torch.from_numpy(initial_parameters.h).float()
        if optimize_C:
            self.CLinear = nn.Linear(input_size, hidden_size, bias=False)
            self.CLinear.weight.data = torch.from_numpy(
                initial_parameters.C).float()
            self.C = self.CLinear.weight
        else:
            self.C = Variable(
                torch.from_numpy(initial_parameters.C).float(),
                requires_grad=False)
        self.mu0 = nn.Parameter(torch.Tensor(hidden_size))
        self.mu0.data = torch.from_numpy(initial_parameters.mu0).float()
        self.Phi = nn.ReLU()

    def forward(self, input, hidden):
        hidden = self.A(hidden) + self.W(self.Phi(hidden)) + input.matmul(
            self.C.t())
        output = self.B(hidden)
        return output, hidden

    def initHidden(self, first_input=None):
        if first_input is None:
            return self.mu0
        else:
            return self.mu0 + first_input.matmul(self.C.t())


class PLRNN(PLRNNCell):
    def __init__(self,
                 initial_parameters,
                 optimize_C=False,
                 provide_first_input_twice=False):
        super(PLRNN, self).__init__(initial_parameters, optimize_C)
        self.provide_first_input_twice = provide_first_input_twice

    def forward(self, inputs):
        T = inputs.size()[1]
        if self.provide_first_input_twice:
            hidden = self.initHidden(inputs[:, 0, :])
        else:
            hidden = self.initHidden()
        for t in range(T):
            hidden = self.A(hidden) + self.W(
                self.Phi(hidden)) + inputs[:, t, :].matmul(self.C.t())
        return self.B(hidden)


Config = namedtuple('Config', ['T', 'd_in', 'd_out', 'd_hidden'])


def random_positiv_definit_matrix(dim):
    """generate a standard normal positiv definit matrix.

    :param dim: dimension of the square matrix
    :returns: random matrix

    """
    matrix = np.random.standard_normal((dim, dim))
    return matrix.dot(matrix.T)


class Params:
    '''Parameter Class which holds the Models Parameter matrices'''

    def __init__(self, A=0, W=0, h=0, C=0, B=0, Sigma=0, Gamma=0, mu0=0):
        self.data = {
            'A': A,
            'B': B,
            'W': W,
            'h': h,
            'C': C,
            'Sigma': Sigma,
            'Gamma': Gamma,
            'mu0': mu0,
        }

    def copy(self):
        new_params = Params()
        new_params.data = copy.deepcopy(self.data)
        return new_params

    @property
    def A(self):
        return self.data['A']

    @A.setter
    def A(self, A):
        self.data['A'] = A

    @property
    def W(self):
        return self.data['W']

    @W.setter
    def W(self, W):
        self.data['W'] = W

    @property
    def B(self):
        return self.data['B']

    @B.setter
    def B(self, B):
        self.data['B'] = B

    @property
    def C(self):
        return self.data['C']

    @C.setter
    def C(self, C):
        self.data['C'] = C

    @property
    def Sigma(self):
        return self.data['Sigma']

    @Sigma.setter
    def Sigma(self, Sigma):
        self.data['Sigma'] = Sigma

    @property
    def Gamma(self):
        return self.data['Gamma']

    @Gamma.setter
    def Gamma(self, Gamma):
        self.data['Gamma'] = Gamma

    @property
    def h(self):
        return self.data['h']

    @h.setter
    def h(self, h):
        self.data['h'] = h

    @property
    def mu0(self):
        return self.data['mu0']

    @mu0.setter
    def mu0(self, mu0):
        self.data['mu0'] = mu0


def generate_random_params(
        config,
        error_independent=False,
):

    AW = np.random.normal(
        loc=0,
        scale=1 / np.sqrt(config.d_hidden),
        size=(config.d_hidden, config.d_hidden))

    def reduction(dim):
        '''scale matrix with 1 over sqrt(dim) in order to decrease spectral radius'''
        return 1 / (1 + np.sqrt(dim))

    while np.max(np.abs(np.linalg.eigvals(AW))) > 1:
        AW *= reduction(config.d_hidden)
    A = np.diag(np.diagonal(AW))
    W = AW - A
    C = np.random.normal(
        loc=0,
        scale=1 / np.sqrt(config.d_hidden),
        size=(config.d_hidden, config.d_in))
    C = C * np.sqrt(2) * np.exp(1.2 / np.maximum(config.d_hidden, 6) - 2.4)
    B = np.random.normal(
        0, 1 / np.sqrt(config.d_hidden), size=(config.d_out, config.d_hidden))
    if error_independent:
        Sigma = np.diag(np.random.random(config.d_hidden))
        Gamma = np.diag(np.random.random(config.d_out))
    else:
        Sigma = random_positiv_definit_matrix(config.d_hidden)
        Gamma = random_positiv_definit_matrix(config.d_out)
    h = np.random.normal(
        loc=0, scale=1 / np.sqrt(config.d_hidden), size=config.d_hidden)
    mu0 = np.random.random(config.d_hidden)
    return Params(A=A, W=W, h=h, C=C, B=B, Sigma=Sigma, Gamma=Gamma, mu0=mu0)
