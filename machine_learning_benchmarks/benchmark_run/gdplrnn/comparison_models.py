# Written by Max Beutelspacher and Dominik Schmidt
# Copyright: Dept. of Theoretical Neuroscience, CIMH, Heidelberg University
#
# Please cite:
#   Dominik Schmidt, Georgia Koppe, Zahra Monfared, Max Beutelspacher,
#   Daniel Durstewitz, Identifying nonlinear dynamical systems with multiple
#   time scales and long-range dependencies, ICLR (2021)
import numpy as np
import torch as th
import torch.nn as nn
from gdplrnn.model import DiagLinear

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.output_weights = nn.Linear(hidden_size, output_size)
        self.init_bias()

    def forward(self, inp):
        _, hnn = self.rnn(inp)
        out = self.output_weights(hnn[0][0])
        return out

    def init_bias(self, value=1):
        # initialize the recurrent bias to value
        # if look the documentation you can find two paramenters
        # for the bias in the recurrent layer
        for names in self.rnn._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.rnn, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(value / 2)


# Model taken from  arXiv:1504.00941v2
class IRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(IRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            nonlinearity='relu',
            batch_first=True,
            bias=True)
        self.output_weights = nn.Linear(hidden_size, output_size)

        # Parameters initialization
        self.rnn.state_dict()['weight_hh_l0'].copy_(th.eye(hidden_size))
        self.rnn.bias_ih_l0.data.fill_(0)
        self.rnn.bias_hh_l0.data.fill_(0)
        self.rnn.state_dict()['weight_ih_l0'].copy_(
            th.randn(hidden_size, input_size) / hidden_size)

    def forward(self, inp):
        _, hnn = self.rnn(inp)
        out = self.output_weights(hnn[0])
        return out

# orthogonal RNN initialization
class oRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(oRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            nonlinearity='relu',
            batch_first=True,
            bias=True)
        self.output_weights = nn.Linear(hidden_size, output_size)

        # Parameters initialization
        nn.init.orthogonal_(self.rnn.state_dict()['weight_hh_l0'])
        self.rnn.bias_ih_l0.data.fill_(0)
        self.rnn.bias_hh_l0.data.fill_(0)

    def forward(self, inp):
        _, hnn = self.rnn(inp)
        out = self.output_weights(hnn[0])
        return out

# Model taken from arXiv:1511.03771v3
class np_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(np_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(
            input_size, hidden_size, nonlinearity='relu', batch_first=True)
        self.output_weights = nn.Linear(hidden_size, output_size)

        # Parameters initialization
        R = np.random.normal(0, 1, size=(hidden_size, hidden_size))
        A = R.T @ R / hidden_size
        I = np.identity(hidden_size)
        e = max(np.linalg.eigvalsh(A + I))
        alpha = np.sqrt(2) * np.exp(1.2 / (max(hidden_size, 6)) - 2.4)

        self.rnn.state_dict()['weight_hh_l0'].copy_(
            th.from_numpy((A + I) / e))
        self.rnn.bias_ih_l0.data.fill_(0)
        self.rnn.bias_hh_l0.data.fill_(0)
        self.rnn.state_dict()['weight_ih_l0'].copy_(
            alpha * th.randn(hidden_size, input_size) / hidden_size)
        self.output_weights.weight.data.normal_(
            0.0, 2 / (hidden_size + output_size))

    def forward(self, inp):
        _, hnn = self.rnn(inp)
        out = self.output_weights(hnn[0])
        return out


class UninitializedRNN(nn.Module):
    '''uninitialized RNN for comparison'''

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(
            input_size, hidden_size, nonlinearity='relu', batch_first=True)
        self.output_weights = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        _, hnn = self.rnn(inp)
        out = self.output_weights(hnn[0])
        return out
