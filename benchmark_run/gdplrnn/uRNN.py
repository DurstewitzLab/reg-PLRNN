import numpy as np
import torch as th
import torch.nn as nn
from gdplrnn.model import DiagLinear
import matplotlib.pyplot as plt
import torch.nn as nn
from gdplrnn.datasets import gen_addition_problem


def initialize_matrix(n_in, n_out, rng):
    bin = np.sqrt(6. / (n_in + n_out))
    values = np.array(rng.uniform(low=-bin,
                                    high=bin,
                                    size=(n_in, n_out)))
    return th.tensor(values, dtype=th.get_default_dtype())

def times_diag(input, n_hidden, diag, swap_re_im):
    d = th.cat([diag, -diag])
    
    Re = th.cos(d).reshape(1,d.shape[0])
    Im = th.sin(d).reshape(1,d.shape[0])

    input_times_Re = input * Re
    input_times_Im = input * Im

    output = input_times_Re + input_times_Im[:, swap_re_im]
   
    return output

def times_reflection(input, n_hidden, reflection):
    input_re = th.squeeze(input[:, :n_hidden])
    input_im = th.squeeze(input[:, n_hidden:])

    reflect_re = reflection[:n_hidden]
    reflect_im = reflection[n_hidden:]
    
    vstarv = (reflection**2).sum()

    input_re_reflect_re = th.dot(input_re, reflect_re)
    input_re_reflect_im = th.dot(input_re, reflect_im)
    input_im_reflect_re = th.dot(input_im, reflect_re)
    input_im_reflect_im = th.dot(input_im, reflect_im)

    a = th.ger(input_re_reflect_re.reshape(1) - input_im_reflect_im.reshape(1), reflect_re)
    b = th.ger(input_re_reflect_im.reshape(1) + input_im_reflect_re.reshape(1), reflect_im)
    c = th.ger(input_re_reflect_re.reshape(1) - input_im_reflect_im.reshape(1), reflect_im)
    d = th.ger(input_re_reflect_im.reshape(1) + input_im_reflect_re.reshape(1), reflect_re)

    output = input
    output[:, :n_hidden] = output[:, :n_hidden] - 2. / vstarv * (a + b)
    output[:, n_hidden:] = output[:, n_hidden:] - 2. / vstarv * (d - c)

    return output  

def do_fft(input, n_hidden):
    fft_input = th.reshape(input, (input.shape[0], 2, n_hidden))
    fft_input = fft_input.permute(0,2,1)
    fft_output = th.fft(fft_input, 2) / th.sqrt(th.tensor(n_hidden, dtype=th.get_default_dtype()))
    fft_output = fft_output.permute(0,2,1)
    output = th.reshape(fft_output, (input.shape[0], 2*n_hidden))
    return output

def vec_permutation(input, index_permute):
    return input[:, index_permute]

def do_ifft(input, n_hidden):
    ifft_input = th.reshape(input, (input.shape[0], 2, n_hidden))
    ifft_input = ifft_input.permute(0,2,1)
    ifft_output = th.ifft(ifft_input, 2) / th.sqrt(th.tensor(n_hidden, dtype=th.get_default_dtype()))
    ifft_output = ifft_output.permute(0,2,1)
    output = th.reshape(ifft_output, (input.shape[0], 2*n_hidden))
    return output

class uRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(uRNN, self).__init__()
        n_input = input_size; n_hidden = hidden_size; n_output = output_size;
        rng = np.random.RandomState(1234)

        self.B = nn.Linear(2*hidden_size, output_size, bias=False)
        
        self.CLinear = nn.Linear(input_size, 2*hidden_size, bias=False)
        #self.C = self.CLinear.weight

        self.reflection = initialize_matrix(2, 2*hidden_size, rng)
        self.theta = th.tensor(np.array(rng.uniform(low=-np.pi, high=np.pi, size=(3, hidden_size))), dtype=th.get_default_dtype())
        #parameters = [V, U, hidden_bias, reflection, out_bias, theta, h_0]
        
        self.index_permute = th.tensor(np.random.permutation(hidden_size), dtype=th.get_default_dtype())
        self.index_permute_long = th.tensor(np.concatenate((self.index_permute, self.index_permute + hidden_size)), dtype=th.get_default_dtype())
        self.swap_re_im = np.concatenate((np.arange(hidden_size, 2*hidden_size), np.arange(hidden_size)))
        
        self.hidden_size = hidden_size
        
        self.mu0 = nn.Parameter(th.zeros(20))
        
        
        
    def initHidden(self, first_input=None):
        if first_input is None:
            return th.cat([self.mu0,-self.mu0])
        else:
            return self.mu0 + first_input.matmul(self.C.t())
        
    def recurrence(self, hidden):
        hidden = times_diag(hidden, self.hidden_size, self.theta[0,:], self.swap_re_im)
        hidden = do_fft(hidden, self.hidden_size)
        hidden = times_reflection(hidden, self.hidden_size, self.reflection[0,:])
        hidden = vec_permutation(hidden, self.hidden_size)
        hidden = times_diag(hidden, self.hidden_size, self.theta[1,:], self.swap_re_im)
        hidden = do_ifft(hidden, self.hidden_size)
        hidden = times_reflection(hidden, self.hidden_size, self.reflection[1,:])
        hidden = times_diag(hidden, self.hidden_size, self.theta[2,:], self.swap_re_im) 
        return hidden
    
    def forward(self, inputs):
        T = inputs.size()[1]
        N = inputs.size()[0]
        out = th.zeros(N,1)
        for n in range(N):
            hidden = self.initHidden()
            for t in range(T):
                rec = self.recurrence(hidden)
                #Cinp = inputs[:, t, :].matmul(self.C.t())
                Cinp = self.CLinear(inputs[n,t,:])
                hidden = Cinp + rec
            out[n,0] = self.B(hidden)
        return out
        
