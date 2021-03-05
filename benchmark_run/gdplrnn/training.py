# Written by Max Beutelspacher and Dominik Schmidt
# Copyright: Dept. of Theoretical Neuroscience, CIMH, Heidelberg University
#
# Please cite:
#   Dominik Schmidt, Georgia Koppe, Zahra Monfared, Max Beutelspacher,
#   Daniel Durstewitz, Identifying nonlinear dynamical systems with multiple
#   time scales and long-range dependencies, ICLR (2021)

from gdplrnn.model import PLRNN, Config, generate_random_params
import torch
import numpy as np
from torch.autograd import Variable
from gdplrnn.comparison_models import IRNN, LSTM, np_RNN, UninitializedRNN, oRNN
from gdplrnn.uRNN import uRNN
from collections import namedtuple
import time
import torch.nn.functional as F


class TrainingConfig(
        namedtuple(
            'TrainingConfig',
            'network_structure problem d_hidden learning_rate use_cuda optimize_C protected_dimension tau_A tau_W tau_C initialize_regulated criterion accuracy_threshold seed W_regularization_scheme provide_first_input_twice'
        )):
    def __new__(cls,
                network_structure,
                problem,
                d_hidden,
                learning_rate,
                use_cuda=False,
                optimize_C=True,
                protected_dimension=0,
                tau_A=0,
                tau_W=0,
                tau_C=0,
                initialize_regulated='AWh',
                accuracy_threshold=0.04,
                W_regularization_scheme='L',
                seed=0,
                provide_first_input_twice=False):
        if network_structure not in ['PLRNN', 'oPLRNN', 'oRNN', 'L2PLRNN', 'L2PLRNN2']:
            optimize_C = True
            protected_dimension = 0
            W_regularization_scheme = None
            tau_A = 0
            tau_W = 0
            tau_C = 0
            initialize_regulated = ''
            provide_first_input_twice = False
        criterions = {
            'Addition': 'MSE',
            'MNIST': 'CrossEntropy',
            'Multiplication': 'MSE',
        }
        return super().__new__(
            cls, network_structure, problem, d_hidden, learning_rate, use_cuda,
            optimize_C, protected_dimension, tau_A, tau_W, tau_C,
            initialize_regulated, criterions[problem], accuracy_threshold,
            seed, W_regularization_scheme, provide_first_input_twice)


def create_model(t_config):
    if t_config.seed > 0:
        np.random.seed(t_config.seed)
        torch.manual_seed(t_config.seed)
    else:
        lastints = lambda x: int(x % 1e16 % 1e16 % 1e15 % 1e14 \
                               % 1e13 % 1e12 % 1e11 % 1e10 % 1e9 )
        np.random.seed(lastints(time.time()))
        torch.manual_seed(lastints(time.time()))
    d_in = {
        'Addition': 2,
        'MNIST': 1,
        'Multiplication': 2,
    }
    d_in = d_in[t_config.problem]
    d_out = {
        'Addition': 1,
        'MNIST': 10,
        'Multiplication': 1,
    }
    d_out = d_out[t_config.problem]
    if t_config.network_structure in ['PLRNN', 'oPLRNN', 'L2PLRNN', 'L2PLRNN2']:
        model = create_plrnn_model(t_config, d_in, d_out)
    elif t_config.network_structure == 'IRNN':
        model = IRNN(d_in, t_config.d_hidden, d_out)
    elif t_config.network_structure == 'np_RNN':
        model = np_RNN(d_in, t_config.d_hidden, d_out)
    elif t_config.network_structure == 'LSTM':
        model = LSTM(d_in, t_config.d_hidden, d_out)
    elif t_config.network_structure in ['uninitialized', 'l2RNN']:
        model = UninitializedRNN(d_in, t_config.d_hidden, d_out)
    elif t_config.network_structure == 'oRNN':
        model = oRNN(d_in, t_config.d_hidden, d_out)
    elif t_config.network_structure == 'uRNN':
        model = uRNN(d_in, t_config.d_hidden, d_out)
    else:
        raise KeyError('{} is an  unknown network_structure'.format(
            t_config.network_structure))
    model.t_config = t_config
    if t_config.use_cuda:
        model.cuda()
    return model


def W_regularization_mask(t_config):
    mask = np.zeros((t_config.d_hidden, t_config.d_hidden), dtype=bool)
    if t_config.W_regularization_scheme == 'L':
        mask[:, :t_config.protected_dimension] = True
    elif t_config.W_regularization_scheme == 'R':
        mask[:t_config.protected_dimension, :] = True
    elif t_config.W_regularization_scheme == 'F':
        mask[:, :] = True
    return np.where(mask)


def generate_random_params_according_to_initialization(t_config, config):
    if t_config.network_structure == 'PLRNN':
        params = generate_random_params(config)
        if 'A' in t_config.initialize_regulated:
            params.A[:t_config.protected_dimension] = 1
        if 'W' in t_config.initialize_regulated:
            params.W[W_regularization_mask(t_config)] = 0
        if 'h' in t_config.initialize_regulated:
            params.h[:t_config.protected_dimension] = 0
        if 'C' in t_config.initialize_regulated:
            params.C[:t_config.protected_dimension, :] = 0
        return params
    elif t_config.network_structure == 'oPLRNN':
        params = generate_random_params(config)
        WW = torch.from_numpy(params.A + params.W)
        torch.nn.init.orthogonal_(WW)
        A = WW.diag().diag()
        W = WW - WW.diag().diag()
        params.A = A.detach().numpy()
        params.W = W.detach().numpy()
        params.h[:] = 0
        return params
    if t_config.network_structure in ['L2PLRNN', 'L2PLRNN2']:
        params = generate_random_params(config)
        if 'A' in t_config.initialize_regulated:
            params.A[:t_config.protected_dimension] = 0
        if 'W' in t_config.initialize_regulated:
            params.W[W_regularization_mask(t_config)] = 0
        if 'h' in t_config.initialize_regulated:
            params.h[:t_config.protected_dimension] = 0
        if 'C' in t_config.initialize_regulated:
            params.C[:t_config.protected_dimension, :] = 0
        return params


def create_plrnn_model(t_config, d_in, d_out):
    config = Config(T=0, d_in=d_in, d_out=d_out, d_hidden=t_config.d_hidden)
    params = generate_random_params_according_to_initialization(
        t_config, config)
    if not t_config.optimize_C:
        if t_config.problem == 'Addition' or t_config.problem == 'Multiplication':
            params.C = np.concatenate(
                [np.identity(2)] * int(config.d_hidden / 2), axis=0)
            if config.d_hidden % 2 != 0:
                params.C = np.concatenate([params.C, np.ones((1, 2))], axis=0)
        else:
            params.C = np.ones((t_config.d_hidden, d_in))
    return PLRNN(
        params,
        optimize_C=t_config.optimize_C,
        provide_first_input_twice=t_config.provide_first_input_twice)


def compute_loss(model, inputs, observations):
    if model.t_config.use_cuda:
        inputs = inputs.cuda()
        observations = observations.cuda()
    output = model(inputs)
    loss = compute_loss_from_output(output, observations, model.t_config)
    return loss + regularization_loss(model)


def compute_loss_from_output(output, target, t_config):
    if t_config.criterion == 'MSE':
        criterion = torch.nn.MSELoss()
        return criterion(output, target[:, -1, :])
    elif t_config.criterion == 'CrossEntropy':
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(output, target)


def regularization_loss(model):
    tau_A = model.t_config.tau_A
    tau_W = model.t_config.tau_W
    tau_C = model.t_config.tau_C
    if model.t_config.network_structure == 'PLRNN':
        if model.t_config.protected_dimension is None or model.t_config.protected_dimension < 1:
            if model.t_config.use_cuda:
                return torch.zeros((1)).cuda()
            return torch.zeros((1))

        def mse_submatrix(matrix, t_config, target=0):
            if matrix.dim() == 1:
                submatrix = matrix[:model.t_config.protected_dimension]
            else:
                submatrix = matrix[W_regularization_mask(model.t_config)]

            if len(submatrix) == 0:
                return 0
            else:
                target_tensor = torch.zeros(submatrix.size())+target
                if model.t_config.use_cuda:
                    target_tensor = target_tensor.cuda()
                return F.mse_loss(
                    submatrix, target=target_tensor)

        reg_loss = tau_A * mse_submatrix(model.A.weight, model.t_config, target=1)
        reg_loss += tau_W * mse_submatrix(model.W.weight, model.t_config)
        reg_loss += tau_W * mse_submatrix(model.W.bias, model.t_config)
        target_tensor = torch.zeros(
                model.C[:model.t_config.protected_dimension, :].size())
        if model.t_config.use_cuda:
            target_tensor = target_tensor.cuda()
        reg_loss += tau_C * F.mse_loss(
            model.C[:model.t_config.protected_dimension, :],
            target=target_tensor)

        return reg_loss
    elif model.t_config.network_structure == 'oPLRNN':
        WW = model.A.weight + model.W.weight
        reg_loss = tau_A * F.mse_loss(WW@WW.transpose(0,1), torch.eye(WW.size(0)))
        reg_loss += tau_W * F.mse_loss(model.W.bias, torch.zeros(WW.size(0)))
        return reg_loss
    if model.t_config.network_structure == 'L2PLRNN2':
        part_A = model.A.weight[:model.t_config.protected_dimension]
        part_W = model.W.weight[:model.t_config.protected_dimension, :model.t_config.protected_dimension]
        part_h = model.W.bias[:model.t_config.protected_dimension]
        reg_loss  = tau_A * F.mse_loss(part_A**2, torch.zeros(part_A.size()))
        reg_loss += tau_W * F.mse_loss(part_W**2, torch.zeros(part_W.size()))
        reg_loss += tau_W * F.mse_loss(part_h**2, torch.zeros(part_h.size()))
        return reg_loss
    if model.t_config.network_structure == 'L2PLRNN':
        if model.t_config.protected_dimension is None or model.t_config.protected_dimension < 1:
            if model.t_config.use_cuda:
                return torch.zeros((1)).cuda()
            return torch.zeros((1))

        def mse_submatrix(matrix, t_config, target=0):
            if matrix.dim() == 1:
                submatrix = matrix[:model.t_config.protected_dimension]
            else:
                submatrix = matrix[W_regularization_mask(model.t_config)]

            if len(submatrix) == 0:
                return 0
            else:
                target_tensor = torch.zeros(submatrix.size())+target
                if model.t_config.use_cuda:
                    target_tensor = target_tensor.cuda()
                return F.mse_loss(
                    submatrix, target=target_tensor)

        reg_loss = tau_A * mse_submatrix(model.A.weight, model.t_config, target=0)
        reg_loss += tau_W * mse_submatrix(model.W.weight, model.t_config)
        reg_loss += tau_W * mse_submatrix(model.W.bias, model.t_config)
        target_tensor = torch.zeros(
                model.C[:model.t_config.protected_dimension, :].size())
        if model.t_config.use_cuda:
            target_tensor = target_tensor.cuda()
        reg_loss += tau_C * F.mse_loss(
            model.C[:model.t_config.protected_dimension, :],
            target=target_tensor)

        return reg_loss
    elif model.t_config.network_structure == 'oRNN':
        W = model.state_dict()['rnn.weight_hh_l0']
        return tau_W * F.mse_loss(W@W.transpose(0,1), torch.eye(W.size(0)))
    elif model.t_config.network_structure == 'l2RNN':
        W = model.state_dict()['rnn.weight_hh_l0']
        return tau_W * F.mse_loss(W**2, torch.eye(W.size(0)))
    else:
        return 0


def train_model(model, dataloader, optimizer, clip=3):
    """train model for one epoch

    :param model: model
    :param dataloader: dataloader object for trainings data
    :param optimizer: optimizer library e.g. Adam
    :param lr: learning_rate
    :param tau: regularization parameter for A
    :param tau: regularization parameter for W/h
    :param reg_perc: percentage of regularized units
    :param criterion: loss criterion
    :param clip: clip_grad_norm parameter

    """
    for i, data in enumerate(dataloader):
        try:
            loss = compute_loss(model, Variable(data['inputs']),
                                Variable(data['observations']))
        except ValueError:
            break
#        if loss>100:
#            continue
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()


def evaluate_model(model, dataloader):
    """compute MSELoss of model and percentage of correct results

    :param model: model
    :param dataloader: dataloader object for test data
    :param accuracy_threshold: error threshold under which result is determined as correct
    :returns: mean MSE, percentage of correct results

    """
    mean_loss = 0
    number_correct_test_cases = 0
    for i, data in enumerate(dataloader):
        inputs = Variable(data['inputs'])
        observations = Variable(data['observations'])
        if model.t_config.use_cuda:
            inputs = inputs.cuda()
            observations = observations.cuda()
        output = model(inputs)
        loss = compute_loss_from_output(output, observations, model.t_config)
        number_correct_test_cases += compute_number_correct_test_cases_from_output(
            output, observations, model.t_config)
        mean_loss += loss

    return mean_loss.sum() / len(dataloader), float(
        number_correct_test_cases) / len(dataloader.dataset)


def compute_number_correct_test_cases_from_output(output, target, t_config):
    if t_config.criterion == 'MSE':
        error = torch.abs(target[:, -1, :] - output)
        number_correct_test_cases = torch.sum(
            error < t_config.accuracy_threshold)
    elif t_config.criterion == 'CrossEntropy':
        number_correct_test_cases = (torch.max(output.data,
                                               1)[1] == target.data).sum()
    return number_correct_test_cases
