# Written by Max Beutelspacher and Dominik Schmidt
# Copyright: Dept. of Theoretical Neuroscience, CIMH, Heidelberg University
#
# Please cite:
#   Dominik Schmidt, Georgia Koppe, Zahra Monfared, Max Beutelspacher,
#   Daniel Durstewitz, Identifying nonlinear dynamical systems with multiple
#   time scales and long-range dependencies, ICLR (2021)
import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms


def gen_marked_input_problem(T, n_trials, function):
    """generate inputs and outputs of problem with 2 random marked inputs and the last observation being the result of a function applied on the marked inputs

    :param T: length of each time series
    :param n_trials: number of time series
    :param function: function that takes to values and return one value
    :returns: inputs(n_trials x T x 2), outputs(n_trials x T x 1)
    :rtype: np.ndarray

    """
    inputs = np.zeros((n_trials, T, 2))
    inputs[:, :, 0] = np.random.random((n_trials, T))
    marked_indices = np.zeros((n_trials, 2))
    while np.any(marked_indices[:, 0] == marked_indices[:, 1]):
        indices_to_regenerate = np.argwhere(
            marked_indices[:, 0] == marked_indices[:, 1]).flatten()
        marked_indices[indices_to_regenerate, 0] = np.random.randint(
            0, 10, size=(len(indices_to_regenerate)))
        marked_indices[indices_to_regenerate, 1] = np.random.randint(
            0, int(T/2), size=(len(indices_to_regenerate)))
    marked_indices = marked_indices.astype(int)
    observations = np.full((n_trials, T, 1), np.nan)

    for (mi, inp, obs) in zip(marked_indices, inputs, observations):
        first_index, second_index = mi
        inp[first_index, 1] = 1
        inp[second_index, 1] = 1
        obs[-1] = function(inp[first_index, 0], inp[second_index, 0])

    return inputs, observations


def gen_addition_problem(T, n_trials):
    '''generate inputs and outputs of n_trials of length T of an addition problem'''
    return gen_marked_input_problem(T, n_trials, lambda x, y: x + y)


def gen_multiplication_problem(T, n_trials):
    '''generate inputs and outputs of n_trials of length T of an multiplication problem'''
    return gen_marked_input_problem(T, n_trials, lambda x, y: x * y)


def gen_memory_problem(T, n_trials):
    """generates input and outputs such that the first input is equal to the last output

    and the other outputs are missing and the other inputs are zero

    :param T: timelap
    :param n_trials: number of trials
    :returns:inputs,outputs

    """
    inputs = np.zeros((n_trials, T, 1))
    inputs[:, 0] = np.random.random((n_trials, 1))
    observations = np.empty((n_trials, T, 1))
    observations[:, :-1] = np.nan
    observations[:, -1] = inputs[:, 0]
    return inputs, observations


def gen_decision_problem(T, n_trials):
    """generates input and output where two inputs are marked (by second dimension) and the last observation should be if the first or last number was greater

    :param T: timelag
    :param n_trials: number of trials
    :returns: inputs,outputs

    """
    inputs = np.zeros((n_trials, T, 2))
    inputs[:, :, 0] = np.random.random((n_trials, T))
    addition_indices = np.zeros((n_trials, 2))
    while np.any(addition_indices[:, 0] == addition_indices[:, 1]):
        indices_to_regenerate = np.argwhere(
            addition_indices[:, 0] == addition_indices[:, 1]).flatten()
        addition_indices[indices_to_regenerate, :] = np.random.randint(
            0, T - 1, size=(len(indices_to_regenerate), 2))
    addition_indices = addition_indices.astype(int)
    observations = np.full((n_trials, T, 1), np.nan)

    for (ad, inp, obs) in zip(addition_indices, inputs, observations):
        first_index, second_index = np.sort(ad)
        inp[first_index, 1] = 1
        inp[second_index, 1] = 1
        obs[-1] = np.sign(inp[first_index, 0] - inp[second_index, 0])

    return inputs, observations


class GeneratedDataset(data.Dataset):
    def __init__(self, T, train=True):
        self.train = train
        if self.train:
            seed = 100
            self.n_trials = 100000
        else:
            seed = 101
            self.n_trials = 10000
        np.random.seed(seed)

    def __getitem__(self, index):
        return {
            'inputs': self.inputs[index, :],
            'observations': self.observations[index, :]
        }

    def __len__(self):
        return self.n_trials

    def get_loader(self, batch_size, shuffle, num_workers):
        """returns loader for this dataset

        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param num_workers: how many subprocesses to use for data loading.
        :returns: data loader
        :rtype: torch.utils.data.DataLoader

        """
        return torch.utils.data.DataLoader(
            self, batch_size, shuffle, num_workers=num_workers)


class AdditionProblemDataset(GeneratedDataset):
    def __init__(self, T, train=True):
        """creates Dataset of addition problems

        :param T: timelag
        :param train: flag to differentiate training and test data

        """
        super().__init__(T, train)
        self.inputs, self.observations = gen_addition_problem(T, self.n_trials)
        self.inputs = torch.from_numpy(self.inputs).float()
        self.observations = torch.from_numpy(self.observations).float()


class MultiplicationDataset(GeneratedDataset):
    def __init__(self, T, train=True):
        """creates Dataset of multiplication problems

        :param T: timelag
        :param train: flag to differentiate training and test data

        """
        super().__init__(T, train)
        self.inputs, self.observations = gen_multiplication_problem(T, self.n_trials)
        self.inputs = torch.from_numpy(self.inputs).float()
        self.observations = torch.from_numpy(self.observations).float()



class MemoryProblemDataset(GeneratedDataset):
    def __init__(self, T, train=True):
        """creates Dataset of memory problems

        :param T: timelag
        :param train: flag to differentiate training and test data

        """
        super().__init__(T, train)
        self.inputs, self.observations = gen_memory_problem(T, self.n_trials)
        self.inputs = torch.from_numpy(self.inputs).float()
        self.observations = torch.from_numpy(self.observations).float()


class DecisionProblemDataset(GeneratedDataset):
    def __init__(self, T, train=True):
        """creates Dataset of decision problems

        :param T: timelag
        :param train: flag to differentiate training and test data

        """
        super().__init__(T, train)
        self.inputs, self.observations = gen_decision_problem(T, self.n_trials)
        self.inputs = torch.from_numpy(self.inputs).float()
        self.observations = torch.from_numpy(self.observations).float()


class MNISTDataset(GeneratedDataset, datasets.MNIST):
    def __init__(self, train=True):
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, )),
            transforms.Lambda(lambda x: x.view(28 * 28, 1))
        ])
        datasets.MNIST.__init__(
            self, root='./.data/', train=train, transform=trans, download=True)

    def __getitem__(self, index):
        img, target = datasets.MNIST.__getitem__(self, index)
        return {'inputs': img, 'observations': target}

    def __len__(self):
        return datasets.MNIST.__len__(self)
