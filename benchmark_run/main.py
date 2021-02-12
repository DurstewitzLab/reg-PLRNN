import sys
import os
import yaml
import re
from gdplrnn.datasets import AdditionProblemDataset, MNISTDataset, MultiplicationDataset
from gdplrnn.training import create_model, train_model, evaluate_model, TrainingConfig
from create_configfile import create_config_file
import torch
import numpy as np
import socket

torch.set_num_threads(1)

def is_code_running_in_foreground():
    return os.getpgrp() == os.tcgetpgrp(sys.stdout.fileno())

# If the argument is a label, continue with this config,
# otherwise, create a new config file
if len(sys.argv) in [2,3] and not ('=' in sys.argv[1]) \
                       and len(sys.argv[1]) == 8:
    label = sys.argv[1]
    if sys.argv[-1] in ['new','new_run']:
        new_run = True
        sys.argv = sys.argv[:-1]
    else:
        new_run = False
else:
    # Check if this is supposed to be a new run:
    if sys.argv[-1] in ['new','new_run']:
        new_run = True
        sys.argv = sys.argv[:-1]
    else:
        new_run = False
    label = create_config_file(sys.argv)

with open('config_files/' + label + '.yaml', 'r') as ymlfile:
    config = yaml.load(ymlfile)

use_cuda = False

protected_dimension = int(config['d_hidden'] * config['per_reg'])
T = config['T']
t_config = TrainingConfig(
    d_hidden=config['d_hidden'],
    problem=config['problem'],
    use_cuda=use_cuda,
    protected_dimension=protected_dimension,
    network_structure=config['network_structure'],
    optimize_C=config['optimize_C'],
    learning_rate=config['lr'],
    tau_A=config['tau_A'],
    tau_W=config['tau_W'],
    tau_C=config['tau_C'],
    initialize_regulated=config['initialize_regulated'],
    seed=config['seed'],
    W_regularization_scheme=config['W_regularization_scheme'],
    provide_first_input_twice=config['provide_first_input_twice'])

model = create_model(t_config)
ProblemDatasets = {
    'Addition': AdditionProblemDataset,
    'MNIST': lambda T, train=True: MNISTDataset(train),
    'Multiplication': MultiplicationDataset
}
ProblemDataset = ProblemDatasets[t_config.problem]

train_batch_size = config['train_batch_size']
train_loader = ProblemDataset(T=T).get_loader(
    batch_size=train_batch_size, shuffle=True, num_workers=0)

test_batch_size = config['test_batch_size']
test_loader = ProblemDataset(
    T=T, train=False).get_loader(
        batch_size=test_batch_size, shuffle=True, num_workers=0)

optimizer_libraries = {
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD,
    'ADD': torch.optim.Adadelta,
    'ADG': torch.optim.Adagrad,
    'Adamax': torch.optim.Adamax,
    'ASGD': torch.optim.ASGD
}


def reset_optimizer():
    optimizer = optimizer_libraries[config['optimizer']]
    optimizer = optimizer(model.parameters(), lr=model.t_config.learning_rate)
    return optimizer

def is_fit_finished(folder):
    files = os.listdir(folder)
    return 'finished.txt' in files

optimizer = reset_optimizer()
n_epochs = config['n_epochs']
mses = np.zeros(n_epochs)
per_correct = np.zeros(n_epochs)
#    files = os.listdir(subdir)
N = 0
# if new_run, create a new path, else continue last one
get_subdir = lambda N: os.path.join("./Fits", label + '_{0:0>3d}'.format(N))
while os.path.isdir(get_subdir(N)):
    N += 1
if os.path.isdir(get_subdir(N-1)) and not is_fit_finished(get_subdir(N-1))\
                                  and not new_run:
    subdir = get_subdir(N-1)
else:
    subdir = get_subdir(N)

if not os.path.isdir(subdir):
    os.makedirs(subdir)

if (os.path.isdir(subdir)) and (os.listdir(subdir)):
    print('Continue aborted run:')
    files = os.listdir(subdir)
    optimizer_regex = re.compile("optimizer_after_(\d+)_epochs.pt")
    model_regex = re.compile("model_after_(\d+)_epochs.pt")
    optimizer_epochs = [optimizer_regex.match(f) for f in files]
    optimizer_epochs = [
        int(e.group(1)) for e in optimizer_epochs if e is not None
    ]
    model_epochs = [model_regex.match(f) for f in files]
    model_epochs = [int(e.group(1)) for e in model_epochs if e is not None]
    if len(optimizer_epochs) > 0 and len(model_epochs) > 0:
        last_epoch = min(max(model_epochs), max(optimizer_epochs))
        model.load_state_dict(
            torch.load(
                os.path.join(subdir,
                             'model_after_{}_epochs.pt'.format(last_epoch))))
        optimizer.load_state_dict(
            torch.load(
                os.path.join(
                    subdir,
                    'optimizer_after_{}_epochs.pt'.format(last_epoch))))
        print('Continue starting with epoch {}'.format(last_epoch + 1))
    else:
        last_epoch = -1
else:
    last_epoch = -1
    torch.save(model.state_dict(), os.path.join(subdir, 'initial_model.pt'))
for epoch_nr in range(last_epoch + 1, n_epochs):
    if is_code_running_in_foreground():
        print('Starting epoch {}\n'.format(epoch_nr))
    logfile = open(os.path.join(subdir, 'logfile.txt'), 'a')
    logfile.write('Starting epoch {}\n'.format(epoch_nr))
    logfile.close()
    train_model(model, train_loader, optimizer)
    mses[epoch_nr], per_correct[epoch_nr] = evaluate_model(model, test_loader)
    np.savetxt(os.path.join(subdir, 'mses.txt'), mses)
    np.savetxt(os.path.join(subdir, 'per_corr.txt'), per_correct)
    if np.isnan(mses[epoch_nr]):
        logfile = open(os.path.join(subdir, 'logfile.txt'), 'a')
        logfile.write(
                'Error: MSE is nan for epoch {}'.format(
                epoch_nr))
        logfile.close()
        errorfile = open(os.path.join(subdir, 'error.txt'), 'w')
        errorfile.write(
                'Error: MSE is nan for epoch {}'.format(
                epoch_nr))
        errorfile.close()
        raise ValueError(
                'Error: MSE is nan for epoch {}'.format(
                epoch_nr))
        break
    if mses[epoch_nr] > config['error_threshold_to_abort']:
        logfile = open(os.path.join(subdir, 'logfile.txt'), 'a')
        logfile.write(
            'error > error_threshold_to_abort -> abort after {} epochs with MSE={}'.format(
                epoch_nr, mses[epoch_nr]))
        logfile.close()
        errorfile = open(os.path.join(subdir, 'error.txt'), 'w')
        errorfile.write(
            'error > error_threshold_to_abort -> abort after {} epochs with MSE={}'.format(
                epoch_nr, mses[epoch_nr]))
        errorfile.close()
        raise ValueError(
            'error > error_threshold_to_abort -> abort after {} epochs with MSE={}'.format(
                epoch_nr, mses[epoch_nr]))
        break
    torch.save(model.state_dict(),
               os.path.join(subdir,
                            'model_after_{}_epochs.pt'.format(epoch_nr)))
    torch.save(optimizer.state_dict(),
               os.path.join(subdir,
                            'optimizer_after_{}_epochs.pt'.format(epoch_nr)))
    if config['reset_optimizer']:
        optimizer = reset_optimizer()

# Create finished.txt file
finishfile = open(os.path.join(subdir, 'finished.txt'), 'w')
finishfile.close()
