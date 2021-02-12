import os
import shutil
import tarfile
import tempfile
import warnings

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sumatra.projects import load_project
from torch.autograd import Variable

from gdplrnn.datasets import AdditionProblemDataset
from gdplrnn.model import Config, PLRNNCell, generate_random_params
from gdplrnn.training import TrainingConfig, create_model, evaluate_model

project_path = '/zifnas/max.beutelspacher/gdplrnn/'

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def getSMTRecords(records=None,
                  tags=[],
                  reason='',
                  parameters={},
                  atol=1e-10,
                  rtol=1e-10,
                  path=project_path):
    if not records:
        project = load_project(path)
        records = project.record_store.list(project.name, tags=tags)
    records_out = []
    for r in records:
        if set(tags).issubset(set(r.tags)) and reason in r.reason:
            allclose = []
            if not set(parameters).issubset(set(r.parameters.as_dict())):
                continue
            for k, v in parameters.items():
                if type(v) == str:
                    allclose.append(v == r.parameters.as_dict()[k])
                elif np.allclose(
                        v, r.parameters.as_dict()[k], atol=atol, rtol=rtol):
                    allclose.append(True)
                else:
                    allclose.append(False)
            if np.all(allclose):
                records_out.append(r)
    return records_out


def extract_record_data(label, project_path=project_path):
    '''extract record data corresponding to label in tempdir and return dictionary of file_name:file_path pairs'''
    data = {}
    project = load_project(project_path)
    record = project.get_record(label)
    tempdir = tempfile.mkdtemp()
    filename = record.timestamp.strftime('%Y%m%d-%H%M%S.tar.gz')
    archive_file_path = '{}archive/{}'.format(project_path, filename)
    folder_path = '{}Data/{}'.format(project_path, label)
    if os.path.isfile(archive_file_path):
        t = tarfile.open(archive_file_path, 'r')
        t.extractall(tempdir)
    elif os.path.isdir(folder_path) and len(os.listdir(folder_path)) > 0:
        copytree(folder_path,tempdir)
    else:
        raise FileNotFoundError('neither archive file nor data folder is found for label {} {}'.format(label,folder_path))
    file_paths = {}
    for output_data in record.output_data:
        file_name = output_data.path.split('/')[-1]
        file_paths[file_name] = os.path.join(tempdir, output_data.path)
    return file_paths


def get_mse(file_paths):
    return np.genfromtxt(
        file_paths.get('mses.txt',
                       os.path.join(
                           os.path.dirname(list(file_paths.values())[0]),
                           'mses.txt')))


def get_per_corr(file_paths):
    return np.genfromtxt(
        file_paths.get('per_corr.txt',
                       os.path.join(
                           os.path.dirname(list(file_paths.values())[0]),
                           'per_corr.txt')))


def delete_files(files_dict):
    file = next(iter(files_dict.values()))
    directory = os.path.dirname(file)
    shutil.rmtree(directory)


def reconstruct_model(label, epoch, path=project_path):
    project = load_project(path)
    record = project.get_record(label)
    params = record.parameters.as_dict()

    protected_dimension = int(params['per_reg'] * params['d_hidden'])

    if 'criterion' in params:
        if params['criterion'] == 'MSE':
            problem = 'Addition'
        elif params['criterion'] == 'CrossEntropy':
            problem = 'MNIST'
    else:
        problem = params['problem']
    t_config = TrainingConfig(
        network_structure=params['network_structure'],
        d_hidden=params['d_hidden'],
        problem=problem,
        learning_rate=params['lr'],
        optimize_C=params['optimize_C'],
        protected_dimension=protected_dimension,
        tau_A=params['tau_A'],
        tau_W=params['tau_W'],
        tau_C=params.get('tau_C', 0),
        initialize_regulated=params.get('initialize_regulated', 'AWh'),
        W_regularization_scheme=params.get('W_regularization_scheme', 'R'),
        seed=params.get('seed', 666),
        provide_first_input_twice=params.get('provide_first_input_twice',
                                             True))

    model = create_model(t_config)

    files = extract_record_data(label,project_path=path)
    if epoch == 0:
        state_dict = torch.load(files['initial_model.pt'.format(epoch)],map_location={'cuda:0': 'cpu'})
    else:
        state_dict = torch.load(
            files['model_after_{:d}_epochs.pt'.format(epoch - 1)],map_location={'cuda:0': 'cpu'})
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        if t_config.network_structure != 'PLRNN':
            raise e
        else:
            old_state_dict = model.state_dict()
            if 'A.bias' in state_dict and 'W.bias' in old_state_dict:
                state_dict['W.bias'] = state_dict['A.bias']
                del state_dict['A.bias']

            if 'CLinear.weight' in state_dict and 'CLinear.weight' not in old_state_dict:
                t_config = t_config._replace(optimize_C=True)
                model = create_model(t_config)
                warnings.warn(
                    'optimize_C was enabled although parameters said it was disabled'
                )
            model.load_state_dict(state_dict)

    delete_files(files)
    return model


def convert_plrnn_to_plrnn_cell(model):
    d_hidden = model.d_hidden
    d_out = model.B.weight.size()[0]
    d_in = model.C.size()[1]
    state_dict = model.state_dict()
    optimize_C = 'CLinear.weight' in state_dict

    config = Config(T=0, d_in=d_in, d_out=d_out, d_hidden=d_hidden)
    plrnn_cell = PLRNNCell(generate_random_params(config), optimize_C)
    plrnn_cell.load_state_dict(state_dict)
    return plrnn_cell


def gradient(function, variable):
    copy_of_variable = variable
    output = function(copy_of_variable)
    assert output.dim() == 1
    assert variable.dim() == 1
    dim_output = output.size()[0]
    dim_variable = output.size()[0]
    grad = torch.zeros((dim_output, dim_variable))
    for i in range(dim_output):
        output.backward(
            torch.FloatTensor([j == i for j in range(dim_output)]),
            retain_graph=True)
        grad[i, :] = variable.grad
        variable.grad.data.zero_()
    return grad


def compute_gradient_z(model_cell, inputs, timestep):
    hidden = model_cell.initHidden()
    for t in range(timestep):
        _, hidden = model_cell(inputs[t, :], hidden)
    hidden = Variable(hidden, requires_grad=True)
    T = inputs.shape[0]

    def f(z):
        for t in range(timestep, T):
            _, z = model_cell(inputs[t, :], z)
        return z

    return gradient(f, hidden)


def visualize_gradient(model_cell, inputs, filename=None):

    T = inputs.shape[0]
    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(5)
    d_hidden = model_cell.A.weight.size()[0]
    gradients = np.zeros((T, d_hidden, d_hidden))
    for t in range(T):
        gradients[t, :, :] = compute_gradient_z(model_cell, inputs, t + 1)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    linestyles = ['-', ':', '--', '.-']
    plots = []
    for i in range(d_hidden):
        for j in range(d_hidden):
            plots.append(
                ax.plot(
                    range(1, T + 1),
                    gradients[:, j, i],
                    ls=linestyles[i],
                    c=colors[j])[0])

    ax.legend(
        plots, [''] * len(plots),
        title=r'$\frac{\partial \mathbf{z}_{T}}{\partial \mathbf{z}_{t}}$',
        ncol=d_hidden,
        handlelength=1.5,
        columnspacing=0.2,
        labelspacing=0.2)
    ax.set_xlim(1, T + 1)
    ax.set_ylabel(r'$\frac{\partial \mathbf{z}_T}{\partial \mathbf{z}_t}$')
    ax.set_xlabel('t')
    if filename:
        return plt.savefig(filename)
    plt.show()


def visualize_PLRNN_Addition_Problem(label,
                                     T,
                                     epoch,
                                     shuffle=False,
                                     filename=None,
                                     path=project_path):
    model = reconstruct_model(label, epoch,path=path)
    model_cell = convert_plrnn_to_plrnn_cell(model)
    dataloader = AdditionProblemDataset(
        T=T, train=False).get_loader(
            batch_size=300, num_workers=1, shuffle=shuffle)
    result = evaluate_model(model, dataloader)
    dataloader = AdditionProblemDataset(
        T=T, train=False).get_loader(
            batch_size=1, num_workers=1, shuffle=shuffle)
    for i, data in enumerate(dataloader):
        if i > 0:
            break
        inputs = Variable(data['inputs'])
        observations = Variable(data['observations'])
        T = inputs.size()[1]
        if model.t_config.provide_first_input_twice:
            hidden = model_cell.initHidden(inputs[:, 0, :])
        else:
            hidden = model_cell.initHidden()

        input_series = np.zeros((T, 2))
        hidden_series = np.zeros((T, model.t_config.d_hidden))
        output_series = np.zeros(T)
        for t in range(T):
            output, hidden = model_cell(inputs[:, t, :], hidden)
            input_series[t, :] = inputs[:, t, :].data.numpy()
            hidden_series[t, :] = hidden.data.numpy()
            output_series[t] = output.data.numpy()
        plt.style.use('ggplot')
        fig, ax = plt.subplots(3, 1)
        ax[0].plot(
            input_series[:, 0] * input_series[:, 1], label='effective input')
        ax[0].legend(loc='best')
        for i in range(hidden_series.shape[1]):
            ax[1].plot(hidden_series[:, i], label='{:d}'.format(i))
        ax[1].legend(loc='right', numpoints=5)
        ax[2].plot(output_series, label='output')
        ax[2].axhline(
            y=observations[0, -1].item(),
            color='r',
            linestyle='--',
            label='groundtruth')
        ax[2].legend(loc='best')
        if filename is None:
            plt.show()
        else:
            return plt.savefig(filename)
        # visualize_gradient(model_cell, inputs[0, :])


#by Joe Kington
# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def visualize_matrix(matrix, title, filename=None):
    if matrix.ndim != 2:
        matrix = matrix.reshape(-1, 1)
    fig, ax = plt.subplots()
    cmap = sns.diverging_palette(
        240, 10, as_cmap=True)  # set the colormap to soemthing diverging
    min_val, max_val = np.min(matrix), np.max(matrix)
    upper_lim = max(np.abs(min_val), np.abs(max_val))
    im = ax.matshow(
        matrix,
        cmap=cmap,
        norm=MidpointNormalize(midpoint=0., vmin=-upper_lim, vmax=upper_lim))
    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[0]):
            c = matrix[j, i]
            ax.text(i, j, '{:.2E}'.format(c), va='center', ha='center')
    ax.set_title(title)
    ax.set_axis_off()
    fig.colorbar(im)
    if not filename:
        plt.show()
    else:
        plt.savefig(filename)
        fig.clf()


def visualize_matrices_of_plrnn_model(label, epoch):
    model = reconstruct_model(label, epoch)
    visualize_matrix(np.diag(model.A.weight.data.numpy()), 'A')
    visualize_matrix(model.W.weight.data.numpy(), 'W')
    visualize_matrix(model.W.bias.data.numpy(), 'h')
    visualize_matrix(model.B.weight.data.numpy(), 'B')
    visualize_matrix(model.C.data.numpy(), 'C')
