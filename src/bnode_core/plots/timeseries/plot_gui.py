from pathlib import Path
import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import argparse
import os

def plot_gui(dataset_path: Path, n_trajectory_rows: int = 3):
    print(dataset_path)
    file = h5py.File(dataset_path, 'r')
    time = file['time'][:]
    if max(time) >10e3:
        time = time / 3600
        time_label = 'time [h]'
    else:
        time_label = 'time [s]'

    # get all outputs, states, control names in one list, map position idx to (outputs, states, controls)
    # maybe in a class
    # this class returns timeseries y and name for a given idx
    class IdxMapper():
        def __init__(self, dataset: h5py.File) -> None:
            self.data_types = ['states', 'parameters', 'outputs', 'controls']
            self.context_keys = ['train', 'common_test']
            self.data_type_exist = {}
            for data_type_key in ['states', 'parameters', 'controls', 'outputs']:
                if data_type_key in dataset['train']:
                    if not data_type_key == 'states':
                        self.data_type_exist[data_type_key] = True
                    else: # bugfix for states_hat (bnode with only states_hat predictions)
                        if not 'states_hat' in dataset['train']:
                            self.data_type_exist[data_type_key] = False
                        else:
                            self.data_type_exist[data_type_key] = True
                else:
                    self.data_type_exist[data_type_key] = False
            self.trajcetory_idx_to_data_type_and_channel_idx_map = []
            for data_type_key in self.data_types:
                if self.data_type_exist[data_type_key] and not data_type_key == 'parameters':
                    for channel_idx in range(dataset['train'][data_type_key].shape[1]):
                        channel_name = np.array(dataset[data_type_key + '_names'][:],dtype='str')[channel_idx]
                        self.trajcetory_idx_to_data_type_and_channel_idx_map.append({
                            'data_type': data_type_key,
                            'channel_name': channel_name,
                            'channel_idx': channel_idx,
                        })
            self.sample_idx_to_context_and_sample_idx_map = []
            for context_key in self.context_keys:
                for sample_idx_dataset in range(dataset[context_key]['states'].shape[0]):
                    self.sample_idx_to_context_and_sample_idx_map.append({
                        'context': context_key,
                        'sample_idx_dataset': sample_idx_dataset,
                    })
        def get_data(self, trajectory_idx: int, sample_idx: int):
            data = {}
            context = self.sample_idx_to_context_and_sample_idx_map[sample_idx]['context']
            sample_idx_in_dataset = self.sample_idx_to_context_and_sample_idx_map[sample_idx]['sample_idx_dataset']
            dataset_path = context + '/' + self.trajcetory_idx_to_data_type_and_channel_idx_map[trajectory_idx]['data_type']
            dataset_path_hat = dataset_path + '_hat'
            channel_idx = self.trajcetory_idx_to_data_type_and_channel_idx_map[trajectory_idx]['channel_idx']
            data['y'] = file[dataset_path][sample_idx_in_dataset][channel_idx]
            try:
                data['y_hat'] = file[dataset_path_hat][sample_idx_in_dataset][channel_idx]
            except KeyError:
                data['y_hat'] = None
            data['name'] = self.trajcetory_idx_to_data_type_and_channel_idx_map[trajectory_idx]['channel_name']
            data['dataset_path'] = dataset_path
            data['channel_idx'] = channel_idx
            data['ymin'] = file[dataset_path][:][:, channel_idx].min()
            data['ymax'] = file[dataset_path][:][:, channel_idx].max()
            data['context'] = context
            data['sample_idx_dataset'] = sample_idx_in_dataset
            return data
        
        def get_parameters(self, sample_idx: int):
            data = {}
            context = self.sample_idx_to_context_and_sample_idx_map[sample_idx]['context']
            sample_idx_in_dataset = self.sample_idx_to_context_and_sample_idx_map[sample_idx]['sample_idx_dataset']
            data['names'] = np.array(file['parameters_names'][:], dtype='str')
            data['parameters'] = file[context]['parameters'][sample_idx_in_dataset][:]
            return data

    idx_mapper = IdxMapper(file)

    # create figure
    n_rows = n_trajectory_rows + 1 if idx_mapper.data_type_exist['parameters'] else n_trajectory_rows
    fig, ax = plt.subplots(n_rows, 1, figsize=(10, 13))
    # fig.canvas.manager.window.wm_geometry("+0+0")  # Adjust the position on screen
    # make space for sliders
    plt.subplots_adjust(hspace=0.3)
    # reduce space over ax[0]
    plt.subplots_adjust(top=0.90)

    # first n_rows-1 plots shall share the x axis
    for i in range(n_trajectory_rows):
        ax[i].sharex(ax[0])
    # last plot shall have its own x axis
    ax[n_trajectory_rows-1].set_xlabel(time_label)

    # define one slider to choose the sample and plot which dataset it is next to it:
    context_keys = ['train', 'common_test']

    # define sample slider
    axcolor = 'lightgoldenrodyellow'
    # position slider below last plot
    pos = ax[0].get_position()
    sample_slider_ax = plt.axes([pos.x0, pos.y0+0.18 , pos.width, 0.03], facecolor=axcolor)
    sample_slider = Slider(sample_slider_ax, 'sample', 0, len(idx_mapper.sample_idx_to_context_and_sample_idx_map)-1, valinit=0, valstep=1)

    lines = [] # this will be a list of lists: n_rows, [truth, prediction]
    trajectory_sliders = []
    trajectory_update_functions = []
    for i in range(n_trajectory_rows):
        # plot truth and prediction
        data = idx_mapper.get_data(0, 0)
        truth_line = ax[i].plot(time, data['y'], label='truth', color='red', linestyle='dashed')
        prediction_line = ax[i].plot(time, data['y_hat'], label='prediction', color='blue')
        ax[i].set_ylabel(data['name'], fontsize=6)
        ax[i].legend()
        ax[i].set_ylim(data['ymin'], data['ymax'])
        lines.append([truth_line, prediction_line])
        # position each slider below the corresponding plot
        pos = ax[i].get_position()
        slider_ax = plt.axes([pos.x0, pos.y0 - 0.05, pos.width, 0.03])
        # define a slider for each trajectory plot
        trajectory_sliders.append(Slider(slider_ax, 'trajectory', 0, len(idx_mapper.trajcetory_idx_to_data_type_and_channel_idx_map)-1, valinit=0, valstep=1))
    
    # define update function for each slider
    def update_trajectory(val, i_row):
        print('update trajectory with trajectoy_idx = {}, sample_idx = {}'.format(val, sample_slider.val))
        trajectory_idx = int(val)
        sample_idx = int(sample_slider.val)
        data = idx_mapper.get_data(trajectory_idx, sample_idx)
        lines[i_row][0][0].set_ydata(data['y'])
        lines[i_row][1][0].set_ydata(data['y_hat'])
        ax[i_row].set_ylim(data['ymin'], data['ymax'])
        ax[i_row].set_ylabel(data['name'], fontsize=6)
        ax[i_row].legend()
        fig.canvas.draw_idle()

    for i_row in range(n_trajectory_rows):
        trajectory_sliders[i_row].on_changed(lambda val, i_row=i_row: update_trajectory(val, i_row))

    # make bar chart for parameters
    if idx_mapper.data_type_exist['parameters']:
        parameters = idx_mapper.get_parameters(0)
        # get min and max of parameters
        param_mins = file['train']['parameters'][:].min(axis=0)
        param_maxs = file['train']['parameters'][:].max(axis=0)
        def normalize_parameters(params):
            return (params - param_mins) / (param_maxs - param_mins)
        bar_line = ax[n_trajectory_rows].bar(parameters['names'], normalize_parameters(parameters['parameters']))
        ax[n_trajectory_rows].set_xticklabels(parameters['names'], rotation=15)
        ax[n_trajectory_rows].set_ylim(0, 1)
    # define update functions:

    class sample_updater():
        def __init__(self) -> None:
            self.text_on_bars = []
            self.text_sample_idx = None
            self.sample_idx_changer = None
        # trajectory update --> per plot
        # sample update --> for all plots
        def update_sample(self, val):
            sample_idx = int(val)
            if self.sample_idx_changer != None:
                self.sample_idx_changer.sample_idx = sample_idx
            for i in range(n_trajectory_rows):
                trajectory_idx = int(trajectory_sliders[i].val)
                update_trajectory(trajectory_idx, i)
            if idx_mapper.data_type_exist['parameters']:
                parameters = idx_mapper.get_parameters(sample_idx)
                params_normalized = normalize_parameters(parameters['parameters'])
                for rect, true_height in zip(bar_line, params_normalized):
                    rect.set_height(true_height)
                ax[n_trajectory_rows].set_xticklabels(parameters['names'], rotation=15)
                # set text for each bar
                for item in self.text_on_bars:
                    item.remove()
                self.text_on_bars = []
                for rect, true_height in zip(bar_line, parameters['parameters']):
                    height = rect.get_height()
                    self.text_on_bars.append(ax[n_trajectory_rows].text(rect.get_x() + rect.get_width()/2., 1.05*height,
                            '{:.2f}'.format(true_height), ha='center', va='bottom'))
            
            context = idx_mapper.get_data(0, sample_idx)['context']
            sample_idx_in_dataset = idx_mapper.get_data(0, sample_idx)['sample_idx_dataset']
            string = 'Context: {}, idx_in_dataset: {}'.format(context, sample_idx_in_dataset)
            if self.text_sample_idx != None:
                self.text_sample_idx.set_text(string)
            else:
                self.text_sample_idx = ax[0].text(0.5, 1.2, string, ha='center', va='bottom', transform=ax[0].transAxes, fontsize=14)
    sample_updater = sample_updater()
    sample_slider.on_changed(sample_updater.update_sample)
    sample_updater.update_sample(0)

    # add a forward and backward button to change the sample idx
    class sample_idx_changer():
        def __init__(self) -> None:
            self.sample_idx = 0
        def forward(self, event):
            self.sample_idx = self.sample_idx + 1 if self.sample_idx < len(idx_mapper.sample_idx_to_context_and_sample_idx_map) - 1 else self.sample_idx
            sample_slider.set_val(self.sample_idx)
        def backward(self, event):
            self.sample_idx = self.sample_idx - 1 if self.sample_idx > 0 else self.sample_idx
            sample_slider.set_val(self.sample_idx)
    
    sample_idx_changer = sample_idx_changer()
    sample_updater.sample_idx_changer = sample_idx_changer
    ax_sample_idx_changer_backward = plt.axes([0.0, 0.05, 0.05, 0.05])
    ax_sample_idx_changer_forward = plt.axes([0.95, 0.05, 0.05, 0.05])
    button_sample_idx_changer_forward = plt.Button(ax_sample_idx_changer_forward, 'Forward')
    button_sample_idx_changer_forward.on_clicked(sample_idx_changer.forward)
    button_sample_idx_changer_backward = plt.Button(ax_sample_idx_changer_backward, 'Backward')
    button_sample_idx_changer_backward.on_clicked(sample_idx_changer.backward)

    plt.show()
    pass

# add parser for file path

parser = argparse.ArgumentParser()
parser.add_argument_group('dataset_path', 'path to dataset')
parser.add_argument('--dataset_path', type=str, help='path to dataset',
                    default=None)
parser.add_argument('-n', '--n_trajectory_rows', type=int, help='number of trajectory rows to plot', default=3)

def main():
    args = parser.parse_args()
    if args.dataset_path == None:
        args.dataset_path = input('Enter path to dataset: ')
        _path = Path(args.dataset_path)
    if args.dataset_path.startswith('file://'):
        _path = Path(args.dataset_path.replace('file://',''))
    else:
        _path = Path(args.dataset_path)
        if not _path.exists():
            raise FileNotFoundError(f"The dataset path {args.dataset_path} can not be opened.")
    print('opening dataset at {}'.format(_path))
    plot_gui(_path, n_trajectory_rows=args.n_trajectory_rows)

if __name__ == '__main__':
    main()