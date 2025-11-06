import h5py
import torch
from pathlib import Path
import logging
from omegaconf import OmegaConf
from bnode_core.config import base_pModelClass, train_test_config_class
from bnode_core.filepaths import filepath_dataset_config_from_name, filepath_dataset_from_config

def load_validate_dataset_config(path: Path):
    """ Load and validate dataset configuration from given path.

    Args:
        path (Path): Path to the dataset configuration file.
    Returns:
        dataset_config (base_pModelClass): Loaded and validated dataset configuration.
    """

    if not path.exists():
        raise FileNotFoundError('Dataset config file not found: {}'.format(path))

    logging.info('Loading dataset config file: {}'.format(path))
    _dataset_config_dict = OmegaConf.load(path)
    _dataset_config_dict = OmegaConf.to_object(_dataset_config_dict) # make dict
    dataset_config = base_pModelClass(**_dataset_config_dict) # validate
    logging.info('Validated dataset config file: {}'.format(path))
    return dataset_config


def load_dataset_and_config(dataset_name: str, dataset_path: str ):
    """ Load dataset and its configuration from given dataset name.
    
    Args:
        dataset_name (str): Name of the dataset. If it is a file path, the dataset is loaded from that path.
    Returns:
        dataset (h5py.File): Loaded HDF5 dataset.
        dataset_config (base_pModelClass or None): Loaded and validated dataset configuration, or None if no config file is found.
    
    """
    _path = filepath_dataset_from_config(dataset_name, dataset_path)

    dataset = h5py.File(_path, 'r')
    logging.info('Loaded dataset from file: {}'.format(_path))

    _path = filepath_dataset_config_from_name(dataset_name)
    if not _path.exists():
        logging.info('No dataset config file found, using information from dataset file')
        dataset_config = None
    else:
        dataset_config = load_validate_dataset_config(_path)
    return dataset, dataset_config

def make_stacked_dataset(dataset: h5py.File, context: str, seq_len_from_file: int = None, seq_len_batches: int = None):
    '''
    returns toch.utils.data.StackDataset Dataset, which returns a dict when iterated

    Args:
        dataset (h5py.File): dataset
        context (str): context for dataset, either 'train', 'test', 'validation', 'common_test', 'common_validation'
        seq_len (int): sequence length for dataset, if None, use dataset_config.RawData.seq_len. Else, Data will be reduced to seq_len
    Returns:
        torch_dataset (torch.utils.data.StackDataset): dataset which returns a dict when iterated
    '''
    assert context in ['train', 'test', 'validation', 'common_test', 'common_validation'], 'context must be one of train, test, validation, common_test, common_validation'

    # get tensors of dataset
    time = dataset['time'][:]
    states = dataset[context]['states'][:]
    states_der = dataset[context]['states_der'][:] if 'states_der' in dataset[context].keys() else None
    parameters = dataset[context]['parameters'][:] if 'parameters' in dataset[context].keys() else None
    controls = dataset[context]['controls'][:] if 'controls' in dataset[context].keys() else None
    outputs = dataset[context]['outputs'][:] if 'outputs' in dataset[context].keys() else None

    # cut data from file to seq_len
    if seq_len_from_file is not None:
        time = time[:seq_len_from_file]
        states = states[:,:,:seq_len_from_file]
        states_der = states_der[:,:,:seq_len_from_file] if states_der is not None else None # TODO: add finite difference calculation for states_der and cfg.dataset_prep entries to say if derivatives are included
        parameters = parameters[:] if parameters is not None else None
        controls = controls[:,:,:seq_len_from_file] if controls is not None else None
        outputs = outputs[:,:,:seq_len_from_file] if outputs is not None else None

    # define wrapper to delete nones from kwargs dict
    def _delete_nones(**kwargs):
        kwargs = dict((k,v) for k,v in kwargs.items() if v is not None)
        return kwargs

    # make torch dataset with dict as output
    dataset_type = torch.utils.data.StackDataset if seq_len_batches is None else lambda *args, **kwargs: TimeSeriesDataset(seq_len_batches, *args, **kwargs)
    torch_dataset = dataset_type(
        **_delete_nones(
            time = torch.tensor(time, dtype=torch.float32).unsqueeze(0).expand(states.shape[0], -1).unsqueeze(1),
            states = torch.tensor(states, dtype=torch.float32),
            states_der = torch.tensor(states_der, dtype=torch.float32) if states_der is not None else None,
            parameters = torch.tensor(parameters, dtype=torch.float32) if parameters is not None else None,
            controls = torch.tensor(controls, dtype=torch.float32) if controls is not None else None,
            outputs = torch.tensor(outputs, dtype=torch.float32) if outputs is not None else None,
        )
    )
    logging.info('Created {} with {} and sequence length {}'.format(type(torch_dataset),torch_dataset.datasets.keys(), torch_dataset.datasets['time'].shape[2]))
    return torch_dataset

class TimeSeriesDataset(torch.utils.data.StackDataset):
    """
    This dataset is used for time series data, to return only a part of the whole sequence.
    The sequences are saved in full length, but only a part of the sequence is returned.
    The mapping is saved in a dict.
    We assume the datasets have dim=3 and more, if they are timeseries. 
    dim=0 is sample_number, dim=1 is channel_number, dim=2 is timerseries_position
    """
    def __init__(self, seq_len: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.datasets, dict), "can only handle dict style stacked datasets with one key-value pair time"
        assert 'time' in self.datasets.keys(), "need one dataset with key time to define the map" 
        self._length_old = self._length
        if seq_len > self.datasets['time'].shape[2]:
            Warning("seq_len is {}, setting to len of timeseries".format(seq_len))
            seq_len = self.datasets['time'].shape[2]
        self.seq_len = seq_len
        self.initialize_map(seq_len)

    def set_seq_len(self, seq_len: int):
        """
        Function to call to change the seq_len.
        """
        if seq_len == None or seq_len == 0 or seq_len > self.datasets['time'].shape[2]:
            Warning("seq_len is {}, setting to len of timeseries".format(seq_len))
            seq_len = self.datasets['time'].shape[2]
        self.seq_len = seq_len
        self.initialize_map(seq_len)

    def initialize_map(self, seq_len: int):
        """
        Function to call to initialize the map. This can be called again to change the seq_len.
        """
        assert seq_len > 0, "seq_len must be at least 1"
        # define map
        n_batches_per_sample = (self.datasets['time'].shape[2] - (seq_len - 1))
        n_batches_total = n_batches_per_sample * self._length_old
        self.mapping = [[] for i in range(n_batches_total)]
        self._length = n_batches_total
        # fill out map. mapping shall contain [n_sample, start_position, stop_position]
        k_stop=seq_len # stop position in sequence
        j=0 # sample position in datasets
        for i in range(n_batches_total):
            self.mapping[i] = [j, k_stop-seq_len, k_stop]
            if k_stop + 2 > n_batches_per_sample + seq_len: # if the over next sequence would be out of bounds, go to next sample (+1 more because of > and not >=)
                k_stop = seq_len
                j += 1
            else:
                k_stop += 1
        #self.mapping = torch.tensor(np.array(self.mapping), dtype=torch.int64)


    def __getitem__(self,index):
        i, k_start, k_stop = self.mapping[index]
        ret_val = {}
        for key, value in self.datasets.items():
            if value.ndim == 2:
                ret_val[key] = value[i, :]
            else:
                ret_val[key] = value[i, :, k_start:k_stop]
        return ret_val
    
    def __getitems__(self, indices):
        # print('getitems function not tested yet, necessary torch version >= 2.2')
        # see https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#StackDataset
        # __getitems__ was not implemented in torch.utils.data.StackDataset for 2.1
        samples = [None] * len(indices)
        for i, index in enumerate(indices):
            samples[i] = self.__getitem__(index)
        return samples
    
    def __len__(self):
        return self._length