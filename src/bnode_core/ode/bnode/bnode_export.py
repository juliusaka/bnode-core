import hydra
from pathlib import Path
import logging
import mlflow
import yaml
import h5py
import torch
from omegaconf import OmegaConf

import bnode_core.filepaths as filepaths
from bnode_core.ode.trainer import initialize_model
from bnode_core.nn.nn_utils.load_data import make_stacked_dataset
from bnode_core.config import onnx_export_config_class

# How should this work?


# load model configuration (yaml) + a corresponding dataset (Optional?)
# could automatically get this from mlflow information?
# could use hydra for this to have multirun possibility

# provide output directory (hydra output?)

# initialize model as in trainer.py (or reuse everything there), but then break out and get the model here

# set everything to eval

# then export from with a for loop over a dict ['encoder': self.xxx, 'decoder']


def load_trained_latent_ode(cfg_export):
    # get artifacts directory
    if cfg_export.mlflow_run_id is not None:
        mlflow.set_tracking_uri(cfg_export.mlflow_tracking_uri)
        dir_artifacts = filepaths.filepath_from_ml_artifacts_uri(mlflow.get_run(cfg_export.mlflow_run_id).info.artifact_uri)
    else:
        dir_artifacts = filepaths.filepath_from_local_or_ml_artifacts(cfg_export.model_directory)
    logging.info('Resolved artifacts uri as {}'.format(str(dir_artifacts)))
    if cfg_export.config_path is None:
        path_config = dir_artifacts / '.hydra' / 'config_validated.yaml'
    else: 
        path_config = filepaths.filepath_from_local_or_ml_artifacts(cfg_export.config_path)
    if cfg_export.dataset_path is None:
        path_dataset = dir_artifacts / 'dataset.hdf5'
    else:
        path_dataset = filepaths.filepath_from_local_or_ml_artifacts(cfg_export.dataset_path)

    # load config (and validate it using the dataclass?)
    with open(path_config) as file:
        cfg_dict = yaml.load(file, Loader=yaml.FullLoader)
        cfg = OmegaConf.create(cfg_dict)
        cfg.use_cuda = False
    logging.info('Loaded config of BNODE: {}'.format(str(cfg)))
    
    # load training dataset
    if path_dataset.is_file():
        dataset_file = h5py.File(path_dataset, 'r')
    else:
        raise FileNotFoundError(f'Dataset file {path_dataset} not found. Please provide a valid dataset path.')
    dataset = make_stacked_dataset(dataset_file, 'train')
    model = initialize_model(cfg, train_dataset=dataset, hdf5_dataset=None, 
                             initialize_normalization=False, model_type='bnode')
    
    # load latest checkpoint
    if cfg_export.model_checkpoint_path is None:
        path_checkpoint = sorted(dir_artifacts.rglob('model_phase_*.pt'))[-1]
    else:
        path_checkpoint = filepaths.filepath_from_local_or_ml_artifacts(cfg_export.model_checkpoint_path)
    model.load(path_checkpoint, device='cpu')
    return {'model': model, 'cfg': cfg, 'dataset_file': dataset_file, 'dataset': dataset}

def export_example_io_data(res, inputs, path_example_io):
    with h5py.File(path_example_io, 'w') as f:
        # Save inputs
        grp_in = f.create_group('inputs')
        for in_key, in_val in inputs.items():
            if isinstance(in_val, torch.Tensor):
                grp_in.create_dataset(in_key, data=in_val.detach().numpy())
            elif in_val is not None:
                grp_in.create_dataset(in_key, data=in_val)
        # Save outputs
        grp_out = f.create_group('outputs')
        if isinstance(res, torch.Tensor):
            grp_out.create_dataset('output', data=res.detach().cpu().numpy())
        elif isinstance(res, (tuple, list)):
            for i, out_val in enumerate(res):
                grp_out.create_dataset(f'output_{i}', data=out_val.detach().cpu().numpy())
        elif isinstance(res, dict):
            for out_key, out_val in res.items():
                grp_out.create_dataset(out_key, data=out_val.detach().cpu().numpy())

def log_shapes_of_dict(d, name=''):
    if name:
        logging.info(f"Shapes in {name}:")
    else:
        logging.info("Shapes in .... :")
    if isinstance(d, dict):
        for key, value in d.items():
            if isinstance(value, torch.Tensor):
                logging.info(f"\t{key}: {value.shape}")
            elif isinstance(value, (tuple, list)):
                logging.info(f"\t{key}: {[v.shape for v in value if isinstance(v, torch.Tensor)]}")
            else:
                logging.info(f"\t{key}: {value}")
    elif isinstance(d, (tuple, list)):
        for i, value in enumerate(d):
            if isinstance(value, torch.Tensor):
                logging.info(f"\t[{i}]: {value.shape}")
            elif isinstance(value, (tuple, list)):
                logging.info(f"\t[{i}]: {[v.shape for v in value if isinstance(v, torch.Tensor)]}")
            else:
                logging.info(f"\t[{i}]: {value}")
    else:
        logging.info(f"\t{type(d)}: {d}")

@hydra.main(config_path=str(Path('conf').absolute()), config_name="onnx_export", version_base=None)
def main(cfg_export: onnx_export_config_class):
    logging.info('Exporting BNODE using the following config {}'.format(str(cfg_export)))
    
    # load model
    res = load_trained_latent_ode(cfg_export)
    model, cfg, dataset_file, dataset = res['model'], res['cfg'], res['dataset_file'], res['dataset']
    model.eval()

    # determine output dir
    dir_output = cfg_export if cfg_export.output_dir is not None else filepaths.dir_current_hydra_output()

    # export bnode config
    path_config = dir_output / 'bnode_config.yaml'
    logging.info(f'Exporting BNODE config to {path_config}')
    with open(path_config, 'w') as f:
        yaml.dump(OmegaConf.to_container(cfg, resolve=True), f, default_flow_style=False)

    # get test points for graph construction
    test_state = dataset[0]['states'][:,0].unsqueeze(0)
    test_control = dataset[0]['controls'][:,0].unsqueeze(0) if model.include_controls else None
    test_parameters = dataset[0]['parameters'].unsqueeze(0) if model.include_parameters else None



    # export the encoders
    encoders = {'states': model.state_encoder, 
                'controls': model.controls_encoder if model.include_controls else None,
                'parameters': model.parameter_encoder if model.include_params_encoder else None
            }
    # construct test inputs for graph construction
    inputs_dict = {
        'states': {'x': test_state},
        'controls': {'x': test_control} if model.params_to_control_encoder is False else {'x': test_control, 'params': test_parameters},
        'parameters': {'x': test_parameters},
    }
    # handling of additional inputs to state encoder
    if model.params_to_state_encoder is True:
        inputs_dict['states']['params'] = test_parameters
    if model.controls_to_state_encoder is True:
        inputs_dict['states']['controls'] = test_control
    
    latents_dict = {}
    for key, encoder in encoders.items():
        if encoder is not None:
            path_encoder = dir_output / f'encoder_{key}.onnx'
            logging.info(f'Exporting {key} encoder to {path_encoder}')
            # test model
            log_shapes_of_dict(inputs_dict[key], f'Inputs for {key} encoder')
            res = encoder(**inputs_dict[key])
            log_shapes_of_dict(res, f'Outputs of {key} encoder')
            logging.info(f'Test result {res}')
            # export
            input_names = list(inputs_dict[key].keys())
            output_names=['latent_' + key + '_mu', 'latent_' + key + '_logvar']
            dynamic_axes={}
            for name in input_names:
                dynamic_axes[name] = {0: 'batch_size'}
            for name in output_names:
                dynamic_axes[name] = {0: 'batch_size'}
            torch.onnx.export(encoder, args=inputs_dict[key], f=path_encoder, 
                              input_names=input_names,
                              output_names=output_names,
                              dynamic_axes=dynamic_axes
            )
            logging.info(f'Exported {key} encoder successfully')
            # export also example io
            path_example_io = dir_output / f'encoder_{key}_example_io.hdf5'
            export_example_io_data(res, inputs_dict[key], path_example_io)
            # save latent variable
            latents_dict[key] = res[0] # the first is mu

    # export ssm from parameters model and get A_from_param and B_from_param for the latent ODE function
    ode = model.latent_ode_func
    if ode.include_parameters is True and ode.linear is True:
        # this is only possible if the model is linear and has parameters
        logging.info('Exporting SSM from parameters')
        ssm = model.latent_ode_func.ssm_from_param
        path_ssm = dir_output / 'latent_ode_ssm_from_param.onnx'
        logging.info(f'Export latent ODE SSM from parameters to {path_ssm}')
        # construct test input
        inputs = {
            'lat_parameters': latents_dict['parameters'],
        }
        # test model
        log_shapes_of_dict(inputs, 'Inputs for latent ODE SSM from parameters')
        res = ssm(**inputs)
        log_shapes_of_dict(res, 'Outputs of latent ODE SSM from parameters')
        logging.info(f'Test result {res}')
        # export
        input_names=['lat_parameters']
        output_names=['A', 'B'] if ode.include_controls else ['A']
        dynamic_axes={}
        for name in input_names:
            dynamic_axes[name] = {0: 'batch_size'}
        for name in output_names:
            dynamic_axes[name] = {0: 'batch_size'}
        torch.onnx.export(ssm, args=inputs, f=path_ssm, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)
        logging.info(f'Exported latent ODE SSM from parameters successfully')
        # get A_from_param and B_from_param
        if ode.include_controls:
            A_from_param, B_from_param = res
        else:
            A_from_param = res

    # export the latent ode function
    path_ode = dir_output / 'latent_ode.onnx'
    logging.info(f'Export latent ODE to {path_ode}')
    # construct test input
    inputs = {
        'lat_states': latents_dict['states'],
        'lat_parameters': latents_dict['parameters'] if ode.include_parameters is True else None,
        'lat_controls': latents_dict['controls'] if ode.include_controls is True else None,
        'A_from_param': A_from_param if ode.include_parameters is True and ode.linear else None,
        'B_from_param': B_from_param if ode.include_parameters is True and ode.linear and ode.include_controls else None,
    }
    # test model
    log_shapes_of_dict(inputs, 'Inputs for latent ODE')
    res = ode(**inputs)
    log_shapes_of_dict(res, 'Outputs of latent ODE')
    logging.info(f'Test result {res}')
    # export
    input_names=[]
    for key in inputs.keys():
        if inputs[key] != None:
            input_names.append(key)
    dynamic_axes={}
    for name in input_names:
        dynamic_axes[name] = {0: 'batch_size'}
    if model.lat_ode_type == 'variance_constant' or model.lat_ode_type == 'vanilla':
        output_names = ['lat_states_mu_dot']
        dynamic_axes['lat_states_mu_dot'] = {0: 'batch_size'}
    elif model.lat_ode_type == 'variance_dynamic':
        output_names = ['concat(lat_states_mu_dot,lat_states_logvar_dot)']
        dynamic_axes['concat(lat_states_mu_dot,lat_states_logvar_dot)'] = {0: 'batch_size'}
    torch.onnx.export(ode, args=inputs, f=path_ode, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)
    logging.info(f'Exported latent ODE successfully')
    # export also example io
    path_example_io = dir_output / f'latent_ode_example_io.hdf5'
    export_example_io_data(res, inputs, path_example_io)

    # export the decoder
    # TODO: What to with split return? because we have to tak the first n elements for states etc now... implement other function for this? / optional argument?
    decoder = model.decoder
    decoder.onnx_export = True  # disable concatenation of outputs for ONNX export
    path_decoder = dir_output / 'decoder.onnx'
    logging.info(f'Export decoder to {path_decoder}')
    # construct test input
    inputs = {
        'lat_state': latents_dict['states'],
        'lat_parameters': latents_dict['parameters'] if decoder.include_parameters is True else None,
        'lat_controls': latents_dict['controls'] if decoder.include_controls is True else None,
    }
    # test model
    log_shapes_of_dict(inputs, 'Inputs for decoder')
    res = decoder(**inputs)
    log_shapes_of_dict(res, 'Outputs of decoder')
    logging.info(f'Test result {res}')
    input_names = []
    # export
    for key in inputs.keys():
        if inputs[key] != None:
            input_names.append(key)
    if decoder.include_outputs and decoder.include_states:
        output_names = ['states', 'outputs']
    elif decoder.include_outputs:
        output_names = ['outputs']
    elif decoder.include_states:
        output_names = ['states']
    dynamic_axes={}
    for name in input_names:
        dynamic_axes[name] = {0: 'batch_size'}
    for name in output_names:
        dynamic_axes[name] = {0: 'batch_size'}
    torch.onnx.export(decoder, args=inputs, f=path_decoder, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)
    logging.info(f'Exported decoder successfully')
    # export also example io
    path_example_io = dir_output / f'decoder_example_io.hdf5'
    export_example_io_data(res, inputs, path_example_io)

if __name__ == '__main__':
    main()