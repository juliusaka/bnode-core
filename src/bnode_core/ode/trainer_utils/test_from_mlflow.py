"""Test trained models from MLflow runs on new datasets.

This module provides functionality to load trained neural ODE models from MLflow runs
and test them on different datasets. It downloads artifacts directly from the MLflow
server (no local file access required) and executes validation runs.

You can use all configuration options from trainer.py to override parameters for testing.

Typical Usage Example
---------------------
Test a single run on a new dataset using experiment_id (recommended):

```python
    python test_from_mlflow.py \\
        experiment_id=123456789 \\
        run_name=bemused-hen-59 \\
        # or run_id=8c2c32b9407a4e20946f72cd1c714776 \\
        dataset_name=myTestData \\
        mlflow_experiment_name=validation_results \\
        mlflow_tracking_uri=http://localhost:5000 \\
        n_processes=1 \\
        nn_model_base=latent_ode_base \\
        # specify overrides:
        override nn_model.training.batch_size_test=128 \\
        override nn_model.training.test_save_internal_variables=false \\
        override n_workers_train_loader=1 \\
        override n_workers_other_loaders=1 \\
        override use_cuda=true
```

Command Line Arguments
----------------------
Required:

    dataset_name : str or list[str]
        Name(s) of dataset(s) to test on. Comma-separated for multiple datasets.
    mlflow_experiment_name : str
        Name for the new MLflow experiment where test results will be logged.
    nn_model_base : str
        Base configuration for the neural network model (e.g., 'latent_ode_base').

Run Selection (one of):

    run_id : str or list[str]
        MLflow run ID(s) to test. Comma-separated for multiple runs.
    experiment_id + run_name : str
        Experiment ID and specific run name(s) within that experiment (recommended).
    experiment_id : str
        Experiment ID to test all runs from that experiment.
    experiment : str (deprecated)
        Experiment name - triggers warning as multiple experiments can share names.

Optional:

    mlflow_tracking_uri : str
        URI of MLflow tracking server. Defaults to 'http://localhost:5000'.
    n_processes : int
        Number of parallel processes for testing. Defaults to 1 (sequential).
    override <key>=<value> : 
        Override specific config parameters (e.g., 'override use_cuda=false').

Notes
-----
- Artifacts are downloaded from MLflow server to Hydra output directory.
- Downloaded artifacts are stored in {hydra_output}/mlflow_test_artifacts/run_{run_id}/.
- Artifacts persist after testing for inspection and debugging.
- Model checkpoints are automatically retrieved from MLflow artifact storage.
- Results are logged to a new MLflow experiment specified by mlflow_experiment_name.
- When testing multiple runs/datasets, a Cartesian product is created (all combinations).
- Use experiment_id instead of experiment name to avoid ambiguity.

See Also
--------
bnode_core.ode.trainer : Main training module with train_all_phases function.
mlflow : MLflow documentation for run and artifact management.
"""

# Idea: call train_all_phases from trainer.py with config loaded from mlflow-run-id
# Change parameters previously to load old model (with sequence length), put it to test mode 
# and use a different dataset. 


# read the mlflow-run-id from the command line
# read the dataset from the command line
# recognize possibly multirun-command (if "," and "-m" present in the command line)

# load the config from the mlflow-run-id
# update the config with the dataset, the last sequence length, the model path, the test mode, the new experiment name
# save new config to a temporary file

# call train_all_phases with the new config (such that hydra is used for validation)

# remove the temporary file

# Helpful mlflow commands:
# mlflow.get_run(run_id: str)
# mlflow.get_artifact_uri(run_id: str)


    # mlflow.set_tracking_uri("http://localhost:5000")
    # experiment = mlflow.get_experiment("232112805895166389")
    # runs = mlflow.search_runs(experiment.experiment_id)
    # # or run = mlflow.get_run("8c2c32b9407a4e20946f72cd1c714776")

import mlflow
import sys
import argparse
import logging
import shutil
import bnode_core.filepaths as filepaths
import yaml
import pathlib
import subprocess
import hydra
from bnode_core.ode.trainer import train_all_phases
from multiprocessing import Pool
import warnings
from pathlib import Path
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Test trained models from MLflow runs on new datasets.")
    parser.add_argument('--dataset_name', required=True, type=str,
                        help='Name(s) of dataset(s) to test on. Comma-separated for multiple datasets.')
    parser.add_argument('--mlflow_experiment_name', required=True, type=str,
                        help='Name for the new MLflow experiment where test results will be logged.')
    parser.add_argument('--nn_model_base', required=True, type=str,
                        help='Base configuration for the neural network model (e.g., latent_ode_base).')
    parser.add_argument('--mlflow_tracking_uri', type=str, default='http://localhost:5000',
                        help='URI of MLflow tracking server.')
    parser.add_argument('--n_processes', type=int, default=1,
                        help='Number of parallel processes for testing.')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run_id', type=str,
                       help='MLflow run ID(s) to test. Comma-separated for multiple runs.')
    group.add_argument('--experiment_id', type=str,
                       help='Experiment ID to test all runs from that experiment or with --run_name.')
    group.add_argument('--experiment', type=str,
                       help='Experiment name (deprecated, triggers warning).')

    parser.add_argument('--run_name', type=str,
                        help='Run name(s) within experiment_id. Comma-separated for multiple runs.')
    parser.add_argument('--override', action='append', default=[],
                        help='Override specific config parameters, e.g. --override nn_model.training.batch_size_test=128')

    args = parser.parse_args()
    # Convert comma-separated lists to Python lists
    args.dataset_name = [x.strip() for x in args.dataset_name.split(",")]
    if args.run_id:
        args.run_id = [x.strip() for x in args.run_id.split(",")]
    if args.run_name:
        args.run_name = [x.strip() for x in args.run_name.split(",")]
    return args

def parse_overrides(override_list):
    """
    Parse list of override strings into a dict of key-value pairs.
    Each override should be in the form key=value.
    """
    overrides = {}
    for item in override_list:
        if '=' not in item:
            raise ValueError(f"Override argument '{item}' is not in key=value format.")
        key, value = item.split('=', 1)
        overrides[key.strip()] = value.strip()
    return overrides

def get_run_ids(args):
    """Retrieve MLflow run IDs based on provided selection criteria.
    
    Resolves run IDs from either direct run_id specification, run_name lookup
    within an experiment, or all runs from an experiment.
    
    Args:
        command_line_args (dict): Parsed command line arguments containing one of:

            - 'run_id': Direct list of run IDs.
            - 'experiment_id' + 'run_name': Experiment ID and specific run names.
            - 'experiment_id': All runs from the experiment.
            - 'experiment': Experiment name (deprecated, triggers warning).
            
            Optional:

            - 'mlflow_tracking_uri': MLflow server URI.
    
    Returns:
        list (list[str]): List of MLflow run IDs to test.
    
    Raises:
        ValueError: If incompatible argument combinations are provided
            (e.g., both run_name and run_id, or both run_id and experiment_id).
    
    Side Effects:
        - Sets MLflow tracking URI via mlflow.set_tracking_uri().
        - Prints progress messages about run ID retrieval.
        - Issues warning if 'experiment' (name) is used instead of 'experiment_id'.
    
    Examples:
        >>> get_run_ids({'experiment_id': ['123456'], 'run_name': ['run1', 'run2']})
        ['abc123', 'def456']
    """
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    if args.run_id:
        return args.run_id
    if args.experiment:
        warnings.warn(
            "Using 'experiment' (name) is deprecated. Multiple experiments can have the same name. "
            "Please use 'experiment_id' instead for unambiguous experiment identification.",
            DeprecationWarning,
            stacklevel=2
        )
        experiment = mlflow.get_experiment_by_name(args.experiment)
        if experiment is None:
            raise ValueError(f"Experiment with name '{args.experiment}' not found.")
        # TODO: What happens if there are multiple experiments with same name?
        experiment_id = experiment.experiment_id
    else:
        experiment_id = args.experiment_id
    runs = mlflow.search_runs(experiment_id)
    if args.run_name:
        run_ids = []
        for run_name in args.run_name:
            matching_runs = runs[runs["tags.mlflow.runName"] == run_name]
            if len(matching_runs) == 0:
                raise ValueError(f"No run found with name '{run_name}' in experiment {experiment_id}")
            run_ids.append(matching_runs["run_id"].values[0])
        return run_ids
    return runs["run_id"].to_list()

def main():
    """Main execution function for testing models from MLflow runs.
    
    Orchestrates the complete workflow:
    1. Parses and validates command line arguments.
    2. Retrieves run IDs from MLflow using experiment_id (or deprecated experiment name).
    3. For each run-dataset combination:
       - Downloads artifacts from MLflow server to Hydra output directory.
       - Loads the training configuration from downloaded artifacts.
       - Updates config for testing (new dataset, model path, test mode).
       - Applies any override parameters.
       - Saves modified config to temporary files in Hydra output.
    4. Executes testing jobs either sequentially or in parallel.
    
    The function creates a Cartesian product of runs × datasets, generating
    one test job for each combination.
    
    Raises:
        ValueError: If command line arguments are invalid or incompatible.
        FileNotFoundError: If MLflow artifacts (config, model) cannot be found.
    
    Side Effects:
        - Creates mlflow_test_artifacts directory in Hydra output for configs and artifacts.
        - Downloads artifacts from MLflow server (if not local).
        - Artifacts persist after testing for inspection.
        - Launches subprocess calls to trainer.py for each test job.
        - Logs results to MLflow under the specified experiment name.
        - Prints progress messages throughout execution.
    
    Notes:
        - Model checkpoints are retrieved from the final training phase.
        - Sequence length is set to match the last training phase.
        - Original training dataset name is preserved for reference.
        - Artifacts are organized as: {hydra_output}/mlflow_test_artifacts/run_{run_id}/
        - Use experiment_id instead of experiment name to avoid ambiguity warnings.
    
    Examples:
        Command line usage::
        
            python test_from_mlflow.py \\
                experiment_id=123456789 \\
                run_name=final-model-123 \\
                dataset_name=validation_set \\
                mlflow_experiment_name=validation_results \\
                nn_model_base=latent_ode_base \\
                n_processes=1
    """
    args = parse_args()
    overrides = parse_overrides(args.override)
    run_ids = get_run_ids(args)
    dataset_names = args.dataset_name

    # get temporary directory in Hydra output folder for saving config files and artifacts
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    hydra_output_dir = Path.cwd() / "outputs" / date_str / time_str
    hydra_output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir_path = hydra_output_dir / "mlflow_test_artifacts"
    temp_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"using directory for artifacts and config files: {temp_dir_path}")
    # create nn_model directory
    (temp_dir_path / "nn_model").mkdir(parents=True, exist_ok=True)

    jobs = 0
    training_dataset_list = []
    artifact_dirs = {}  # Store artifact directories per run_id
    
    for run_id in run_ids:
        for dataset in dataset_names:
            # Download artifacts from MLflow server
            mlflow_run = mlflow.get_run(run_id)
            artifact_uri = mlflow_run.info.artifact_uri
            
            # Check if we need to download artifacts (not already local)
            if run_id not in artifact_dirs:
                if not artifact_uri.startswith('file://'):
                    # Download artifacts from remote MLflow server
                    run_artifact_dir = temp_dir_path / f"run_{run_id}"
                    run_artifact_dir.mkdir(parents=True, exist_ok=True)
                    print(f"Downloading artifacts for run {run_id} from MLflow server to {run_artifact_dir}")
                    
                    mlflow.artifacts.download_artifacts(
                        run_id=run_id,
                        dst_path=str(run_artifact_dir)
                    )
                    artifact_dirs[run_id] = run_artifact_dir
                    print(f"Successfully downloaded artifacts to {run_artifact_dir}")
                else:
                    # Local artifacts - use direct path
                    artifact_dirs[run_id] = Path(artifact_uri.replace('file://',''))
                    print(f"Using local artifacts from {artifact_dirs[run_id]}")
            
            # Get config path from downloaded/local artifacts
            _config_path = artifact_dirs[run_id] / ".hydra" / "config_validated.yaml"
            
            if not _config_path.exists():
                raise FileNotFoundError(f"Config file not found at {_config_path}")

            # copy config to temporary directory
            temp_config = temp_dir_path / f"config_{jobs}.yaml"
            print(f"copying config from {_config_path} to {temp_config}")
            shutil.copy(_config_path, temp_config)
            
            # update config with dataset_name, sequence_length, model_path, test_mode
            # open yaml file
            with open(temp_config) as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
            
            # update config
            training_dataset_list.append(config["dataset_name"])
            config["dataset_name"] = dataset
            config["mlflow_experiment_name"] = args.mlflow_experiment_name
            # TODO: is this important
            config["nn_model"]["training"]["pre_trained_model_seq_len"] = config["nn_model"]["training"]["main_training"][-1]["seq_len_train"]
            
            # Use local path to downloaded model checkpoint
            # TODO: could also look for latest checkpoint in dir
            _model_filename = f"model_phase_{len(config['nn_model']['training']['main_training'])}.pt"
            _model_path = artifact_dirs[run_id] / _model_filename
            
            if not _model_path.exists():
                raise FileNotFoundError(f"Model checkpoint not found at {_model_path}")
            
            config["nn_model"]["training"]["path_trained_model"] = str(_model_path)
            print(f"\tmodel path: {_model_path}")
            config["nn_model"]["training"]["load_trained_model_for_test"] = True

            # set overrides (propagate to config using key-path logic)
            for key, value in overrides.items():
                print(f"setting override: {key}={value}")
                path = key.split('.')
                ref = config
                for i, part in enumerate(path):
                    if i == len(path) - 1:
                        ref[part] = value
                    else:
                        if part not in ref:
                            ref[part] = {}
                        ref = ref[part]

            config_nn_model = config["nn_model"]
            # delete nn_model from config
            del config["nn_model"]

            # add defaults to the very top
            config["defaults"]= ["base_train_test", {"nn_model": f"model{jobs}"}, "_self_"]
            # also to nn_model
            config_nn_model["defaults"] = [args.nn_model_base, "_self_"]

            # save updated config to temporary directory
            with open(temp_config, 'w') as file:
                yaml.dump(config, file)

            # save nn_model to temporary directory
            with open(temp_dir_path / "nn_model" / f"model{jobs}.yaml", 'w') as file:
                yaml.dump(config_nn_model, file)
            jobs += 1
    print(f"Successfully created {jobs} jobs.")
    print(f"Artifacts and configs saved in: {temp_dir_path}")

    print("starting jobs...")
    # remove all arguments from sys.argv

    def wrap_train_all_phases(temp_dir, temp_config_name, training_dataset):
        """Execute trainer.py in a subprocess with specified configuration.
        
        Args:
            temp_dir (str): Path to temporary directory containing config files.
            temp_config_name (str): Name of the config file to use.
            training_dataset (str): Name of the original training dataset (for logging).
        
        Returns:
            subprocess.CompletedProcess: Result of the subprocess execution.
        """
        # Split the command into executable and arguments instead of one string with spaces.
        cmd = [
            "uv",
            "run",
            "trainer",
            f"-cp={temp_dir}",
            f"-cn={temp_config_name}",
            f"+nn_model.training.training_dataset_name={training_dataset}"
        ]
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        return result

    if args.n_processes == 1:
        print("running jobs sequentially")
        for i in range(jobs):
            result = wrap_train_all_phases(str(temp_dir_path), f"config_{i}.yaml", training_dataset_list[i])
            print(result)
    else:
        print("running jobs in parallel")
        warnings.warn("Parallel execution is not fully tested yet.")
        with Pool(processes=args.n_processes) as pool:
            results = [pool.apply_async(wrap_train_all_phases, (str(temp_dir_path), f"config_{i}.yaml", training_dataset_list[i])) for i in range(jobs)]
            for result in results:
                result.get()
    
    print(f"All jobs completed.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()