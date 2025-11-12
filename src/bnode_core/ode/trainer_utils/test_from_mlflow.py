"""Test trained models from MLflow runs on new datasets.

This module provides functionality to load trained neural ODE models from MLflow runs
and test them on different datasets. It automates the process of retrieving model
configurations and checkpoints from MLflow, updating test parameters, and executing
validation runs.
You can use all configuration options from trainer.py to override parameters for testing.

Typical Usage Example
---------------------
Test a single run on a new dataset:

```python
    python test_from_mlflow.py \\
        experiment=myModel # or experiment id \\
        run_name=bemused-hen-59 \\
        # or run_id=8c2c32b9407a4e20946f72cd1c714776
        dataset_name=myTestData \\
        mlflow_experiment_name=validation_results \\
        mlflow_tracking_uri=http://localhost:5000 \\
        n_processes=1 \\
        nn_model_base=latent_ode_base\\
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
    experiment + run_name : str
        Experiment name and specific run name within that experiment.
    experiment : str
        Experiment name to test all runs from that experiment.

Optional:

    mlflow_tracking_uri : str
        URI of MLflow tracking server. Defaults to 'http://localhost:5000'.
    n_processes : int
        Number of parallel processes for testing. Defaults to 1 (sequential).
    override <key>=<value> : 
        Override specific config parameters (e.g., 'override use_cuda=false').

Notes
-----
- The module creates temporary configuration files combining the original training
  config with test-specific updates.
- Model checkpoints are automatically retrieved from MLflow artifact storage.
- Results are logged to a new MLflow experiment specified by mlflow_experiment_name.
- When testing multiple runs/datasets, a Cartesian product is created (all combinations).

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
import logging
import tempfile
import shutil
import bnode_core.filepaths as filepaths
import yaml
import pathlib
import subprocess
import hydra
from bnode_core.ode.trainer import train_all_phases
from multiprocessing import Pool
import warnings

def parse_command_line_args(sys_argv):
    """Parse command line arguments into structured dictionaries.
    
    Processes command-line arguments and separates them into regular arguments
    and override arguments. Regular arguments use '=' syntax, while overrides
    use 'override key=value' syntax.
    
    Args:
        sys_argv (list[str]): System argument vector (typically sys.argv).
            Expected format:
                - Regular: 'key=value1,value2' (values split by comma)
                - Override: 'override key=value'
    
    Returns:
        tuple (tuple[dict, dict]): A tuple containing:
            - command_line_args (dict): Regular arguments with values as lists.
            - overrides (dict): Override arguments with values as strings.
    
    Examples:
        >>> parse_command_line_args(['script.py', 'dataset_name=test1,test2', 'override use_cuda=false'])
        ({'dataset_name': ['test1', 'test2']}, {'use_cuda': 'false'})
    """
    command_line_args = {}
    overrides = {}
    sys_argv = sys_argv[1:]
    for arg in sys_argv:
        if not arg.startswith("override"):
            key, value = arg.split("=")
            command_line_args[key] = value.split(",")
        else:
            _command = arg.split(" ")
            key, value = _command[1].split("=")
            overrides[key] = value
    return command_line_args, overrides

def validate_command_line_args(command_line_args):
    """Validate required command line arguments are present.
    
    Checks that all required arguments are provided and sets defaults for
    optional arguments.
    
    Args:
        command_line_args (dict): Parsed command line arguments.
    
    Raises:
        ValueError: If required arguments (dataset_name, mlflow_experiment_name,
            nn_model_base) are missing.
    
    Side Effects:
        - Sets command_line_args['n_processes'] = 1 if not provided.
        - Prints warning messages for missing optional arguments.
    """
    if "dataset_name" not in command_line_args:
        raise ValueError("No dataset_name provided. Please provide a dataset_name.")
    if "mlflow_experiment_name" not in command_line_args:
        raise ValueError("No mlflow_experiment_name provided. Please provide a mlflow_experiment_name.")
    if "n_processes" not in command_line_args:
        print("No n_processes provided. Using default value of 1.")
        command_line_args["n_processes"] = 1
    if "nn_model_base" not in command_line_args:
        raise ValueError("No nn_model_base provided. Please provide a nn_model_base.")

def get_run_ids(command_line_args):
    """Retrieve MLflow run IDs based on provided selection criteria.
    
    Resolves run IDs from either direct run_id specification, run_name lookup
    within an experiment, or all runs from an experiment.
    
    Args:
        command_line_args (dict): Parsed command line arguments containing one of:
            - 'run_id': Direct list of run IDs.
            - 'experiment' + 'run_name': Experiment name and specific run names.
            - 'experiment': All runs from the experiment.
            Optional:
            - 'mlflow_tracking_uri': MLflow server URI.
    
    Returns:
        list (list[str]): List of MLflow run IDs to test.
    
    Raises:
        ValueError: If incompatible argument combinations are provided
            (e.g., both run_name and run_id, or both run_id and experiment).
    
    Side Effects:
        - Sets MLflow tracking URI via mlflow.set_tracking_uri().
        - Prints progress messages about run ID retrieval.
    
    Examples:
        >>> get_run_ids({'experiment': ['my_exp'], 'run_name': ['run1', 'run2']})
        ['abc123', 'def456']
    """
    if "mlflow_tracking_uri" in command_line_args:
        mlflow.set_tracking_uri(command_line_args["mlflow_tracking_uri"][0])
    else:
        print("No mlflow_tracking_uri provided. Using default.")
        mlflow.set_tracking_uri("http://localhost:5000")

    # get run_ids
    if "run_name" in command_line_args.keys() and "run_id" in command_line_args.keys():
        raise ValueError("Both run_name and run_id provided. Please provide only one.")
    
    if "run_id" in command_line_args.keys() and "experiment" in command_line_args.keys():
        raise ValueError("Both run_id and experiment provided. Please provide only one.")

    if "experiment" in command_line_args.keys():
        experiment_name = command_line_args["experiment"]
        print("experiment_name: ", experiment_name)
        print("searching for runs of experiment")
        experiment = mlflow.get_experiment_by_name(experiment_name[0])
        runs = mlflow.search_runs(experiment.experiment_id)
        if "run_name" in command_line_args:
            # retrieve run_ids from run_names
            print("searching for run_ids of run_names")
            run_names = command_line_args["run_name"]
            run_ids = []
            for run_name in run_names:
                run_ids.append(runs[runs["tags.mlflow.runName"] == run_name]["run_id"].values[0])
        else:
            print("using all runs of experiment")
            run_ids = runs["run_id"].to_list()
    else:
        run_ids = command_line_args["run_id"]
    print("run_ids: ", run_ids)
    return run_ids

def main():
    """Main execution function for testing models from MLflow runs.
    
    Orchestrates the complete workflow:
    1. Parses and validates command line arguments.
    2. Retrieves run IDs from MLflow.
    3. For each run-dataset combination:
       - Downloads and loads the training configuration from MLflow artifacts.
       - Updates config for testing (new dataset, model path, test mode).
       - Applies any override parameters.
       - Saves modified config to temporary files.
    4. Executes testing jobs either sequentially or in parallel.
    
    The function creates a Cartesian product of runs × datasets, generating
    one test job for each combination.
    
    Raises:
        ValueError: If command line arguments are invalid or incompatible.
        FileNotFoundError: If MLflow artifacts (config, model) cannot be found.
    
    Side Effects:
        - Creates temporary directory for config files (auto-cleaned on exit).
        - Launches subprocess calls to trainer.py for each test job.
        - Logs results to MLflow under the specified experiment name.
        - Prints progress messages throughout execution.
    
    Notes:
        - Model checkpoints are retrieved from the final training phase.
        - Sequence length is set to match the last training phase.
        - Original training dataset name is preserved for reference.
    
    Examples:
        Command line usage::
        
            python test_from_mlflow.py \\
                experiment=trained_models \\
                run_name=final-model-123 \\
                dataset_name=validation_set \\
                mlflow_experiment_name=validation_results \\
                nn_model_base=latent_ode_base \\
                n_processes=1
    """
    command_line_args, overrides = parse_command_line_args(sys.argv)
    validate_command_line_args(command_line_args)

    run_ids = get_run_ids(command_line_args)
    
    dataset_names = command_line_args["dataset_name"]

    # get temporary directory for saving config files
    temp_dir = tempfile.TemporaryDirectory()
    print("using temporary directory for saving config files: ", temp_dir.name)
    # create nn_model directory in temporary directory
    pathlib.Path(temp_dir.name + "/nn_model").mkdir(parents=True, exist_ok=True)

    jobs = 0
    training_dataset_list = []
    for run_id in run_ids:
        for dataset in dataset_names:
            # retrieve config from mlflow
            _artifact_uri = mlflow.get_run(run_id).info.artifact_uri
            # TODO: this should be downloaded from the artifact store
            _config_path = filepaths.filepath_from_ml_artifacts_uri(_artifact_uri + "/.hydra/config_validated.yaml")

            # copy config to temporary directory
            temp_config = temp_dir.name + "/config " + str(jobs) + ".yaml"
            print(f"copying config from {_config_path} to {temp_config}")
            shutil.copy(_config_path, temp_config)
            # update config with dataset_name, sequence_length, model_path, test_mode
            # open yaml file
            with open(temp_config) as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
            
            # update config
            training_dataset_list.append(config["dataset_name"])
            config["dataset_name"] = dataset
            config["mlflow_experiment_name"] = command_line_args["mlflow_experiment_name"][0]
            config["nn_model"]["training"]["pre_trained_model_seq_len"] = config["nn_model"]["training"]["main_training"][-1]["seq_len_train"]
            _model_path = _artifact_uri + "/model_phase_" + str(len(config["nn_model"]["training"]["main_training"])) + ".pt"
            config["nn_model"]["training"]["path_trained_model"] = _model_path
            print(f"\tmodel path: {_model_path}")
            config["nn_model"]["training"]["load_trained_model_for_test"] = True

            # set overrides
            for key, value in overrides.items():
                print(f"setting override: {key}={value}")
                path = key.split(".")
                if len(path) == 1:
                    config[path[0]] = value
                elif len(path) == 2:
                    config[path[0]][path[1]] = value
                elif len(path) == 3:
                    config[path[0]][path[1]][path[2]] = value
                else:
                    raise ValueError("Invalid override path.")

            config_nn_model = config["nn_model"]
            # delete nn_model from config
            del config["nn_model"]

            # add defaults to the very top
            config["defaults"]= ["base_train_test", {"nn_model": f"model{jobs}"}, "_self_"]

            # also to nn_model
            config_nn_model["defaults"] = [command_line_args["nn_model_base"][0], "_self_"]

            # save updated config to temporary directory
            with open(temp_config, 'w') as file:
                yaml.dump(config, file)

            # save nn_model to temporary directory
            with open(temp_dir.name + "/nn_model/model" + str(jobs) + ".yaml", 'w') as file:
                yaml.dump(config_nn_model, file)
            jobs += 1
    print(f"Successfully created {jobs} jobs.")

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
        result = subprocess.run(["uv run trainer", f"-cp={temp_dir}" , f"-cn={temp_config_name}", f"+nn_model.training.training_dataset_name={training_dataset}"])
        return result

    if command_line_args["n_processes"][0] == "1":
        print("running jobs sequentially")
        for i in range(jobs):
            result = wrap_train_all_phases(temp_dir.name, f"config {str(i)}.yaml", training_dataset_list[i])
            print(result)
    else:
        print("running jobs in parallel")
        warnings.warn("Parallel execution is not fully tested yet.")
        with Pool(processes=int(int(command_line_args["n_processes"][0]))) as pool:
            results = [pool.apply_async(wrap_train_all_phases, (temp_dir.name, f"config {str(i)}.yaml")) for i in range(jobs)]
            for result in results:
                result.get()
    # start jobs

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()