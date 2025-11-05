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
import filepaths
import yaml
import pathlib
import subprocess
import hydra
from networks.neural_ode.trainer import train_all_phases
from multiprocessing import Pool

def parse_command_line_args(sys_argv):
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
        result = subprocess.run(["python", "networks/neural_ode/trainer.py", f"-cp={temp_dir}" , f"-cn={temp_config_name}", f"+nn_model.training.training_dataset_name={training_dataset}"])
        return result

    if command_line_args["n_processes"][0] == "1":
        print("running jobs sequentially")
        for i in range(jobs):
            result = wrap_train_all_phases(temp_dir.name, f"config {str(i)}.yaml", training_dataset_list[i])
            print(result)
    else:
        print("running jobs in parallel")
        # with Pool(processes=int(int(command_line_args["n_processes"][0]))) as pool:
        #     results = [pool.apply_async(wrap_train_all_phases, (temp_dir.name, f"config {str(i)}.yaml")) for i in range(jobs)]
        #     for result in results:
        #         result.get()
    # start jobs




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # experiment_id
    sys.argv += ["experiment=myModel"] # oder 232112805895166389"]
    # run_name
    sys.argv += ["run_name=bemused-hen-59"]
    # # or run_id
    # sys.argv = ["run_id=8c2c32b9407a4e20946f72cd1c714776,fef513818b67421eb5f450169a429508"]
    # dataset
    sys.argv += ["dataset_name=myData"] 
    # mlflow_experiment_name
    sys.argv += ["mlflow_experiment_name=myNewExperiment"]
    # mlflow_tracking_uri
    sys.argv += ["mlflow_tracking_uri=http://localhost:5000"]
    # launch n_processes
    sys.argv += ["n_processes=1"]
    # specifiy nn_model_base
    sys.argv += ["nn_model_base=latent_ode_base"]
    # specify overrides
    sys.argv += ["override nn_model.training.batch_size_test=128"]
    sys.argv += ["override nn_model.training.test_save_internal_variables=false"]
    sys.argv += ["override n_workers_train_loader=1"]
    sys.argv += ["override n_workers_other_loaders=1"]
    sys.argv += ["override use_cuda=true"]
    main()