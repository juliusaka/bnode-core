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
    sys_argv = sys_argv[1:]
    for arg in sys_argv:
        key, value = arg.split("=")
        command_line_args[key] = value.split(",")
    return command_line_args

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
    command_line_args = parse_command_line_args(sys.argv)

    run_ids = get_run_ids(command_line_args)
    
    jobs = 0
    for run_id in run_ids:
        # retrieve config from mlflow
        _artifact_uri = mlflow.get_run(run_id).info.artifact_uri
        _error_file_path = filepaths.filepath_from_ml_artifacts_uri(_artifact_uri + "/could_not_log_artifacts.txt")
        if pathlib.Path(_error_file_path).exists():
            print("For run_id {} files could not be logged.".format(run_id))
            with open(_error_file_path, 'r') as f:
                file = f.read()
            unlogged_files = file.split("\nFile: ")
            unlogged_files = unlogged_files[1:]
            unlogged_files[-1] = unlogged_files[-1].split("\n")[0]
            print(unlogged_files)
            for file in unlogged_files:
                _file_path = pathlib.Path(file)
                _target_path = filepaths.filepath_from_ml_artifacts_uri(_artifact_uri) / _file_path.name
                print("Copying file {} \n\tto {}".format(_file_path, _target_path))
                _copy = False
                if _target_path.exists():
                    # compare file sizes
                    if _file_path.stat().st_size == _target_path.stat().st_size:
                        print("Files are equal. Skipping.")
                    else:
                        print("Files are not equal. Copying file")
                        _copy = True
                else:
                    print("File does not exist. Copying file.")
                    _copy = True
                if _copy:
                    shutil.copy(_file_path, _target_path)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # experiment_id
    sys.argv += ["experiment=myModel"] # oder 232112805895166389"]
    # run_name
    # sys.argv += ["run_name=rebellious-quail-290,traveling-worm-98,capable-hawk-622"]
    # # or run_id
    # sys.argv = ["run_id=8c2c32b9407a4e20946f72cd1c714776,fef513818b67421eb5f450169a429508"]
    # mlflow_tracking_uri
    sys.argv += ["mlflow_tracking_uri=http://localhost:5000"]
    main()