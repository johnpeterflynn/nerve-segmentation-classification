import os
from io import StringIO
import csv
import pandas as pd
import argparse
from pathlib import Path


def build_group_dir(args):
    """
        Returns the path to the group dir
    """

    username = args.user
    group = args.group
    project = args.project

    path = f"/outputs/{username}/{project}/groups/{group}/"
    return Path(path)


def get_latest_experiment(experiment_path):
    """
        In case of resume, each experiment path will have 
        more than one folder named with time tag
        In this case, we consider the latest run
    """
    experiments = [x for x in experiment_path.iterdir() if x.is_dir()]
    # Take the latest experiment
    experiments.sort(reverse=True)
    latest_exp = experiments[0]

    best_model = [x for x in latest_exp.iterdir() if x.suffixes[0]
                  == ".pth" and x.name == "model_best.pth"]
    assert len(best_model) == 1, \
        "Check the experiment %s folder, it seems that the model has no best_mode.pth " % latest_exp
    return best_model[0]


def genereate_experiments_dir(group_path: Path, model_name):

    # %d/models/{model_name}/%d/model_best.pth
    exp_path_str = "{0}" + f"/models/{model_name}/"
    experiments_path = [(x.name, Path(exp_path_str.format(x)))
                        for x in group_path.iterdir() if x.is_dir()]
    experiments = [(exp_num, get_latest_experiment(exp)) for exp_num, exp in experiments_path]
    return experiments


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="Cross Validatin Calculator")
    args.add_argument("-g", "--group", type=int, required=True,
                      help="Polyaxon Experiment Group number")
    args.add_argument("-p", "--project", type=str, required=True,
                      help="Polyaxon Project Name")
    args.add_argument("-u", "--user", type=str, required=True,
                      help="Polyaxon username")
    args.add_argument("-m", "--model-name", type=str, required=True,
                      help="Model name - should match the model name of the network json file.")

    args = args.parse_args()

    group_dir = build_group_dir(args)
    experiments_paths = genereate_experiments_dir(group_dir, args.model_name)


    for exp_num, exp in experiments_paths:
        cmd = f"python3 -u testers/opus_tester.py -r {exp}  -e polyaxon --suffix {exp_num}"
        stream = os.popen(cmd)
        print("=" * 10 + f"Exp {exp_num} testing is starting" + "=" * 10)
        print(stream.read())
        print("=" * 10 + f"Exp {exp_num} testing is done" + "=" * 10)
