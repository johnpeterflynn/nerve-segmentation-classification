import os
from io import StringIO
import csv
import pandas as pd
import argparse


def parse_polyaxon_output(group_number):
    """
        Query Polyaxon for Experiment group with group_number
        and returns a df with the result (metrics  from experiments)

        @params:
            group_number: int
        @return:
            df
    """

    results_cmd = f'polyaxon group -g {group_number} experiments  -m'
    cmds = [r'sed -e "s/[0-9]\+h [0-9]\+m/irrelevant/g"', 'sed "s/^ *//g"',
            r'sed "s/ \+/,/g"',  'sed -e "1,12d"', 'sed -e "2d"']

    cmd = "| ".join([results_cmd, *cmds])
    stream = os.popen(cmd)
    data = pd.read_csv(stream)

    return data


def evaluate(result_df, metric, best_definition):

    # filter out all zeros metrics
    # data = data.loc[:, (data != 0).all(axis=0)]

    result_df = result_df.sort_values(by=metric)

    if best_definition == "max":
        idx = result_df[metric].idxmax()
    elif best_definition == "min":
        idx = result_df[metric].idxmin()
    else:
        raise Exception("Invalid best_definition value")

    best_metric_value = result_df[metric][idx]
    experiment_number = result_df["id"][idx]

    print(result_df)

    print(" ".join(["Best", metric, "=", str(best_metric_value)]))
    print(" ".join(["Experiment Number", "=", str(experiment_number)]))


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="Cross Validatin Calculator")
    args.add_argument("-g", "--group", type=int, required=True,
                      help="Polyaxon Experiment Group number")
    args.add_argument("-b", "--best-criterion-def", type=str,
                      choices={"max", "min"}, required=True)
    args.add_argument("-m", "--metric", type=str, required=True,
                      choices={"best_val_loss", "val_dice_agreement_in_samples", "val_iou_samples_per_label"})

    args = args.parse_args()

    results_df = parse_polyaxon_output(args.group)

    evaluate(results_df, args.metric, args.best_criterion_def)
