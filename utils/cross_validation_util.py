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


def evaluate(result_df):
    result_df.sort_values(by="best_val_loss")
    print(result_df)

if __name__ == "__main__":

    args = argparse.ArgumentParser(description="Cross Validatin Calculator")
    args.add_argument("-g", "--group", type=int, required=True,
                      help="Polyaxon Experiment Group number")

    args = args.parse_args()
        

    results_df = parse_polyaxon_output(args.group)

    evaluate(results_df)
