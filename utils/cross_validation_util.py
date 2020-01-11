import os
from io import StringIO
import csv
import pandas as pd
import argparse
from polyaxon_client import PolyaxonClient
from functools import reduce


def parse_polyaxon_output(args):
    """
        Query Polyaxon for Experiment group with group_number
        and returns a df with the result (metrics  from experiments)

        @params:
            group_number: int
        @return:
            df
    """
    pc = PolyaxonClient()
    experiments = pc.experiment_group.list_experiments(
        args.user, args.project, args.group)

    alldict = [exp['last_metric'] for exp in experiments['results']]
    metrics = reduce(lambda x, y: x.union(y.keys()), alldict, set())

    results = {}
    # init the keys to empty lists
    allkeys = ['id'] + list(metrics)
    results = {key: [] for key in allkeys}

    for exp in experiments['results']:
        results['id'].append(exp['id'])
        for metric in metrics:
            val = exp['last_metric'][metric] if metric in exp['last_metric'] else None
            results[metric].append(val)

    df = pd.DataFrame(results, columns=results.keys())
    return df


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
    args.add_argument("-p", "--project", type=str, required=True,
                      help="Polyaxon Project Name")
    args.add_argument("-u", "--user", type=str, required=True,
                      help="Polyaxon username")
    args.add_argument("-b", "--best-criterion-def", type=str,
                      choices={"max", "min"}, required=True)
    args.add_argument("-m", "--metric", type=str, required=True,
                      choices={"best_val_loss", "val_dice_agreement_in_samples", "val_iou_samples_per_label"})

    args = args.parse_args()

    results_df = parse_polyaxon_output(args)

    evaluate(results_df, args.metric, args.best_criterion_def)
