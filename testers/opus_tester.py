import argparse
import importlib
import os
import sys
from datetime import datetime
from pathlib import Path

# This is important to be able to call other modules
# in the upper directory (root dir for our code)
sys.path.append(os.getcwd())

import torch
from polyaxon_client.tracking import (Experiment, get_data_paths,
                                      get_outputs_path)
from tqdm import tqdm

import data_loaders as module_data
import model as module_arch
import model.loss as module_loss
import model.metric as module_metric
from base import BaseRunner, CustomArgs
from parse_config import ConfigParser
from utils import build_segmentation_grid, save_grid, util



class OpusTester(BaseRunner):

    def __init__(self):
        super().__init__("OpusTester")
        self.metrics_sample_count = 1
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def add_static_arguments(self):
        super().add_static_arguments()

        self.static_arguments.add_argument("--suffix", type=str, default=None,
            help="Use this prefix when storing any file realted to this test")

    def add_dynamic_arguments(self):
        super().add_dynamic_arguments()

    def _run(self, config):

        # This should never be done, but the framework
        # needs to be update later to give access to the variables
        # that are not network-realted, but for control purposes
        control_args = self.static_arguments.parse_args()

        logger = config.get_logger('test')
        experiment = Experiment()
        experiment.set_name("Test")

        self.metrics_sample_count = config['trainer']['mc_sample_count']['val_test']
        # setup data_loader instances
        data_loader = getattr(module_data, config['data_loader']['type'])(
            config['data_loader']['args']['data_dir'],
            batch_size=1,
            shuffle=False,
            validation_split=0.0,
            training=False,
            num_workers=2,
            with_idx=True
        )

        # build model architecture
        model = config.init_obj('arch', module_arch)
        logger.info(model)

        # get function handles of loss and metrics
        loss_fn = getattr(module_loss, config['loss'])
        metric_fns = [getattr(module_metric, met) for met in config['metrics']]

        logger.info('Loading checkpoint: {} ...'.format(config.resume))

        checkpoint = torch.load(
            config.resume, map_location=torch.device(self.device))

        state_dict = checkpoint['state_dict']
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)

        # prepare model for testing
        model = model.to(self.device)
        model.eval()

        total_loss = 0.0
        total_metrics = torch.zeros(len(metric_fns))
        results_list = []
        metrics_results = []
        model.enable_test_dropout()

        with torch.no_grad():
            for i, (data, target, idx) in enumerate(tqdm(data_loader)):
                data, target = data.to(self.device), target.to(self.device)
                output, samples = util.sample_and_compute_mean(
                    model, data, self.metrics_sample_count, 2, self.device)

                # computing loss, metrics on test set
                loss = loss_fn(output, target)
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size

                samples = util.argmax_over_dim(samples)

                for i, metric in enumerate(metric_fns):
                    if metric.__name__ in ["ged", "dice_agreement_in_samples", "iou_samples_per_label", "variance_ncc_samples"]:
                        s = metric(samples, target)
                        metrics_results.append(round(s, 2))
                        total_metrics[i] += s
                    else:
                        s = metric(output, target)
                        metrics_results.append(round(s.item(), 2))
                        total_metrics[i] += s

                output = util.argmax_over_dim(output, dim=1)
                class_label  = data_loader.dataset.classes_list[idx]
                for i in range(data.shape[0]):
                    results_list.append(
                        (data[i, ...], samples[i, ...], target[i, ...], metrics_results, output[i, ...], class_label))

                metrics_results = []

        # TODO: Very ugly fix later - lazy to write something different
        data = torch.cat([tup[0].unsqueeze(0) for tup in results_list])
        samples = torch.cat([tup[1].unsqueeze(0) for tup in results_list])
        target = torch.cat([tup[2].unsqueeze(0) for tup in results_list])
        output = torch.cat([tup[4].unsqueeze(0) for tup in results_list])
        class_labels = [tup[5] for tup in results_list]

        save_dir = config['trainer']['save_dir']
        grid = util.build_segmentation_grid(self.metrics_sample_count, target, data, samples, output)
      
        if control_args.suffix is not None:
            save_grid(grid, save_dir, control_args.suffix)
        else:
            save_grid(grid, save_dir)

        metrics_results = [tup[3] for tup in results_list]
        
        save_dir_csv = Path(save_dir) / 'test-csv/'
        save_dir_csv.mkdir(parents=True, exist_ok=True)
        target_csv = "test-results.csv"
        if control_args.suffix is not None:
            target_csv = f"test-results-{control_args.suffix}.csv"
        with open(save_dir_csv / target_csv, "w") as f:
            logger.info(", ".join([metric.__name__ for metric in metric_fns]) + ",label") 
            f.writelines(", ".join([metric.__name__ for metric in metric_fns]) + ",label" + "\n") 
            for metrics, label in zip(metrics_results, class_labels):
                f.writelines(", ".join([str(metric) for metric in metrics]) + "," + str(label) + "\n")
                logger.info(", ".join([str(metric) for metric in metrics]) +  "," + str(label))

        
        n_samples = len(data_loader.sampler)
        log = {'loss': total_loss / n_samples}
        log.update({
            met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
        })
        logger.info(log)

if __name__ == "__main__":
    runner = OpusTester()
    runner.run()
