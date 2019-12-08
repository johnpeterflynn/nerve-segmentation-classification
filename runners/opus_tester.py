import argparse
import os
import sys


# This is important to be able to call other modules
# in the upper directory (root dir for our code)
sys.path.append(os.getcwd())

import torch
from tqdm import tqdm

import data_loaders as module_data
import model as module_arch
import model.loss as module_loss
import model.metric as module_metric
from base import BaseRunner, CustomArgs
from parse_config import ConfigParser
from utils import build_segmentation_grid, save_grid

class OpusTester(BaseRunner):

    def __init__(self):
        super().__init__("OpusTester")
        self.metrics_sample_count = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def add_static_arguments(self):
        super().add_static_arguments()

    def add_dynamic_arguments(self):
        super().add_dynamic_arguments()

    def _sample(self, model, data):
        num_samples = self.metrics_sample_count

        batch_size, num_channels, image_size = data.shape[0], 1, tuple(
            data.shape[2:])
        samples = torch.zeros(
            (batch_size, num_samples, num_channels, *image_size)).to(self.device)
        for i in range(num_samples):
            output = model(data)

            _, idx = torch.max(output, 1)
            sample = idx.unsqueeze(dim=1)
            samples[:, i, ...] = sample

        return samples

    def _run(self, config):
        logger = config.get_logger('test')
        self.metrics_sample_count = config['trainer']['metrics_sample_count']
        # setup data_loader instances
        data_loader = getattr(module_data, config['data_loader']['type'])(
            config['data_loader']['args']['data_dir'],
            batch_size=1,
            shuffle=False,
            validation_split=0.0,
            training=False,
            num_workers=2
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

        with torch.no_grad():
            for _, (data, target) in enumerate(tqdm(data_loader)):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)

                # [BATCH_SIZE x SAMPLE_SIZE x NUM_CHANNELS x H x W]
                samples = self._sample(model, data)

                # computing loss, metrics on test set
                loss = loss_fn(output, target)
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(metric_fns):
                    if metric.__name__ in ["ged", "dice_agreement_in_samples", "iou_samples_per_label", "variance_ncc_samples"]:
                        total_metrics[i] += metric(samples,
                                                   target) * batch_size
                    else:
                        total_metrics[i] += metric(output, target) * batch_size
                for idx in range(data.shape[0]):
                    results_list.append((data[idx, ...], samples[idx, ...], target[idx, ...]))

        # TODO: Very ugly fix later - lazy to write something different 
        data = torch.cat([tup[0].unsqueeze(0) for tup in results_list])
        samples = torch.cat([tup[1].unsqueeze(0) for tup in results_list])
        target = torch.cat([tup[2].unsqueeze(0) for tup in results_list])
        grid = build_segmentation_grid(self.metrics_sample_count, target, data, samples)
    
        
        save_grid(grid, config['trainer']['save_dir'])

        n_samples = len(data_loader.sampler)
        log = {'loss': total_loss / n_samples}
        log.update({
            met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
        })
        logger.info(log)


if __name__ == "__main__":
    runner = OpusTester()
    runner.run()
