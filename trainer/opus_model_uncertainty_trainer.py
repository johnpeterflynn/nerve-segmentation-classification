import numpy as np
import torch
from torchvision.utils import make_grid

from base import BaseTrainer
from trainer import Trainer
from utils import util


class OPUSWithUncertaityTrainer(Trainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, experiment=None):

        super().__init__(model, criterion, metric_ftns, optimizer, config, data_loader,
                         valid_data_loader=valid_data_loader, lr_scheduler=lr_scheduler, len_epoch=len_epoch, experiment=experiment)

        self.train_mc_sample_count = config['trainer']['mc_sample_count']['train']
        self.val_mc_sample_count = config['trainer']['mc_sample_count']['val_test']

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target, _) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            output, _ = util.sample_and_compute_mean(self.model, data, self.train_mc_sample_count, 2, self.device)

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                if met.__name__ not in ["ged", "dice_agreement_in_samples", "iou_samples_per_label", "variance_ncc_samples"]:
                    self.train_metrics.update(
                        met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.model.enable_test_dropout()
        self.valid_metrics.reset()
        results_list = []

        with torch.no_grad():
            for batch_idx, (data, target, idxs) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output, samples = util.sample_and_compute_mean(self.model, data, self.val_mc_sample_count, 2, self.device)

                loss = self.criterion(output, target)

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

                samples = util.argmax_over_dim(samples)

                for met in self.metric_ftns:
                    if met.__name__ in ["ged", "dice_agreement_in_samples", "iou_samples_per_label", "variance_ncc_samples"]:
                        self.valid_metrics.update(
                            met.__name__, met(samples, target))
                    else:
                        self.valid_metrics.update(
                            met.__name__, met(output, target))

                output = util.argmax_over_dim(output, dim=1)

                for idx in range(data.shape[0]):
                    results_list.append(
                        (data[idx, ...], samples[idx, ...], target[idx, ...], idxs[idx], output[idx]))

        # TODO: Very ugly fix later
        results_list.sort(key=lambda tup: tup[3])

        data = torch.cat([tup[0].unsqueeze(0) for tup in results_list])
        samples = torch.cat([tup[1].unsqueeze(0) for tup in results_list])
        target = torch.cat([tup[2].unsqueeze(0) for tup in results_list])
        output = torch.cat([tup[4].unsqueeze(0) for tup in results_list])

        self._visualize_validation_set(data, samples, target, output)

        return self.valid_metrics.result()

    def _visualize_validation_set(self, inputs, samples, target, output):
        """
            inputs: [BATCH_SIZE x NUM_CHANNELS x H x W] #  NUM_CHANNELS = 7 for opus 
            samples: [BATCH_SIZE x SAMPLE_SIZE x NUM_CHANNELS x H x W]
            target: [BATCH_SIZE x  H x W]
            output: [BATCH_SIZE x  H x W]
        """
        img_metric_grid = util.build_segmentation_grid(
            self.val_mc_sample_count, target, inputs, samples, output)

        self.writer.add_image(f'segmentations_ouput', img_metric_grid.cpu())