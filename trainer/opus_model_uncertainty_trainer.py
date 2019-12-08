import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, binary, impose_labels_on_image, visualization, build_segmentation_grid


class OPUSWithUncertaityTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, experiment=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config, experiment)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.metrics_sample_count = config['trainer']['metrics_sample_count']

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
            output = self.model(data)
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

        #val_size, num_channels, image_size = len(self.valid_data_loader.dataset), 1, tuple(data.shape[2:])
        #samples = torch.zeros((val_size, self.metrics_sample_count, num_channels, *image_size)).to(self.device)

        with torch.no_grad():
            for batch_idx, (data, target, idxs) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

                # [BATCH_SIZE x SAMPLE_SIZE x NUM_CHANNELS x H x W]
                samples = self._sample(self.model, data)

                for met in self.metric_ftns:
                    if met.__name__ in ["ged", "dice_agreement_in_samples", "iou_samples_per_label", "variance_ncc_samples"]:
                        self.valid_metrics.update(
                            met.__name__, met(samples, target))
                    else:
                        self.valid_metrics.update(
                            met.__name__, met(output, target))

                #self._visualize_batch(data, epoch, samples, target)
                for idx in range(data.shape[0]):
                    results_list.append(
                        (data[idx, ...], samples[idx, ...], target[idx, ...], idxs[idx]))

        # TODO: Very ugly fix later
        results_list.sort(key=lambda tup: tup[3])

        data = torch.cat([tup[0].unsqueeze(0) for tup in results_list])
        samples = torch.cat([tup[1].unsqueeze(0) for tup in results_list])
        target = torch.cat([tup[2].unsqueeze(0) for tup in results_list])

        self._visualize_batch(data, epoch, samples, target)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _visualize_batch(self, inputs, batch_idx, samples, targets):
        """
            inputs: [BATCH_SIZE x NUM_CHANNELS x H x W] #  NUM_CHANNELS = 7 for opus 
            samples: [BATCH_SIZE x SAMPLE_SIZE x NUM_CHANNELS x H x W]
            targets: [BATCH_SIZE x  H x W]
        """
        img_metric_grid = build_segmentation_grid(
            self.metrics_sample_count, targets, inputs, samples)

        #self.writer.add_image(f'segmentations_batch_idx_{batch_idx}', img_metric_grid.cpu())
        self.writer.add_image(f'segmentations_ouput', img_metric_grid.cpu())

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
