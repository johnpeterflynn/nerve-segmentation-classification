import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, binary, impose_labels_on_image


class OPUSTrainer(BaseTrainer):
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

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target_seg, target_cl) in enumerate(self.data_loader):
            data, target_seg, target_cl = data.to(self.device), target_seg.to(self.device), target_cl.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target_seg, target_cl)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                if met.__name__ == "accuracy":
                    self.train_metrics.update(met.__name__, met(output[1], target_cl))
                else:
                    self.train_metrics.update(met.__name__, met(output[0], target_seg))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

                self._visualize_input(data.cpu())

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

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
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target_seg, target_cl) in enumerate(self.valid_data_loader):
                data, target_seg, target_cl = data.to(self.device), target_seg.to(self.device), target_cl.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target_seg, target_cl)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    if met.__name__ == "accuracy":
                        self.valid_metrics.update(met.__name__, met(output[1], target_cl))
                    else:
                        self.valid_metrics.update(met.__name__, met(output[0], target_seg))

                data_cpu = data.cpu()
                self._visualize_input(data_cpu)
                self._visualize_prediction(data_cpu, output[0].cpu(), target_seg.cpu())

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

    def _visualize_input(self, input):
        """format and display input data on tensorboard"""
        self.writer.add_image('input', make_grid(input[0, 0, :, :], nrow=8, normalize=True))

    def _visualize_prediction(self, input, output, target):
        """format and display output and target data on tensorboard"""
        out_b1 = binary(output)
        out_b1 = impose_labels_on_image(input[0, 0, :, :], target[0, :, :], out_b1[0, 1, :, :])
        self.writer.add_image('output', make_grid(out_b1, nrow=8, normalize=False))
