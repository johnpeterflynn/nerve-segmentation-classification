import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, binary, impose_labels_on_image, draw_confusion_matrix


class QuickFCNClassifierTrainer(BaseTrainer):
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

        self.best_val_accuracy = 0



    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        train_confusion_matrix = torch.zeros(3, 3, dtype=torch.long)
        print('train epoch: ', epoch)
        for batch_idx, (data, label, target_class) in enumerate(self.data_loader):
            data, target_class = data.to(self.device), target_class.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target_class)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target_class))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

                self._visualize_input(data.cpu())

            p_cls = torch.argmax(output, dim=1)
            for i, t_cl in enumerate(target_class):
                train_confusion_matrix[p_cls[i], t_cl] += 1

            if batch_idx == self.len_epoch:
                break

        print('train confusion matrix:')
        print(train_confusion_matrix)
        self._visualize_prediction(train_confusion_matrix)
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
            val_confusion_matrix = torch.zeros(3, 3, dtype=torch.long)
            print('val epoch: ', epoch)
            for batch_idx, (data, label, target_class) in enumerate(self.valid_data_loader):
                data, target_class = data.to(self.device), target_class.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target_class)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target_class))

                self._visualize_input(data.cpu())
                #prediction = torch.argmax(output)
                #self.logger.debug('val class prediction, actual: {}, {}'.format(prediction, target_class))

                p_cls = torch.argmax(output, dim=1)
                for i, t_cl in enumerate(target_class):
                    val_confusion_matrix[p_cls[i], t_cl] += 1

            print('val confusion matrix:')
            print(val_confusion_matrix)
            self._visualize_prediction(val_confusion_matrix)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        val_log = self.valid_metrics.result()

        # TODO: Super hacky way to display best val dice score. Better way possible?
        self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'best_valid')
        val_scores = {k: v for k, v in val_log.items()}
        current_val_accuracy = val_scores['accuracy']

        if current_val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = current_val_accuracy
            self.valid_metrics.update('accuracy', self.best_val_accuracy)

        return val_log

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

    def _visualize_prediction(self, matrix):
        """format and display output and target data on tensorboard"""
        out = draw_confusion_matrix(matrix)
        self.writer.add_image('output', make_grid(out, nrow=8, normalize=False))