import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, experiment):
        self.config = config
        self.logger = config.get_logger(
            'trainer', config['trainer']['verbosity'])
        self.experiment = experiment

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = TensorboardWriter(
            config.log_dir, self.logger, cfg_trainer['tensorboard'], experiment)

        if config.resume is not None:
            self._resume_checkpoint(config.resume)
        else:
            # Temporary
            seg_path = '/outputs/johnpeterflynn/multitask-common/groups/1406/122698/models/QuickNat/0203_053832/model_best.pth'
            classif_path = '/outputs/johnpeterflynn/multitask-common/experiments/122323/models/QuickFCNClassifier/0130_134509/model_best.pth'
            self._load_seg_and_classif(seg_path, classif_path)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            self.experiment.log_metrics(**log)
            
            self.writer.set_step(epoch)
            if self.writer is not None:
                for key, value in result.items():
                    self.writer.add_scalar(key, value)


            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode ==
                                'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break
            if improved:
                # Save the model if there is an improvement immediately
                self._save_checkpoint(epoch, save_best=True)
                self.log_best_model_validation_results(log)
            elif epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=False)

    def log_best_model_validation_results(self, log):
        """
            Whenever the results imporove, this means we have a new
            record (best_model), So we log the validation metrics 
            using a different metric so that we can use it later
            when running cross_validation ,for example, to select
            the best model
        """
        log = {k: v for k, v in log.items() if k.split("_")[0] == "val"}
        log.update(**{'best_'+k: v for k, v in log.items()})
        self.experiment.log_metrics(**log)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir /
                       'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _load_seg_and_classif(self, segmentation_path, classification_path):

        segmentation_state_dict = self._load_pretrain(segmentation_path, "_seg")
        classification_state_dict = self._load_pretrain(classification_path, "_class")

        self._soft_load_new_stat_dict(self.model, segmentation_state_dict, classification_state_dict)

    def _load_pretrain(self, pretrain_path, postfix):
        pretrain_path = str(pretrain_path)
        self.logger.info("Loading pretraining state: {} ...".format(pretrain_path))
        loaded_state_dict = torch.load(pretrain_path,
                                             map_location=torch.device(self.device))['state_dict']

        # Motify loaded_state_dict to match segmentation-specific parameters
        state_dict = {}
        for key, value in loaded_state_dict.items():
            state_dict[key.replace(".", postfix + ".", 1)] = value

        print('state dict:')
        for key, value in state_dict.items():
            print(key, ', ', value.shape)

        print('model params:')
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

        return state_dict

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(
            resume_path, map_location=torch.device(self.device))
        
        if self.use_transfer_learning():
            self.start_epoch = 1
        else:
            self.start_epoch = checkpoint['epoch'] + 1
            self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")


        self._load_new_stat_dict(self.model, checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            if not self.use_transfer_learning():
                self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def use_transfer_learning(self):
        """
            Returns true if we are using transfer learning
        """
        return 'pre_training' in self.config['trainer'] and self.config['trainer']['pre_training']

    def _soft_load_new_stat_dict(self, object_to_load, segmentation_dict, classification_dict):
        """
        Identical to _load_new_stat_dict() except keeps all parameters in model_dict that do not exist in
        pretrained_dict
        :param object_to_load:
        :param pretrained_dict:
        :return:
        """
        model_dict = object_to_load.state_dict()

        segmentation_dict = {k: v for k, v in segmentation_dict.items() if k in model_dict and (model_dict[k].shape == segmentation_dict[k].shape)}
        model_dict.update(segmentation_dict)
        classification_dict = {k: v for k, v in classification_dict.items() if k in model_dict and (model_dict[k].shape == classification_dict[k].shape)}
        model_dict.update(classification_dict)

        object_to_load.load_state_dict(model_dict)

    def _load_new_stat_dict(self, object_to_load, pretrained_dict):

        model_dict = object_to_load.state_dict()

        # Overwrite conflicting keys
        for k, _ in pretrained_dict.items():
            if k in model_dict and pretrained_dict[k].shape != model_dict[k].shape:
                pretrained_dict[k] = model_dict[k]

        # load the new state dict
        object_to_load.load_state_dict(pretrained_dict)
