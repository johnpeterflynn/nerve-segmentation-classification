import argparse
import collections
import torch
import numpy as np
import data_loaders as module_data
import trainer as trainers_module
import model.loss as module_loss
import model.metric as module_metric
import model as module_arch
from parse_config import ConfigParser
from trainer import Trainer, ProbabilisticTrainer
from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path
import utils as util


def main(config):

    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj(
        'lr_scheduler', torch.optim.lr_scheduler, optimizer)

    experiment = Experiment()
    experiment.set_name(config['name'])

    if 'type' in config['trainer'].keys():
        trainer_name = config['trainer']['type']
    else:
        trainer_name = "Trainer"

    trainer = getattr(trainers_module, trainer_name)
    trainer = trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    trainer.train()


def static_arguments():
    """
        Arguments which are not related to the json file
        Where spicifc logic need to be performed for configuration
        purposes for example
    """
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    args.add_argument('-e', '--env', default=util.RuntimeEnvironment.LOCAL,
                      type=util.RuntimeEnvironment,
                      help='Where to run code (default: local)',
                      choices=list(util.RuntimeEnvironment),
                      action=util.EnvironmentAction
                      )
    args.add_argument('-s', '--seed', default=None, type=int,
                      help='Seed to enable reproducibility')
    return args


def dynamic_arguments():
    """
     custom cli options to modify configuration from default values 
     given in json file.
     """

    CustomArgs = util.namedtuple_with_defaults(
        'CustomArgs', 'flags type target action', (None, ) * 4)
    options = [
        CustomArgs(['--lr', '--learning_rate'],
                   type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int,
                   target='data_loader;args;batch_size'),
        CustomArgs(['--save_dir'], type=str, target='trainer;save_dir'),
        CustomArgs(['--data_dir'], type=str,
                   target='data_loader;args;data_dir'),
    ]
    return options


if __name__ == '__main__':
    args = static_arguments()
    options = dynamic_arguments()
    config = ConfigParser.from_args(args, options)

    main(config)
