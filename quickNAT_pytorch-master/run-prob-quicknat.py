import argparse
import os

import torch
from torch.utils.data.sampler import SubsetRandomSampler

import utils.evaluator as eu
from quicknat import QuickNat
from probabilistic_quicknat import ProbabilisticQuickNat
from settings import Settings
from solver import Solver
from solver_prob_quickant import SolverProbQuickNat
from utils.data_utils import get_lidc_dataset
from utils.log_utils import LogWriter
import shutil

torch.set_default_tensor_type('torch.FloatTensor')


def load_lidc_dataset(data_params):
    print("Loading dataset")
    dataset, train_indices, test_indices = get_lidc_dataset(data_params)
    print("Train size: %i" % len(train_indices))
    print("Test size: %i" % len(test_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    return dataset, train_sampler, test_sampler


def train(train_params, common_params, data_params, net_params, probabilistic=False):
    dataset, train_sampler, test_sampler = load_lidc_dataset(data_params)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=train_params['train_batch_size'], sampler=train_sampler,
                                               num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=train_params['val_batch_size'], sampler=test_sampler,
                                             num_workers=4, pin_memory=True)
    if train_params['use_pre_trained']:
        try:
            quicknat_model = torch.load(train_params['pre_trained_path'])
        except RuntimeError:
            print("Load the pre-trained model into cpu")
            quicknat_model = torch.load(train_params['pre_trained_path'], map_location=torch.device('cpu'))
    else:
        quicknat_model = ProbabilisticQuickNat(net_params)


    solver = SolverProbQuickNat(quicknat_model,
                    device=common_params['device'],
                    num_class=net_params['num_class'],
                    optim_args={"lr": train_params['learning_rate'],
                                "betas": train_params['optim_betas'],
                                "eps": train_params['optim_eps'],
                                "weight_decay": train_params['optim_weight_decay']},
                    model_name=common_params['model_name'],
                    exp_name=train_params['exp_name'],
                    labels=data_params['labels'],
                    log_nth=train_params['log_nth'],
                    num_epochs=train_params['num_epochs'],
                    lr_scheduler_step_size=train_params['lr_scheduler_step_size'],
                    lr_scheduler_gamma=train_params['lr_scheduler_gamma'],
                    use_last_checkpoint=train_params['use_last_checkpoint'],
                    log_dir=common_params['log_dir'],
                    exp_dir=common_params['exp_dir'])
    
    solver.train(train_loader, val_loader)
    final_model_path = os.path.join(common_params['save_model_dir'], train_params['final_model_file'])
    quicknat_model.save(final_model_path)
    print("final model saved @ " + str(final_model_path))


def evaluate(eval_params, net_params, data_params, common_params, train_params):
    eval_model_path = eval_params['eval_model_path']
    num_classes = net_params['num_class']
    labels = data_params['labels']
    data_dir = eval_params['data_dir']
    label_dir = eval_params['label_dir']
    volumes_txt_file = eval_params['volumes_txt_file']
    remap_config = eval_params['remap_config']
    device = common_params['device']
    log_dir = common_params['log_dir']
    exp_dir = common_params['exp_dir']
    exp_name = train_params['exp_name']
    save_predictions_dir = eval_params['save_predictions_dir']
    prediction_path = os.path.join(exp_dir, exp_name, save_predictions_dir)
    orientation = eval_params['orientation']

    logWriter = LogWriter(num_classes, log_dir, exp_name, labels=labels)

    avg_dice_score, class_dist = eu.evaluate_dice_score(eval_model_path,
                                                        num_classes,
                                                        data_dir,
                                                        label_dir,
                                                        volumes_txt_file,
                                                        remap_config,
                                                        orientation,
                                                        prediction_path,
                                                        device,
                                                        logWriter)
    logWriter.close()


def delete_contents(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', required=True, help='run mode, valid values are train and eval')
    #parser.add_argument('--type', default="normal", help='[normal, probabilistic]')
    
    args = parser.parse_args()

    settings = Settings()
    common_params, data_params, net_params, train_params, eval_params = settings['COMMON'], settings['DATA'], settings[
        'NETWORK'], settings['TRAINING'], settings['EVAL']

    if args.mode == 'train':
        train(train_params, common_params, data_params, net_params)
    elif args.mode == 'eval':
        evaluate(eval_params, net_params, data_params, common_params, train_params)
    elif args.mode == 'clear':
        shutil.rmtree(os.path.join(common_params['exp_dir'], train_params['exp_name']))
        print("Cleared current experiment directory successfully!!")
        shutil.rmtree(os.path.join(common_params['log_dir'], train_params['exp_name']))
        print("Cleared current log directory successfully!!")

    elif args.mode == 'clear-all':
        delete_contents(common_params['exp_dir'])
        print("Cleared experiments directory successfully!!")
        delete_contents(common_params['log_dir'])
        print("Cleared logs directory successfully!!")
    else:
        raise ValueError('Invalid value for mode. only support values are train, eval and clear')
