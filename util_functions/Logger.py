import os
from tensorboardX import SummaryWriter
from polyaxon_client.tracking import Experiment


class TensorBoardLogger:
    tb_writer: SummaryWriter = None
    experiment: Experiment = None

    def __init__(self, experiment: Experiment):
        self.experiment = experiment

    def init_with_ascending_naming(self, log_path):
        """
            Initializes TensorBoard Writer with exp# subdirectory
            For example:
                    logs
                        /exp0
                        /exp1
                        /exp2
                        /exp3
            Experiment number determined based on number of files/folders in the directory
        :param log_path: [str] Path to directory
        """
        if log_path is None:
            raise Exception("Logdir cannot be empty!")

        os.makedirs(log_path, exist_ok=True)

        current_exp_num = len(next(os.walk(log_path))[1])  # make directory if doesn't exist
        writer_dir = os.path.join(log_path, 'tb_log_exp{}/'.format(current_exp_num))
        self.tb_writer = SummaryWriter(logdir=writer_dir)

    def init_with_exact_path(self, log_path):
        """
            Initializes TensorBoard Writer with the given directory
        :param log_path: [str] Path to directory
        """
        if log_path is None:
            raise Exception("Logdir cannot be empty!")

        os.makedirs(log_path, exist_ok=True)  # make directory if doesn't exist
        self.tb_writer = SummaryWriter(logdir=log_path)

    def _check_init_properly(self):
        if self.tb_writer is None:
            raise Exception("TensorBoardLogger was not initialized properly!")

    def log_metrics(self, epoch, **kwargs):
        self._check_init_properly()

        if epoch is None:
            raise Exception("'epoch' cannot be empty!")

        for arg_name, arg_val in kwargs.items():
            self.tb_writer.add_scalar(arg_name, arg_val, epoch)

        self.experiment.log_metrics(**kwargs)

    def finish(self):
        self._check_init_properly()
        self.tb_writer.close()
