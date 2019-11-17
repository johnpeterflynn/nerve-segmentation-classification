import os
from tensorboardX import SummaryWriter
from util_functions.RuntimeEnvironment import RuntimeEnvironment


class TensorBoardLogger:
    tb_writer: SummaryWriter = None
    _runtime_env: RuntimeEnvironment = None

    def __init__(self, runtime_env: RuntimeEnvironment):
        self._runtime_env = runtime_env

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

        current_exp_num = len(next(os.walk(log_path))[1])
        writer_dir = os.path.join(log_path, 'exp{}/'.format(current_exp_num))
        if self._runtime_env == RuntimeEnvironment.LOCAL or self._runtime_env == RuntimeEnvironment.COLAB:
            self.tb_writer = SummaryWriter(logdir=writer_dir)

    def init_with_exact_path(self, log_path):
        """
            Initializes TensorBoard Writer with the given directory
        :param log_path: [str] Path to directory
        """
        if log_path is None:
            raise Exception("Logdir cannot be empty!")

        if self._runtime_env == RuntimeEnvironment.LOCAL or self._runtime_env == RuntimeEnvironment.COLAB:
            self.tb_writer = SummaryWriter(logdir=log_path)

    def _check_init_properly(self):
        if self._runtime_env == RuntimeEnvironment.LOCAL and self.tb_writer is None:
            raise Exception("TensorBoardLogger was not initialized properly!")
        if self._runtime_env == RuntimeEnvironment.COLAB and self.tb_writer is None:
            raise Exception("TensorBoardLogger was not initialized properly!")

    def add_scalar(self, metric_name, scalar, epoch):
        self._check_init_properly()

        if metric_name is None:
            raise Exception("'metric_name' cannot be empty!")
        if scalar is None:
            raise Exception("'scalar' cannot be empty!")
        if metric_name is None:
            raise Exception("'epoch' cannot be empty!")

        if self._runtime_env == RuntimeEnvironment.LOCAL or self._runtime_env == RuntimeEnvironment.COLAB:
            self.tb_writer.add_scalar(metric_name, scalar, epoch)

    def finish(self):
        if self._runtime_env == RuntimeEnvironment.LOCAL or self._runtime_env == RuntimeEnvironment.COLAB:
            self.tb_writer.close()
