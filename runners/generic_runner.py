import os
import sys

# This is important to be able to call other modules
# in the upper directory (root dir for our code)
sys.path.append(os.getcwd())

from base import BaseRunner, CustomArgs

class GenericRunner(BaseRunner):

    def __init__(self):
        super().__init__("Generic Runner")

    def add_static_arguments(self):
        super().add_static_arguments()

        # self.static_arguments.add_argument()

        # do something else

    def add_dynamic_arguments(self):
        super().add_dynamic_arguments()

        self.dynamic_arguments.append(
            CustomArgs(['--n_folds'], type=int,
                       target="data_loader;args;cross_val;n_fold",
                       help="perform 'n_fold' folds cross validation")
        )

        self.dynamic_arguments.append(
            CustomArgs(['--valset_idx'], type=int,
                       target="data_loader;args;cross_val;valset_idx",
                       help="Which fold to use as validation in case of cross validation")
        )


if __name__ == "__main__":
    runner = GenericRunner()
    runner.run()
