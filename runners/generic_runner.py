import sys
import os

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

if  __name__ == "__main__":
    runner = GenericRunner()
    runner.run()