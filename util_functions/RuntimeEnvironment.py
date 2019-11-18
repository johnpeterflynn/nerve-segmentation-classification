from enum import Enum


class RuntimeEnvironment(Enum):
    LOCAL = 1
    COLAB = 2
    POLYAXON = 3
