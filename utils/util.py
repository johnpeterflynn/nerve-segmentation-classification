import json
import os
import pickle
import pandas as pd
from pathlib import Path
from itertools import repeat
import collections
from enum import Enum
from argparse import Action
from polyaxon_client.tracking import get_outputs_path, Experiment, get_data_paths


class RuntimeEnvironment(Enum):
    LOCAL = "local"
    COLAB = "colab"
    POLYAXON = "polyaxon"

    def __str__(self):
        return self.value


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=collections.OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(
            index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / \
            self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def load_pickle_file(dataset_location):
    max_bytes = 2 ** 31 - 1
    data = {}
    new_data = None
    print("Loading file", dataset_location)
    bytes_in = bytearray(0)
    input_size = os.path.getsize(dataset_location)
    with open(dataset_location, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    new_data = pickle.loads(bytes_in)
    data.update(new_data)

    del new_data
    return data


def namedtuple_with_defaults(typename, field_names, default_values=()):
    T = collections.namedtuple(typename, field_names)
    T.__new__.__defaults__ = (None,) * len(T._fields)
    if isinstance(default_values, collections.Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T


class EnvironmentAction(Action):

    def __call__(self, parser, namespace, value, option_string=None):

        if value == RuntimeEnvironment.POLYAXON:
            save_dir = get_outputs_path()
            setattr(namespace, "save_dir", save_dir)

            # TODO: We can do this programmaticlly
            # But we need to agree upon a certain format
            # for now, just pass it as an argument through
            # cli
            #data_dir = get_data_paths()
            #setattr(namespace, "data_dir", data_dir['data1'])
