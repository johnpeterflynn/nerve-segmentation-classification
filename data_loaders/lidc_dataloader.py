from base import BaseDataLoader


import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import random
import pickle

from torch.utils.data import Dataset
from torchvision import transforms
import torch
from utils import load_pickle_file
import _pickle as cPickle
import gc


class LidcDataLoader(BaseDataLoader):
    """
    LIDC data loader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, test_config=None, use_percentage=None):

        self.data_dir = data_dir
        self.dataset = LIDC_IDRI(self.data_dir, use_percentage=use_percentage)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, test_config=test_config)

    def set_random_sampling_mode(self):
        self.dataset.set_random_sampling_mode()

    def set_no_sampling_mode(self):
        self.dataset.set_no_sampling_mode()


class LIDC_IDRI(Dataset):
    images = []
    labels = []
    sampling_mode = None

    def __init__(self, dataset_location, transform=None, use_percentage=None):
        self.transform = transform
        data = {}
        for file in os.listdir(dataset_location):
            filename = os.fsdecode(file)
            if '.pickle' in filename:
                print("Loading file", filename)
                file_path = dataset_location + filename

                with open(file_path, 'rb') as f_in:
                    # disable garbage collector
                    gc.disable()

                    data = cPickle.load(f_in)
                    # enable garbage collector
                    gc.enable()

        n = len(data.items())
        if use_percentage is not None:
            assert 0 < use_percentage <= 1
            n = int(use_percentage * n)

        print("Using LIDC-IDRI dataset with size: ", n)
        self.images = [None] * n
        self.labels = [None] * n
        i = 0
        for key, value in data.items():
            if i >= n:
                break
            self.images[i] = value['image'].astype(float)
            self.labels[i] = value['masks']
            i += 1

        assert (len(self.images) == len(self.labels))

        for img in self.images:
            assert np.max(img) <= 1 and np.min(img) >= 0
        for label in self.labels:
            assert np.max(label) <= 1 and np.min(label) >= 0

        del data

    def __getitem__(self, index):
        image = np.expand_dims(self.images[index], axis=0)
        image = torch.from_numpy(image).float()

        if self.transform is not None:
            image = self.transform(image)

        labels = torch.tensor(self.labels[index]).float()

        return image, labels

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)
