import os
import pickle

from torch.utils.data import Dataset
from torchvision import transforms
import torch
from dataset_loader.load_LIDC_data import LIDC_IDRI


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


def get_lidc_loaders(dataset_path='data/data_lidc.pickle',
                     batch_size_train=5,
                     batch_size_val=5,
                     batch_size_test=3,
                     train_indices=(0, 50),
                     val_indices=(60, 75),
                     test_indices=(90, 100),
                     num_graders=1):
    lidc_data = load_pickle_file(dataset_path)

    train_dataset = LIDC_IDRI(lidc_data, num_graders, train_indices[0], train_indices[1])
    val_dataset = LIDC_IDRI(lidc_data, num_graders, val_indices[0], val_indices[1])
    test_dataset = LIDC_IDRI(lidc_data, num_graders, test_indices[0], test_indices[1])
    del lidc_data

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    print("Dataloaders are ready!")

    return train_dataloader, val_dataloader, test_dataloader
