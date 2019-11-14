import os
import pickle

from torch.utils.data import Dataset
from torchvision import transforms
import torch
from dataset_loader.load_LIDC_data import LIDC_IDRI

LIDC_PATH = 'data/'

TRAIN_INDICES = (0, 50)
VAL_INDICES = (60, 75)
TEST_INDICES = (90, 100)

TRAIN_BATCH_SIZE = 5
VAL_BATCH_SIZE = 5
TEST_BATCH_SIZE = 3


def load_pickle_file(dataset_location):
    max_bytes = 2 ** 31 - 1
    data = {}
    for file in os.listdir(dataset_location):
        filename = os.fsdecode(file)
        if '.pickle' in filename:
            print("Loading file", filename)
            file_path = dataset_location + filename
            bytes_in = bytearray(0)
            input_size = os.path.getsize(file_path)
            with open(file_path, 'rb') as f_in:
                for _ in range(0, input_size, max_bytes):
                    bytes_in += f_in.read(max_bytes)
            new_data = pickle.loads(bytes_in)
            data.update(new_data)
    del new_data
    return data


def get_lidc_loaders():
    lidc_data = load_pickle_file(LIDC_PATH)

    train_dataset = LIDC_IDRI(lidc_data, False, TRAIN_INDICES[0], TRAIN_INDICES[1])
    val_dataset = LIDC_IDRI(lidc_data, False, VAL_INDICES[0], VAL_INDICES[1])
    test_dataset = LIDC_IDRI(lidc_data, False, TEST_INDICES[0], TEST_INDICES[1])
    del lidc_data

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    print("Dataloaders are ready!")

    return train_dataloader, val_dataloader, test_dataloader
