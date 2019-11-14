import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import pickle

class LIDC_IDRI(Dataset):
    images = []
    labels = []
    series_uid = []
    
    enable_random_selection = True

    def __init__(self, data, enable_random_selection=True, start_index=-1, end_index=-1, transform=None):
        self.enable_random_selection = enable_random_selection
        self.transform = transform

        idx = 0
        for key, value in data.items():
            if start_index < idx < end_index:
                self.images.append(value['image'].astype(float))
                self.labels.append(value['masks'])
                self.series_uid.append(value['series_uid'])
            idx += 1

        assert (len(self.images) == len(self.labels) == len(self.series_uid))

        for img in self.images:
            assert np.max(img) <= 1 and np.min(img) >= 0
        for label in self.labels:
            assert np.max(label) <= 1 and np.min(label) >= 0

    def __getitem__(self, index):
        image = np.expand_dims(self.images[index], axis=0)
        
        #Randomly select one of the four labels for this image
        selected_label_idx = random.randint(0,3) if self.enable_random_selection else 0
        label = self.labels[index][selected_label_idx].astype(float)
        if self.transform is not None:
            image = self.transform(image)

        series_uid = self.series_uid[index]

        # Convert image and label to torch tensors
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        #Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)

        return image, label, series_uid

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)