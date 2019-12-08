import os
import random
import re

import numpy as np
import torch
from skimage import transform
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from base import BaseDataLoader
from utils import elastic_deformation, load_files, norm

#data_path = '/data/OPUS_nerve_segmentation/OPUS_data_1'
#data_path = '/data/OPUS_nerve_segmentation/OPUS_data_2'
#data_path = '/data/OPUS_nerve_segmentation/OPUS_data_3'

_NUM_CLASSES = 3
_CLASS_MEDIANUS = 'medianus'
_CLASS_ULNARIS = 'ulnaris'
_CLASS_RADIALIS = 'radialis'


def class_str_to_index(class_str):
    if class_str == _CLASS_MEDIANUS:
        out = 0
    elif class_str == _CLASS_RADIALIS:
        out = 1
    elif class_str == _CLASS_ULNARIS:
        out = 2
    else:
        out = -1

    return out


# =============================================================================
# dataloader, augmentation, batch
# class for custom dataset
# read paths in __init__, actually load files in __getitem__
# patient_001 - patient_012
# =============================================================================


class OPUSDataset(Dataset):

    def __init__(self, phase, data_path, transform=None):

        self.transform = transform
        self.phase = phase

        self.image_list = list()
        self.us_list = list()
        self.labels_list = list()
        self.classes_list = list()
        self.patients_list = list()

        # =============================================================================
        # TRAINING files
        # =============================================================================
        if phase == 'train':
            ## patients for training
            # self.patients_list = ('patient_004', 'patient_005', 'patient_006', 'patient_007', 'patient_008', 'patient_009', 'patient_010', 'patient_011') # crossval_1
            # self.patients_list = ('patient_002', 'patient_003', 'patient_006', 'patient_007', 'patient_008', 'patient_009', 'patient_010', 'patient_011') # crossval_2
            # self.patients_list = ('patient_002', 'patient_003', 'patient_004', 'patient_005', 'patient_008', 'patient_009', 'patient_010', 'patient_011') # crossval_3
            # self.patients_list = ('patient_002','patient_003','patient_004', 'patient_005', 'patient_006', 'patient_007','patient_010', 'patient_011') # crossval_4
            # self.patients_list = ('patient_002', 'patient_003', 'patient_004', 'patient_005', 'patient_006', 'patient_007', 'patient_008', 'patient_009') # crossval_5
            self.patients_list = ('patient_001', 'patient_002', 'patient_003', 'patient_004', 'patient_005', 'patient_006', 'patient_007', 'patient_008', 'patient_009', 'patient_010')  # complete

        # =============================================================================
        # VALIDATION files
        # =============================================================================
        if phase == 'val':
            ### patients for validation
            # self.patients_list = ('patient_002','patient_003') # crossval_1
            # self.patients_list = ('patient_004','patient_005') # crossval_2
            # self.patients_list = ('patient_006','patient_007') # crossval_3
            # self.patients_list = ('patient_008','patient_009') # crossval_4
            self.patients_list = ('patient_011',)  # crossval_5/ complete

        # =============================================================================
        # TESTING files
        # =============================================================================
        if phase == 'test':
            ### patients for testing
            self.patients_list = ('patient_012',)

        # Load patient data
        for x in self.patients_list:
            data_path_patient = os.path.join(data_path, x)
            self._load_patient(data_path_patient)

        # Sort all lists by image_list
        sorted_data = sorted(zip(self.image_list, self.labels_list, self.classes_list))
        self.image_list = [a for a, _, _ in sorted_data]
        self.labels_list = [b for _, b, _ in sorted_data]
        self.classes_list = [c for _, _, c in sorted_data]

    def _load_patient(self, data_path_patient):
        """Load patient data from path"""

        for nerve_class in [_CLASS_MEDIANUS, _CLASS_ULNARIS, _CLASS_RADIALIS]:

            # OPUS data 2/3 contains 'OPUS' for image_list, 'ROI' for labels_list
            # NOTE: OPUS data additionally contains 'reconOA' for image_list, 'reconUS' for us_list
            opus_path = os.path.join(data_path_patient, nerve_class, 'OPUS')
            roi_path = os.path.join(data_path_patient, nerve_class, 'ROI')

            # Load image and class
            label_files = os.listdir(roi_path)
            for img_filename in os.listdir(opus_path):
                if not img_filename.startswith(('.', '@')):
                    img_case_num, img_id = self._parse_sample_filename(img_filename)
                    if img_case_num is not None:
                        for label_filename in label_files:
                            label_case_num, label_id = self._parse_sample_filename(label_filename)

                            # If the image and label files have the same case number and id
                            if img_case_num == label_case_num and img_id == label_id:
                                self.image_list.append(os.path.join(opus_path, img_filename))
                                self.classes_list.append(nerve_class)
                                self.labels_list.append(os.path.join(roi_path, label_filename))
                                break

    def _parse_sample_filename(self, filename):
        # regular expression for the image/label number in the filenames
        # Example: 'OPUS_NNMF_48_05.mat' where '48' is the case number (group 4) and '05' is the sample id (group 5) and
        # group 2 is '_48_05.'
        file_re = r'(\w+)(_((\d+)_(\d+))\.)\w+$'

        matches = re.search(file_re, filename)

        if matches is not None and matches.group(2) is not None:
            case_num = int(matches.group(4))
            sample_id = int(matches.group(5))
        else:
            case_num = None
            sample_id = None

        return case_num, sample_id


    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, idx):

        image = load_files(self.image_list[idx])
        labels = load_files(self.labels_list[idx])

        if labels.ndim < 3:
            labels = np.expand_dims(labels, axis=2)

        # US only: change image to us, dict
        sample = {'image': image, 'labels': labels}
        if self.transform:
            sample = self.transform(sample)

        # TODO: Need to break apart dictionary and squeeze data so that it fits into the framework. Modify framework to
        #  accept sample tuple
        sample['labels'] = sample['labels'].squeeze()
        return sample['image'].float(), sample['labels'].float()
        #return sample


# =============================================================================
# Transforms
# =============================================================================


class RandomHorizontalFlip(object):

    def __init__(self, p):
        self.p = p

    def __call__(self, data):
        img = data['image']
        lab = data['labels']

        if random.random() < self.p:
            img = np.flip(img, axis=1)
            lab = np.flip(lab, axis=1)

        return {'image': img, 'labels': lab}


class elastic_deform(object):

    def __init__(self, p):
        self.p = p

    def __call__(self, data):
        img = data['image']
        lab = data['labels']

        x_coo = np.random.randint(100, 300)
        y_coo = np.random.randint(100, 300)
        dx = np.random.randint(10, 40)
        dy = np.random.randint(10, 40)
        if random.random() < self.p:
            img = elastic_deformation(img, x_coo, y_coo, dx, dy)
            lab = elastic_deformation(lab, x_coo, y_coo, dx, dy)

            lab = np.where(lab <= 20, 0, lab)
            lab = np.where(lab > 20, 255, lab)

        return {'image': img, 'labels': lab}


class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        h, w, d = image.shape[:3]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h

        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w, d), mode='constant')
        labels = transform.resize(labels, (new_h, new_w), mode='constant')

        labels = np.where(labels <= 0.5, 0, 1)  # for loss function

        return {'image': img, 'labels': labels}


class ToTensor(object):
    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        image = norm(image)

        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image),
                'labels': torch.from_numpy(labels)}


class OPUSDataLoader(BaseDataLoader):
    """
    OPUS data loader
    """

    def __init__(self,
                 data_dir,
                 batch_size,
                 shuffle=True,
                 validation_split=0.0,
                 num_workers=1,
                 training=True,
                 input_size=400,
                 augmentation_probability=0.5):

        self.data_dir = data_dir
        self.input_size = input_size
        self.augmentation_probability = augmentation_probability
        if training:
            self.dataset = OPUSDataset('train', data_path=data_dir, transform=transforms.Compose([
                elastic_deform(augmentation_probability),
                Rescale(input_size),
                ToTensor()]))
        else:
            self.dataset = OPUSDataset('test', data_path=data_dir, transform=transforms.Compose([
                Rescale(input_size),
                ToTensor()
            ]))

        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, num_workers, test_config=None)

    def split_validation(self):
        transformed_dataset_val = OPUSDataset('val', self.data_dir,
                                              transform=transforms.Compose([
                                                  Rescale(self.input_size),
                                                  ToTensor()]))
        batch_size = self.init_kwargs['batch_size']
        num_workers = self.init_kwargs['num_workers']
        return DataLoader(transformed_dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=True)
