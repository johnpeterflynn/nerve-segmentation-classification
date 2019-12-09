"""Quicknat architecture"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nn_common_modules import modules as sm
from squeeze_and_excitation import squeeze_and_excitation as se
from base import BaseModel


class SimpleNet(BaseModel):
    """
    A PyTorch implementation of QuickNAT

    """

    def __init__(self, params):
        """

        :param params: {'num_channels':1,
                        'num_filters':64,
                        'kernel_h':5,
                        'kernel_w':5,
                        'stride_conv':1,
                        'pool':2,
                        'stride_pool':2,
                        'num_classes':28
                        'se_block': False,
                        'drop_out':0.2}
        """
        super(SimpleNet, self).__init__()

        self.conv1 = nn.Conv2d(7, 10, 5)
        self.conv1_bn = nn.BatchNorm2d(10)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 16, 5)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 9409, 10)
        self.fc1_bn = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(10, 3)


    def forward(self, input):
        """

        :param input: X
        :return: probabiliy map
        """
        x = self.pool(F.relu(self.conv1_bn(self.conv1(input))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = x.view(-1, 16 * 9409)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc3(x)
        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with '*.model'.

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
