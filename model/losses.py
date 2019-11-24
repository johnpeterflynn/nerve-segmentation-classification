"""
Description
++++++++++++++++++++++
Addition losses module defines classses which are commonly used particularly in segmentation and are not part of standard pytorch library.
Usage
++++++++++++++++++++++
Import the package and Instantiate any loss class you want to you::
    from nn_common_modules import losses as additional_losses
    loss = additional_losses.DiceLoss()
    Note: If you use DiceLoss, insert Softmax layer in the architecture. In case of combined loss, do not put softmax as it is in-built
Members
++++++++++++++++++++++
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss
import numpy as np
from torch.autograd import Variable
from nn_common_modules import losses as additional_losses


class CombinedLoss_KLdiv(_Loss):
    """
    A combination of dice  and cross entropy loss
    """

    def __init__(self):
        super(CombinedLoss_KLdiv, self).__init__()
        self.cross_entropy_loss = CrossEntropyLoss2d()
        self.dice_loss = DiceLoss()

    def forward(self, input, target, weight=None):
        """
        Forward pass
        """
        input, kl_div_loss = input
        # input_soft = F.softmax(input, dim=1)
        y_2 = torch.mean(self.dice_loss(input, target))
        if weight is None:
            y_1 = torch.mean(self.cross_entropy_loss.forward(input, target))
        else:
            y_1 = torch.mean(
                torch.mul(self.cross_entropy_loss.forward(input, target), weight))
        return y_1, y_2, kl_div_loss
