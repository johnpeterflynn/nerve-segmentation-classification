import torch.nn.functional as F
from nn_common_modules import losses as additional_losses
import torch
from torch.nn.modules.loss import _Loss


def nll_loss(output, target):
    return F.nll_loss(output, target)

# TODO: Add the weights options to the losses below
def dice(output, target):
    criterion = additional_losses.DiceLoss()
    return criterion(output, target.long())

def iou_loss(output, target):
    criterion = additional_losses.IoULoss()
    return criterion(output, target.long())

def crossentropy_plu_loss(output, target):
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(output.double().cpu(), target.long().cpu()) 


def combined_plus(output, target):
    raise Exception("Hey that's not implemented!")

def cross_entropy_loss_2d(output, target):
    criterion = additional_losses.CrossEntropyLoss2d()
    return criterion(output.double().cpu(), target.long().cpu())

def combined_loss(output, target_seg, target_cl, weights=None):
    criterion = CombinedLoss()
    return criterion(output, target_seg.long(), target_cl.long(), weights)

class CombinedLoss(_Loss):
    """
    A combination of dice  and cross entropy loss
    """

    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = additional_losses.CrossEntropyLoss2d()
        self.dice_loss = additional_losses.DiceLoss()

    def forward(self, input, target_seg, target_cl, weight=None):
        """
        Forward pass

        :param input: torch.tensor (NxCxHxW)
        :param target: torch.tensor (NxHxW)
        :param weight: torch.tensor (NxHxW)
        :return: scalar
        """

        seg, cl = input

        y_2 = torch.mean(self.dice_loss(seg, target_seg))
        if weight is None:
            y_1 = torch.mean(self.cross_entropy_loss.forward(cl, target_cl))
        else:
            y_1 = torch.mean(
                torch.mul(self.cross_entropy_loss.forward(cl, target_cl), weight.cuda()))

        #y_1 = 0 * y_1
        #y_2 = 0 * y_2

        print('DL, CEL: ', y_2.item(), ', ', y_1.item())
        return y_1 + y_2