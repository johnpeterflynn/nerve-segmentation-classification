import torch.nn.functional as F
from nn_common_modules import losses as additional_losses
import torch

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

def combined_loss(output, target, weights=None):
    criterion = additional_losses.CombinedLoss()
    return criterion(output, target.long(), weights)

