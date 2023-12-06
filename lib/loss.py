"""
Losses
"""
# pylint: disable=C0301,C0103,R0902,R0915,W0221,W0622


##
# LIBRARIES
import torch
import torch.nn as nn
##
def l1_loss(input, target):
    """ L1 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L1 distance between input and output
    """

    return torch.mean(torch.abs(input - target))

##
def l2_loss(input, target, size_average=True):
    """ L2 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L2 distance between input and output
    """
    if size_average:
        return torch.mean(torch.pow((input-target), 2))
    else:
        return torch.pow((input-target), 2)

def var_loss(input, target):
    """
    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): reference tensor

    Returns:
        [FloatTensor]: KL divergence loss
    """
    #return torch.multiply(torch.mean(1 + sigma - torch.square(mu) - torch.exp(sigma), axis=-1), -5e-4)
    kl_lossfunc = nn.KLDivLoss(reduction='batchmean')
    return kl_lossfunc(input, target)