import torch
import torch.nn as nn
from torch.nn import functional as F
from LibSTFv1.model.SwinSTFM.pytorch_ssim import msssim


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.perceptual_cri = CharbonnierLoss(loss_weight=1.0)
        self.gamma = 1.0

    def forward(self, prediction, target, is_ds=False):
        if not is_ds:

            loss = self.perceptual_cri(prediction, target) + \
                   self.gamma * (1 - msssim(prediction, target, val_range=2., normalize='relu'))
        else:
            loss = self.perceptual_cri(prediction, target)
        return loss


class CharbonnierLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(CharbonnierLoss, self).__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = 1e-3

    def forward(self, pred, target):
        return self.loss_weight * torch.mean(torch.sqrt(torch.square(target - pred) + self.eps * self.eps))
