import torch.nn as nn
import torch.nn.functional as F
import torch


def l2_loss(pred, gt, split="train"):
    loss = F.mse_loss(pred, gt)
    log_dict = {
        f"{split}/l2_loss": loss
    }
    return loss, log_dict


