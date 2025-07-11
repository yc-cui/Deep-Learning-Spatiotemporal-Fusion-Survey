import torch
import torch.nn as nn
import torch.nn.functional as F

import enum

from LibSTFv1.loss.ECPW_STFN import msssim



class Sampling(enum.Enum):
    UPSAMPLING = enum.auto()
    DOWNSAMPLING = enum.auto()
    NOSAMPLING = enum.auto()


def conv3x3(in_channels, out_channels, stride=1, padding='same'):
    if padding == 'same':
        return nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, stride=stride)
        )
    else:
        return nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)


class Upsample(nn.Module):
    def __init__(self, scale_factor=2):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, inputs):
        return F.interpolate(inputs, scale_factor=self.scale_factor)


def conv_block(in_channels, out_channels, sampling=None):
    if sampling == Sampling.UPSAMPLING:
        return nn.Sequential(
            conv3x3(in_channels, out_channels),
            Upsample(2),
            nn.ReLU(True)
        )
    elif sampling == Sampling.DOWNSAMPLING:
        return nn.Sequential(
            nn.MaxPool2d(2),
            conv3x3(in_channels, out_channels),
            nn.ReLU(True)
        )
    return nn.Sequential(
        conv3x3(in_channels, out_channels),
        nn.ReLU(True)
    )


class VisionLoss(nn.Module):
    def __init__(self, alpha=0.84, window_size=11, size_average=True, normalize=True):
        super(VisionLoss, self).__init__()
        self.alpha = alpha
        self.window_size = window_size
        self.size_average = size_average
        self.normalize = normalize

    def forward(self, prediction, target):
        return (F.smooth_l1_loss(prediction, target) + self.alpha * (
                1.0 - msssim(prediction, target, window_size=self.window_size,
                             size_average=self.size_average, normalize=self.normalize)))


class AutoEncoder(nn.Module):
    def __init__(self, bands):
        super(AutoEncoder, self).__init__()
        channels = (bands, 16, 32, 64, 128)
        self.conv1 = conv_block(channels[0], channels[1])
        self.conv2 = conv_block(channels[1], channels[2], Sampling.DOWNSAMPLING)
        self.conv3 = conv_block(channels[2], channels[3], Sampling.DOWNSAMPLING)
        self.conv4 = conv_block(channels[3], channels[4], Sampling.DOWNSAMPLING)
        self.conv5 = conv_block(channels[4], channels[3], Sampling.UPSAMPLING)
        self.conv6 = conv_block(channels[3] * 2, channels[2], Sampling.UPSAMPLING)
        self.conv7 = conv_block(channels[2] * 2, channels[1], Sampling.UPSAMPLING)
        self.conv8 = nn.Conv2d(channels[1] * 2, channels[0], 1)

    def forward(self, inputs, only_encoder=False):
        # inputs = inputs.unsqueeze(0) # 训练时注释
        l1 = self.conv1(inputs)
        l2 = self.conv2(l1)
        l3 = self.conv3(l2)
        l4 = self.conv4(l3)
        if only_encoder:
            return l4
        l5 = self.conv5(l4)
        l6 = self.conv6(torch.cat((l3, l5), 1))
        l7 = self.conv7(torch.cat((l2, l6), 1))
        l8 = self.conv8(torch.cat((l1, l7), 1))
        return l8
