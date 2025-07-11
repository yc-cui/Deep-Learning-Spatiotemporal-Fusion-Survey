import torch
import torch.nn as nn
import torch.nn.functional as F
from LibSTFv1.model.EDCSTFN.ssim import msssim

#NUM_BANDS = 6


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.ReplicationPad2d(1),
        nn.Conv2d(in_channels, out_channels, 3, stride=stride)
    )


def interpolate(inputs, size=None, scale_factor=None):
    return F.interpolate(inputs, size=size, scale_factor=scale_factor,
                         mode='bilinear', align_corners=True)


class CompoundLoss(nn.Module):
    def __init__(self, pretrained, alpha=0.8, normalize=True):
        super(CompoundLoss, self).__init__()
        self.pretrained = pretrained.encoder
        self.alpha = alpha
        self.normalize = normalize

    def forward(self, prediction, target):
        return (F.mse_loss(prediction, target) +
                F.mse_loss(self.pretrained(prediction), self.pretrained(target)) +
                self.alpha * (1.0 - msssim(prediction, target,
                                           normalize=self.normalize)))


class FEncoder(nn.Sequential):
    def __init__(self, num_bands):
        channels = [num_bands, 32, 64, 128]
        super(FEncoder, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            conv3x3(channels[2], channels[3]),
            nn.ReLU(True)
        )


class REncoder(nn.Sequential):
    def __init__(self, num_bands):
        channels = [num_bands * 3, 32, 64, 128]
        super(REncoder, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            conv3x3(channels[2], channels[3])
        )


class Decoder(nn.Sequential):
    def __init__(self, num_bands):
        channels = [128, 64, 32, num_bands]
        super(Decoder, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            nn.Conv2d(channels[2], channels[3], 1)
        )


class Pretrained(nn.Sequential):
    def __init__(self, num_bands):
        channels = [num_bands, 32, 64, 128, 128]
        super(Pretrained, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            conv3x3(channels[2], channels[3], 2),
            nn.ReLU(True),
            conv3x3(channels[3], channels[4]),
            nn.ReLU(True)
        )


class FusionNet(nn.Module):
    def __init__(self, num_bands):
        super(FusionNet, self).__init__()
        self.num_bands = num_bands
        self.encoder = FEncoder(num_bands)
        self.residual = REncoder(num_bands)
        self.decoder = Decoder(num_bands)

    def forward(self, LR_t1, LR_t2, HR_t1):
        inputs = [LR_t1, HR_t1, LR_t2]
        # inputs[0] = interpolate(inputs[0], scale_factor=16)
        # inputs[-1] = interpolate(inputs[-1], scale_factor=16)
        prev_diff = self.residual(torch.cat((inputs[0], inputs[1], inputs[-1]), 1))

        if len(inputs) == 5:
            inputs[2] = interpolate(inputs[2], scale_factor=16)
            next_diff = self.residual(torch.cat((inputs[2], inputs[3], inputs[-1]), 1))
            if self.training:
                prev_fusion = self.encoder(inputs[1]) + prev_diff
                next_fusion = self.encoder(inputs[3]) + next_diff
                return self.decoder(prev_fusion), self.decoder(next_fusion)
            else:
                one = inputs[0].new_tensor(1.0)
                epsilon = inputs[0].new_tensor(1e-8)
                prev_dist = torch.abs(prev_diff) + epsilon
                next_dist = torch.abs(next_diff) + epsilon
                prev_mask = one.div(prev_dist).div(one.div(prev_dist) + one.div(next_dist))
                prev_mask = prev_mask.clamp_(0.0, 1.0)
                next_mask = one - prev_mask
                result = (prev_mask * (self.encoder(inputs[1]) + prev_diff) +
                          next_mask * (self.encoder(inputs[3]) + next_diff))
                result = self.decoder(result)
                return result
        else:
            return self.decoder(self.encoder(inputs[1]) + prev_diff)
