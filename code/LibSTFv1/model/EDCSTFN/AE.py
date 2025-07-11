import torch.nn as nn
import torch.nn.functional as F
from LibSTFv1.model.EDCSTFN.ssim import msssim


#NUM_BANDS = 6


def conv3x3(in_channels, out_channels, stride=1, padding='same'):
    if padding == 'same':
        return nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, stride=stride)
        )
    else:
        return nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)


class VisionLoss(nn.Module):
    def __init__(self, alpha=0.8, window_size=11, size_average=True, normalize=True):
        super(VisionLoss, self).__init__()
        self.alpha = alpha
        self.window_size = window_size
        self.size_average = size_average
        self.normalize = normalize

    def forward(self, prediction, target):
        return (F.l1_loss(prediction, target) + self.alpha * (
                1.0 - msssim(prediction, target, window_size=self.window_size,
                             size_average=self.size_average, normalize=self.normalize)))


class Upsample(nn.Module):
    def __init__(self, scale_factor=2):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, inputs):
        return F.interpolate(inputs, scale_factor=self.scale_factor,
                             mode='bicubic', align_corners=False)


class Encoder(nn.Sequential):
    def __init__(self, num_bands=6): ##
        channels = [num_bands, 32, 64, 128, 128]
        super(Encoder, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            conv3x3(channels[2], channels[3], 2),
            nn.ReLU(True),
            conv3x3(channels[3], channels[4]),
            nn.ReLU(True)
        )


class Decoder(nn.Sequential):
    def __init__(self, num_bands=6):
        channels = [128, 128, 64, 32, num_bands]
        super(Decoder, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            Upsample(2),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            conv3x3(channels[2], channels[3]),
            nn.ReLU(True),
            nn.Conv2d(channels[3], channels[4], 1)
        )


class AutoEncoder(nn.Module):
    def __init__(self, num_bands=6):
        super(AutoEncoder, self).__init__()
        self.num_bands = num_bands #新增
        self.encoder = Encoder(num_bands)
        self.decoder = Decoder(num_bands)

    def forward(self, inputs):
        out = self.encoder(inputs)
        return self.decoder(out)