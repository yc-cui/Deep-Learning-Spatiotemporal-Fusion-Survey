import torch
import torch.nn as nn
import torch.nn.functional as F
from torchgan.layers import SpectralNorm2d
import enum

from LibSTFv1.model.GAN_STFM.normalization import SwitchNorm2d
from LibSTFv1.model.GAN_STFM.ssim import msssim



class Sampling(enum.Enum):
    UpSampling = enum.auto()
    DownSampling = enum.auto()
    Identity = enum.auto()


# NUM_BANDS = 6
PATCH_SIZE = 256
SCALE_FACTOR = 16


class Upsample(nn.Module):
    def __init__(self, scale_factor=2):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, inputs):
        return F.interpolate(inputs, scale_factor=self.scale_factor)


class ReconstructionLoss(nn.Module):
    def __init__(self, model, alpha=1.0, beta=1.0, gamma=1.0, l1_weight=10):
        super(ReconstructionLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.l1_weight = l1_weight
        self.model = model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.encoder = nn.Sequential(
            self.model.conv1,
            self.model.conv2,
            self.model.conv3,
            self.model.conv4
        )

    def forward(self, prediction, target):
        _prediction, _target = self.encoder(prediction), self.encoder(target)
        loss = (self.alpha * F.mse_loss(_prediction, _target) +
                self.gamma * (1.0 - torch.mean(F.cosine_similarity(_prediction, _target, 1))) +
                self.beta * (1.0 - msssim(prediction, target, normalize=True)))
        loss = loss + F.l1_loss(prediction, target) * self.l1_weight
        return loss


class Conv3X3NoPadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv3X3NoPadding, self).__init__(in_channels, out_channels, 3, stride=stride, padding=1)


class Conv3X3WithPadding(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv3X3WithPadding, self).__init__(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, stride=stride)
        )


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, sampling=None):
        layers = []

        if sampling == Sampling.DownSampling:
            layers.append(Conv3X3WithPadding(in_channels, out_channels, 2))
        else:
            if sampling == Sampling.UpSampling:
                layers.append(Upsample(2))
            layers.append(Conv3X3WithPadding(in_channels, out_channels))

        layers.append(nn.LeakyReLU(inplace=True))
        super(ConvBlock, self).__init__(*layers)


class ResidulBlockWtihSwitchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, sampling=None):
        super(ResidulBlockWtihSwitchNorm, self).__init__()
        channels = min(in_channels, out_channels)
        residual = [
            SwitchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            Conv3X3WithPadding(in_channels, channels),
            SwitchNorm2d(channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, out_channels, 1)
        ]
        transform = [
            Conv3X3WithPadding(in_channels, channels),
            nn.Conv2d(channels, out_channels, 1),
            nn.LeakyReLU(inplace=True)
        ]
        if sampling == Sampling.UpSampling:
            residual.insert(2, Upsample(2))
            transform.insert(0, Upsample(2))
        elif sampling == Sampling.DownSampling:
            residual[2] = Conv3X3WithPadding(in_channels, channels, 2)
            transform[0] = Conv3X3WithPadding(in_channels, channels, 2)

        self.residual = nn.Sequential(*residual)
        self.transform = nn.Sequential(*transform)

    def forward(self, inputs):
        trunk = self.residual(inputs[1])
        lateral = self.transform(inputs[0])
        return lateral, trunk + lateral


class ResidulBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sampling=None):
        super(ResidulBlock, self).__init__()
        channels = min(in_channels, out_channels)
        residual = [
            Conv3X3WithPadding(in_channels, channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, out_channels, 1)
        ]
        transform = [nn.Conv2d(in_channels, out_channels, 1)]

        if sampling == Sampling.UpSampling:
            residual.insert(0, Upsample(2))
            transform.insert(0, Upsample(2))
        elif sampling == Sampling.DownSampling:
            residual[0] = Conv3X3WithPadding(in_channels, channels, 2)
            transform[0] = nn.Conv2d(in_channels, out_channels, 1, stride=2)

        self.residual = nn.Sequential(*residual)
        self.transform = transform[0] if len(transform) == 1 else nn.Sequential(*transform)

    def forward(self, inputs):
        trunk = self.residual(inputs)
        lateral = self.transform(inputs)
        return trunk + lateral


class AutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        if out_channels is None:
            out_channels = in_channels
        super(AutoEncoder, self).__init__()
        channels = (16, 32, 64, 128)
        self.conv1 = ConvBlock(in_channels, channels[0])
        self.conv2 = ConvBlock(channels[0], channels[1], Sampling.DownSampling)
        self.conv3 = ConvBlock(channels[1], channels[2], Sampling.DownSampling)
        self.conv4 = ConvBlock(channels[2], channels[3], Sampling.DownSampling)
        self.conv5 = ConvBlock(channels[3], channels[2], Sampling.UpSampling)
        self.conv6 = ConvBlock(channels[2] * 2, channels[1], Sampling.UpSampling)
        self.conv7 = ConvBlock(channels[1] * 2, channels[0], Sampling.UpSampling)
        self.conv8 = nn.Conv2d(channels[0] * 2, out_channels, 1)

    def forward(self, inputs):
        l1 = self.conv1(inputs)
        l2 = self.conv2(l1)
        l3 = self.conv3(l2)
        l4 = self.conv4(l3)
        l5 = self.conv5(l4)
        l6 = self.conv6(torch.cat((l3, l5), 1))
        l7 = self.conv7(torch.cat((l2, l6), 1))
        out = self.conv8(torch.cat((l1, l7), 1))
        return out


class SFFusion(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        if out_channels is None:
            out_channels = in_channels
        channels = (16, 32, 64, 128)
        super(SFFusion, self).__init__()
        self.encoder = nn.Sequential(
            ResidulBlockWtihSwitchNorm(in_channels, channels[0]),
            ResidulBlockWtihSwitchNorm(channels[0], channels[1]),
            ResidulBlockWtihSwitchNorm(channels[1], channels[2]),
            ResidulBlockWtihSwitchNorm(channels[2], channels[3])
        )
        self.decoder = nn.Sequential(
            ResidulBlock(channels[3] * 2, channels[3]),
            ResidulBlock(channels[3], channels[2]),
            ResidulBlock(channels[2], channels[1]),
            ResidulBlock(channels[1], channels[0]),
            nn.Conv2d(channels[0], out_channels, 1)
        )

    def forward(self, LR_t1, LR_t2, HR_t1):
        inputs = [LR_t1, HR_t1, LR_t2]
        return self.decoder(torch.cat(self.encoder(inputs), 1))


class ResidulBlockWithSpectralNorm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidulBlockWithSpectralNorm, self).__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            SpectralNorm2d(
                Conv3X3NoPadding(in_channels, in_channels, stride=2)),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            SpectralNorm2d(
                nn.Conv2d(in_channels, out_channels, 1)),
        )
        self.transform = SpectralNorm2d(nn.Conv2d(in_channels, out_channels, 1, stride=2))

    def forward(self, inputs):
        return self.transform(inputs) + self.residual(inputs)


# 判别器网络
class Discriminator(nn.Sequential):
    def __init__(self, channels):
        modules = []
        for i in range(1, (len(channels))):
            modules.append(ResidulBlockWithSpectralNorm(channels[i - 1], channels[i]))
        modules.append(SpectralNorm2d(nn.Conv2d(channels[-1], 1, 1)))
        super(Discriminator, self).__init__(*modules)

    def forward(self, inputs):
        prediction = super(Discriminator, self).forward(inputs)
        return prediction.view(-1, 1).squeeze(1)


class MSDiscriminator(nn.Module):
    def __init__(self, num_bands):
        super(MSDiscriminator, self).__init__()
        self.d1 = Discriminator((num_bands * 2, 32, 32, 64, 64, 128, 128, 256, 256))
        self.d2 = Discriminator((num_bands * 2, 32, 64, 64, 128, 128, 256, 256))
        self.d3 = Discriminator((num_bands * 2, 32, 64, 128, 128, 256, 256))

    def forward(self, inputs):
        l1 = self.d1(inputs)
        l2 = self.d2(F.interpolate(inputs, scale_factor=0.5))
        l3 = self.d3(F.interpolate(inputs, scale_factor=0.25))
        return torch.mean(torch.stack((l1, l2, l3)))
