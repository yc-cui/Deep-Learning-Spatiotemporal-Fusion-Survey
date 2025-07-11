import torch
import torch.nn as nn
import torch.nn.functional as F
import enum
import numpy as np
from torch.autograd import Variable
from torchvision.models.vgg import vgg19


class Sampling(enum.Enum):
    UpSampling = enum.auto()
    DownSampling = enum.auto()
    Identity = enum.auto()


# NUM_BANDS = 6
# PATCH_SIZE = 256
# SCALE_FACTOR = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(nn.ReflectionPad2d(padding),
                                  nn.Conv2d(input_size, output_size, kernel_size, stride, 0, bias=bias))

        self.act = torch.nn.LeakyReLU()

    def forward(self, x):
        out = self.conv(x)

        return self.act(out)

class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DeconvBlock, self).__init__()

        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.act = torch.nn.LeakyReLU()

    def forward(self, x):
        out = self.deconv(x)

        return self.act(out)

class ResBlock(nn.Module):
    def __init__(self, in_channels, kernel_size = 3, stride = 1, padding = 1, bias = True):
        super(ResBlock, self).__init__()
        residual = [
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, in_channels,kernel_size, stride, 0, bias=bias),
            nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, in_channels,kernel_size, stride, 0, bias=bias),
            nn.Dropout(0.5),
        ]
        self.residual = nn.Sequential(*residual)


    def forward(self, inputs):
        trunk = self.residual(inputs)
        return trunk + inputs


class FeatureExtract(nn.Module):
    def __init__(self, in_channels):
        super(FeatureExtract, self).__init__()
        channels = (16, 32, 64, 128, 256)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 7, 1, 3),
            ResBlock(channels[0]),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 3, 2, 1),
            ResBlock(channels[1]),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], 3, 2, 1),
            ResBlock(channels[2]),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], 3, 2, 1),
            ResBlock(channels[3]),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(channels[3], channels[4], 3, 2, 1),
            ResBlock(channels[4]),
        )

    def forward(self, inputs):
        l1 = self.conv1(inputs)
        l2 = self.conv2(l1)
        l3 = self.conv3(l2)
        l4 = self.conv4(l3)
        l5 = self.conv5(l4)
        return [l1, l2, l3, l4, l5]

class SignificanceExtraction(nn.Module):
    def __init__(self, in_channels, ifattention = True, iftwoinput = False, outputM = False):
        super(SignificanceExtraction, self).__init__()
        self.attention = ifattention
        self.twoinput = iftwoinput
        self.outputM = outputM
        if self.attention == True:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(in_channels))
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(in_channels))
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(1),
                nn.Sigmoid())
        else:
            pass

    def forward(self, inputs):
        if self.attention == True:

            if self.twoinput == True:
                input1 = self.conv1(inputs[0])
                input2 = self.conv2(inputs[2])
            else:
                input1 = self.conv1(inputs[0])
                input2 = self.conv2(inputs[1])
            x = (input1 - input2)
            M1 = self.conv(x)
            result = inputs[0] * M1 + inputs[2] * (1 - M1)
        else:
            result = 0.5 * inputs[0] + 0.5 * inputs[2]
        if self.outputM == False:
            return result
        else:
            return result, M1

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class CombinFeatureGenerator(nn.Module):
    def __init__(self, bands, ifAdaIN = True, ifAttention = True, ifTwoInput = False, outputM = False):
        super(CombinFeatureGenerator, self).__init__()

        self.ifAdaIN = ifAdaIN
        self.ifAttention = ifAttention
        self.ifTwoInput = ifTwoInput
        self.outputM = outputM

        self.LSNet = FeatureExtract(bands)
        self.HSNet = FeatureExtract(bands)


        channels = (16, 32, 64, 128, 256)
        self.SignE_List = nn.ModuleList()
        for i in range(len(channels)):
            self.SignE_List.append(SignificanceExtraction(in_channels=channels[i],ifattention=self.ifAttention,
                                                          iftwoinput = self.ifTwoInput,outputM = self.outputM))

        self.conv1 = nn.Sequential(
            DeconvBlock(channels[4] * 2, channels[3], 4, 2, 1, bias=True),
            ResBlock(channels[3]),

        )
        self.conv2 = nn.Sequential(
            DeconvBlock(channels[3] * 2, channels[2], 4, 2, 1, bias=True),
            ResBlock(channels[2]),
        )
        self.conv3 = nn.Sequential(
            DeconvBlock(channels[2] * 2, channels[1], 4, 2, 1, bias=True),
            ResBlock(channels[1]),
        )
        self.conv4 = nn.Sequential(
            DeconvBlock(channels[1] * 2, channels[0], 4, 2, 1, bias=True),
            ResBlock(channels[0]),
        )
        self.conv5 = nn.Sequential(
            ResBlock(channels[0] * 2),
            nn.Conv2d(channels[0] * 2, channels[0], 1, 1, 0),
            ResBlock(channels[0]),
            nn.Conv2d(channels[0], bands, 1, 1, 0),
        )

    def forward(self, LR_t1, LR_t2, HR_t1):

        LS1_List = self.LSNet(LR_t2)
        LS2_List = self.LSNet(LR_t1)
        HS_List = self.HSNet(HR_t1)

        SpecFeature_List = []
        for LS1, HS in zip(LS1_List, HS_List):
            if self.ifAdaIN == True:
                SpecFeature_List.append(adaptive_instance_normalization(HS, LS1))
            elif self.ifAdaIN == False:
                SpecFeature_List.append(HS)


        FusionFeature_List = []
        M = []
        if self.outputM == False:
            for SignE, SpecFeature, LS1, LS2 in zip(self.SignE_List,SpecFeature_List,LS1_List,LS2_List):
                FusionFeature_List.append(SignE([LS1, LS2, SpecFeature]))
        else:
            for SignE, SpecFeature, LS1, LS2 in zip(self.SignE_List,SpecFeature_List,LS1_List,LS2_List):
                SignE_output0, SignE_output1 = SignE([LS1, LS2, SpecFeature])

                FusionFeature_List.append(SignE_output0)
                M.append(SignE_output1)


        l5 = self.conv1(torch.cat((FusionFeature_List[4], LS1_List[4]),dim=1))
        l4 = self.conv2(torch.cat((FusionFeature_List[3], l5),dim=1))
        l3 = self.conv3(torch.cat((FusionFeature_List[2], l4),dim=1))
        l2 = self.conv4(torch.cat((FusionFeature_List[1], l3),dim=1))
        l1 = self.conv5(torch.cat((FusionFeature_List[0], l2),dim=1))

        if self.outputM == False:
            return l1
        else:
            return l1,M



class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network

    def forward(self, high_resolution, fake_high_resolution):
        h1 = high_resolution[:, 0:3, :, :]
        fh1 = fake_high_resolution[:, 0:3, :, :]
        h2 = high_resolution[:,  3:6, :, :]
        fh2 = fake_high_resolution[:,  3:6, :, :]
        _h1 = self.loss_network(h1)
        _fh1 = self.loss_network(fh1)
        _h2 = self.loss_network(h2)
        _fh2 = self.loss_network(fh2)

        perception_loss = (F.l1_loss(_h1, _fh1) + F.l1_loss(_h2, _fh2))/2

        return perception_loss

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)
