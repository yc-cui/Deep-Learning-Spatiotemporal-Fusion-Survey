import torch
import torch.nn as nn
import torch.nn.functional as F

from LibSTFv1.loss.ECPW_STFN import msssim
from LibSTFv1.model.ECPW_STFN.dwt_def import dwt2astensor, idwt2astensor
from LibSTFv1.model.ECPW_STFN.wavelet import DWT_Haar, IWT_Haar

import numpy as np





def conv3x3(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.ReplicationPad2d(1),
        nn.Conv2d(in_channels, out_channels, 3, stride=stride)
    )


# loss函数**************************
class CompoundLoss(nn.Module):
    def __init__(self, pretrained, alpha=0.85, normalize=True):
        super(CompoundLoss, self).__init__()
        self.pretrained = pretrained
        self.alpha = alpha
        self.normalize = normalize
        self.DWT = DWT_Haar()
        self.IWT = IWT_Haar()

    def forward(self, prediction, target):

        prediction_LL, prediction_HL, prediction_LH, prediction_HH = self.DWT(prediction)
        target_LL, target_HL, target_LH, target_HH = self.DWT(target)
        LL_loss = F.l1_loss(prediction_LL, target_LL)
        HL_loss = F.l1_loss(prediction_HL, target_HL)
        LH_loss = F.l1_loss(prediction_LH, target_LH)
        HH_loss = F.l1_loss(prediction_HH, target_HH)

        wavelet_loss = LL_loss + HL_loss + LH_loss + HH_loss
        feature_loss = F.l1_loss(self.pretrained(prediction, only_encoder=True), self.pretrained(target, only_encoder=True))
        vision_loss = self.alpha * (1.0 - msssim(prediction, target,normalize=self.normalize))

        loss = wavelet_loss + feature_loss + vision_loss
        return loss

class FEncoder(nn.Sequential):
    def __init__(self, bands):
        channels = [bands, 32, 64, 128]
        super(FEncoder, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            conv3x3(channels[2], channels[3]),
            nn.ReLU(True)
        )

class REncoder(nn.Sequential):
    def __init__(self, bands):
        channels = [bands, 32, 64, 128]
        super(REncoder, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            conv3x3(channels[2], channels[3]),
            nn.ReLU(True)
        )

class Decoder(nn.Sequential):
    def __init__(self, bands):
        channels = [128, 64, 32, bands]
        super(Decoder, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            nn.Conv2d(channels[2], channels[3], 1)
        )


class Conv3X3WithPadding(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv3X3WithPadding, self).__init__(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, stride=stride)
        )

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 输入两个通道，一个是maxpool 一个是avgpool的
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 对池化完的数据cat 然后进行卷积
        return self.sigmoid(x)

class ResidulBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidulBlock, self).__init__()
        residual = [
            Conv3X3WithPadding(in_channels, out_channels),
            nn.ReLU(True),
        ]
        ChannelA = [ChannelAttention(out_channels)]
        SpatialA = [SpatialAttention()]

        self.residual = nn.Sequential(*residual)
        self.ChannelA = nn.Sequential(*ChannelA)
        self.SpatialA = nn.Sequential(*SpatialA)

    def forward(self, inputs):
        residualfeature = self.residual(inputs)
        CA = self.ChannelA(residualfeature)
        channelrefined = residualfeature*CA
        SA = self.SpatialA(channelrefined)
        refined = channelrefined*SA
        return refined

class RFEncoder(nn.Sequential):
    def __init__(self, bands):
        channels = [bands*2, 32, 64, 128]
        super(RFEncoder, self).__init__(
            ResidulBlock(channels[0], channels[1]),
            ResidulBlock(channels[1], channels[2]),
            ResidulBlock(channels[2], channels[3]),
        )

class RFDecoder(nn.Sequential):
    def __init__(self, bands):
        channels = [128, 64, 32, bands]
        super(RFDecoder, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            nn.Conv2d(channels[2], channels[3], 1)
        )

class Pretrained(nn.Sequential):
    def __init__(self, bands):
        channels = [bands, 32, 64, 128]
        super(Pretrained, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2], 2),
            nn.ReLU(True),
            conv3x3(channels[2], channels[3], 2),
            nn.ReLU(True)
        )

class FusionNet_2(nn.Module):
    def __init__(self, bands):
        super(FusionNet_2, self).__init__()
        self.Landsat_encoder = FEncoder(bands)
        self.MLRes_encoder = REncoder(bands)
        self.decoder = Decoder(bands)
        # self.device = torch.device('cuda:0')
        # 计算数据量时改为 cpu
        self.RFEncoder = RFEncoder(bands)
        self.RFDecoder = RFDecoder(bands)

    def forward(self, Modis_ref, Modis_pre, Land_ref):

        # inputs[0]:低分参考 Modis_ref
        # inputs[1]:高分参考 Land_ref
        # inputs[-1]:低分预测 Modis_pre
        device = Modis_ref.device

        res_data = Modis_pre - Land_ref

        # 高分小波分量
        Land_L, (Land_H1, Land_H2, Land_H3) = dwt2astensor(Land_ref, device)
        # 差分小波分量
        res_L, (res_H1, res_H2, res_H3) = dwt2astensor(res_data, device)

        # 高分特征提取
        Landsat_L_encoder = self.Landsat_encoder(Land_L)
        Landsat_H1_encoder = self.Landsat_encoder(Land_H1)
        Landsat_H2_encoder = self.Landsat_encoder(Land_H2)
        Landsat_H3_encoder = self.Landsat_encoder(Land_H3)

        # 变化特征提取
        MLRes_L_encoder = self.MLRes_encoder(res_L)
        MLRes_H1_encoder = self.MLRes_encoder(res_H1)
        MLRes_H2_encoder = self.MLRes_encoder(res_H2)
        MLRes_H3_encoder = self.MLRes_encoder(res_H3)

        # 高分特征 + 变化特征 = 预测特征
        L_encoder = Landsat_L_encoder + MLRes_L_encoder
        H1_encoder = Landsat_H1_encoder + MLRes_H1_encoder
        H2_encoder = Landsat_H2_encoder + MLRes_H2_encoder
        H3_encoder = Landsat_H3_encoder + MLRes_H3_encoder

        # 小波逆变换
        result_encoder = idwt2astensor((L_encoder,(H1_encoder, H2_encoder, H3_encoder)), device)

        # 预测特征重构得到预测影像
        result_pre = self.decoder(result_encoder)

        # 自适应增强
        result_rfe = self.RFEncoder(torch.cat((result_pre, Land_ref), 1))
        result = self.RFDecoder(result_rfe)

        return result