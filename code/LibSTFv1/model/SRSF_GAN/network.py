import torch
import torch.nn as nn
import torch.nn.functional as F

from LibSTFv1.model.SRSF_GAN import ops



class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = ops.BasicBlock(channel , channel // reduction, 3, 1, 3, 3)
        self.c2 = ops.BasicBlock(channel , channel // reduction, 3, 1, 5, 5)
        self.c3 = ops.BasicBlock(channel , channel // reduction, 3, 1, 7, 7)
        self.c4 = ops.BasicBlockSig((channel // reduction)*3, channel , 3, 1, 1)

    def forward(self, x):
        y = self.avg_pool(x)
        c1 = self.c1(y)
        c2 = self.c2(y)
        c3 = self.c3(y)
        c_out = torch.cat([c1, c2, c3], dim=1)
        y = self.c4(c_out)
        return x * y

class Block3(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(Block3, self).__init__()

        self.r1 = ops.ResidualBlock(in_channels, out_channels)
        self.r2 = ops.ResidualBlock(in_channels * 2, out_channels * 2)
        self.r3 = ops.ResidualBlock(in_channels * 4, out_channels * 4)
        self.g = ops.BasicBlock(in_channels * 8, out_channels, 1, 1, 0)
        self.ca = CALayer(in_channels)

    def forward(self, x):
        c0 = x

        r1 = self.r1(c0)
        c1 = torch.cat([c0, r1], dim=1)

        r2 = self.r2(c1)
        c2 = torch.cat([c1, r2], dim=1)

        r3 = self.r3(c2)
        c3 = torch.cat([c2, r3], dim=1)

        g = self.g(c3)
        out = self.ca(g)
        return out

class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),

        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = torch.abs(x - residual)
        xl = self.local_att(xa)
        xg = self.global_att(xa)

        xlg = xl+xg
        wei = self.sigmoid(xlg)

        xo = x * (1-wei) + residual * wei

        return xo
class fusionblock(nn.Module):
    def __init__(self):
        super(fusionblock, self).__init__()
        self.CA = AFF()
    def forward(self, landsat,modis):
        fusion = self.CA(landsat,modis)

        return fusion

class Generator(nn.Module):
    def __init__(self, bands):
        super(Generator, self).__init__()
        self.Maxpool = nn.Conv2d(64,64,3,2,1)
        self.renet =  Block3(64,64)
        self.re =  nn.Sequential(
            nn.Conv2d(128,64,1,1,0),
            Block3(64,64)
        )

        self.Upsample = nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.head = nn.Conv2d(4,64,1,1,0)
        self.head1 = nn.Conv2d(bands,64,1,1,0)
        self.tail = nn.Conv2d(64,bands,1,1,0)

        self.downlandsat1 = nn.Sequential(
            ops.ResidualBlock(64,64)
        )
        self.downlandsat2 = nn.Sequential(
            ops.ResidualBlock(64,64)
        )
        self.downlandsat3 = nn.Sequential(
            ops.ResidualBlock(64, 64)
        )
        self.downlandsat4 = nn.Sequential(
            ops.ResidualBlock(64, 64)
        )
        self.upmodis1 = nn.Sequential(
            ops.ResidualBlock(64, 64)
        )
        self.upmodis2 = nn.Sequential(
            ops.ResidualBlock(64, 64)
        )
        self.upmodis3 = nn.Sequential(
            ops.ResidualBlock(64, 64)
        )
        self.upmodis4 = nn.Sequential(
            ops.ResidualBlock(64, 64)
        )
        self.fusionblock = fusionblock()

    def forward(self, landsat1,modis1):
        ###########################################
        modis1 = F.interpolate(modis1, scale_factor=0.125, mode='bicubic', align_corners=True)
        ###########################################

        modis = self.head1(modis1)
        landsat1 =self.head1(landsat1)
        l1 = self.downlandsat1(landsat1)
        l2 = self.downlandsat2(self.Maxpool(l1))
        l3 = self.downlandsat3(self.Maxpool(l2))
        l4 = self.downlandsat4(self.Maxpool(l3))
        m1 =self.upmodis1(modis)
        m1s = self.Upsample(m1)
        m2 = self.upmodis2(m1s)
        m2s = self.Upsample(m2)
        m3 = self.upmodis3(m2s)
        m3s = self.Upsample(m3)
        m4 = self.upmodis4(m3s)
        fusion1 = self.fusionblock(l4,m1)
        fusion2 = self.fusionblock(l3,m2)
        fusion3 = self.fusionblock(l2,m3)
        fusion4 = self.fusionblock(l1,m4)
        fusion1 = self.Upsample(self.renet(fusion1))
        fusion2 = self.Upsample(self.re(torch.cat((fusion1,fusion2),1)))
        fusion3 = self.Upsample(self.re(torch.cat((fusion2,fusion3),1)))
        fusion4 = self.re(torch.cat((fusion3,fusion4),1))
        fusion = self.tail(fusion4)
        outs = self.tail(m4)

        return fusion,outs
#
class UpSampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        # 512-1024-512
        # 1024-512-256
        # 512-256-128
        # 256-128-64
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.LeakyReLU()
        )
        self.upsample=nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch*2,out_channels=out_ch,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU()
        )

    def forward(self,x,out):
        '''
        :param x: 输入卷积层
        :param out:与上采样层进行cat
        :return:
        '''
        x_out=self.Conv_BN_ReLU_2(x)
        x_out=self.upsample(x_out)
        cat_out=torch.cat((x_out,out),dim=1)
        return cat_out

class DownsampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_ReLU_2=nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU()
        )
        self.downsample=nn.Sequential(
            nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU()
        )

    def forward(self,x):
        """
        :param x:
        :return: out输出到深层，out_2输入到下一层，
        """
        # print(x.shape)
        out=self.Conv_BN_ReLU_2(x)
        out_2=self.downsample(out)
        return out,out_2


class Discriminator(nn.Module):
    def __init__(self, bands):
        super(Discriminator, self).__init__()
        out_channels=[2**(i+4) for i in range(5)] #[64, 128, 256, 512, 1024]
        #下采样
        self.d1=DownsampleLayer(bands,out_channels[0])#3-64
        self.d2=DownsampleLayer(out_channels[0],out_channels[1])#64-128
        self.d3=DownsampleLayer(out_channels[1],out_channels[2])#128-256
        self.d4=DownsampleLayer(out_channels[2],out_channels[3])#256-512
        #上采样
        # self.u1=UpSampleLayer(out_channels[3],out_channels[3])#512-1024-512
        self.u2=UpSampleLayer(out_channels[2],out_channels[2])#1024-512-256
        self.u3=UpSampleLayer(out_channels[3],out_channels[1])#512-256-128
        self.u4=UpSampleLayer(out_channels[2],out_channels[0])#256-128-64
        #输出
        self.o=nn.Sequential(
            nn.Conv2d(out_channels[1],out_channels[0],kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels[0],1,3,1,1),
            nn.Sigmoid(),
            # BCELoss
        )

    def forward(self,x):
        # x = x.unsqueeze(0)
        out_1,out1=self.d1(x)
        out_2,out2=self.d2(out1)
        out_3,out3=self.d3(out2)

        out6=self.u2(out3,out_3)
        out7=self.u3(out6,out_2)
        out8=self.u4(out7,out_1)
        out=self.o(out8)
        return out


if __name__ == '__main__':
    g = Generator()
    a = torch.rand([1, 3, 64, 64])
    b = torch.rand([1, 3, 64, 64])
    print(g(a,b)[0].shape)
    d = Discriminator()
    b = torch.rand([2, 3, 512, 512])
    print(d(b)[1].shape)