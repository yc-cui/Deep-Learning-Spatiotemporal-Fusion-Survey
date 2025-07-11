import os
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
code comes from
https://github.com/jorge-pessoa/pytorch-msssim.git
"""
import torch
import torch.nn.functional as F
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                          for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        max_val = 255 if torch.max(img1) > 128 else 1
        min_val = -1 if torch.min(img1) < -0.5 else 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average,
                       full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models,
    # not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size,
                    size_average=self.size_average)


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, normalize=False):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.normalize = normalize

    def forward(self, img1, img2):
        return msssim(img1, img2, window_size=self.window_size,
                      size_average=self.size_average, normalize=self.normalize)



class VGG(nn.Module):
    def __init__(self, device=None):
        super(VGG, self).__init__()
        vgg = models.vgg19(False)
        pre = torch.load(r'LibSTFv1/model/SRSF_GAN/vgg19-dcbb9e9d.pth')
        vgg.load_state_dict(pre)
        for pa in vgg.parameters():
            pa.requires_grad = False
        self.vgg = vgg.features[:16]
        # self.vgg = self.vgg.to(device)

    def forward(self, x):
        out = self.vgg(x)
        return out


class ContentLoss(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.mse = nn.MSELoss()
        self.vgg19 = VGG(device)

    def forward(self, fake, real):
        feature_fake = self.vgg19(fake)
        feature_real = self.vgg19(real)
        loss = self.mse(feature_fake, feature_real)
        return loss


class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x,y):
        loss = 0.5 * torch.mean((x - y)**2)
        return loss



class RegularizationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        a = torch.mul(
            (x[:, :, :x.shape[2]-1, :x.shape[3]-1] - x[:, :, 1:x.shape[2], :x.shape[3]-1]),(x[:, :, :x.shape[2]-1, :x.shape[3]-1] - x[:, :, 1:x.shape[2], :x.shape[3]-1])
        )
        b = torch.mul(
            (x[:, :, :x.shape[2]-1, :x.shape[3]-1] - x[:, :, :x.shape[2]-1, 1:x.shape[3]]),(x[:, :, :x.shape[2]-1, :x.shape[3]-1] - x[:, :, :x.shape[2]-1, 1:x.shape[3]])
        )
        loss = torch.sum(torch.pow(a+b, 1.25))
        return loss
def compute_gradient(inputs):
    kernel_v = [[-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]]
    kernel_h = [[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]]
    kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0).to(inputs.device)
    kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0).to(inputs.device)
    gradients = []
    for i in range(inputs.shape[1]):
        data = inputs[:, i]
        data_v = F.conv2d(data.unsqueeze(1), kernel_v, padding=1)
        data_h = F.conv2d(data.unsqueeze(1), kernel_h, padding=1)
        data = torch.sqrt(torch.pow(data_v, 2) + torch.pow(data_h, 2) + 1e-6)
        gradients.append(data)

    result = torch.cat(gradients, dim=1)
    return result


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, prediction, target):

        vision_loss = 1.0 - msssim(prediction, target,normalize=True)
        loss =  vision_loss
        return loss

if __name__ == '__main__':

    img = torch.rand([1, 3, 64, 64])
    r = ReconstructionLoss()
    print(r(img,img))










