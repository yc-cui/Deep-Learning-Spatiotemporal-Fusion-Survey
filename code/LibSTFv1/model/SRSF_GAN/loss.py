
import os
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

from LibSTFv1.loss.SRSF_GAN import msssim

class VGG(nn.Module):
    def __init__(self, device):
        super(VGG, self).__init__()
        vgg = models.vgg19(False)
        pre = torch.load(r'vgg19-dcbb9e9d.pth')
        vgg.load_state_dict(pre)
        for pa in vgg.parameters():
            pa.requires_grad = False
        self.vgg = vgg.features[:16]
        self.vgg = self.vgg.to(device)

    def forward(self, x):
        out = self.vgg(x)
        return out


class ContentLoss(nn.Module):
    def __init__(self, device):
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











