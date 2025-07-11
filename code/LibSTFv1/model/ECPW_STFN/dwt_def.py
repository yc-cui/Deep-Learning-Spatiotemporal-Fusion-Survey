import torch
import torch.nn as nn
import torch.nn.functional as F

import pywt

def dwt2astensor(inputs, device):

    inputs1 = inputs.cpu().detach()

    LLY, (LHY, HLY, HHY) = pywt.dwt2(inputs1, 'haar')

    # print("变换后的数据类型", LLY.dtype)

    LLY1 = torch.tensor(LLY)
    LHY1 = torch.tensor(LHY)
    HLY1 = torch.tensor(HLY)
    HHY1 = torch.tensor(HHY)
    # print("第一次转换后的数据类型", LLY.dtype)

    LLY2 = LLY1.to(device)
    LHY2 = LHY1.to(device)
    HLY2 = HLY1.to(device)
    HHY2 = HHY1.to(device)
    # print("第二次转换后的数据类型", LLY.dtype)

    return LLY2, (LHY2, HLY2, HHY2)

def idwt2astensor(inputs, device):

    LLY, (LHY, HLY, HHY) = inputs

    LLY = LLY.cpu().detach().numpy()
    LHY = LHY.cpu().detach().numpy()
    HLY = HLY.cpu().detach().numpy()
    HHY = HHY.cpu().detach().numpy()

    indwt = pywt.idwt2((LLY, (LHY, HLY, HHY)), 'haar')

    indwt = torch.tensor(indwt)

    indwt = indwt.to(device)

    return indwt




