from matplotlib.patheffects import Stroke, Normal
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import os
import torchmetrics.functional.image as MF
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import pandas as pd
import time
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from LibSTFv1.base.base_model import BaseModel
from LibSTFv1.loss.SRSF_GAN import ReconstructionLoss, AdversarialLoss, ContentLoss
from LibSTFv1.metric import cross_correlation
from LibSTFv1.model.SRSF_GAN.network import Generator, Discriminator
from LibSTFv1.util.misc import check_and_make, regularize_inputs
from sorcery import dict_of
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
from io import BytesIO
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
plt.rcParams['mathtext.fontset'] = 'cm'
sns.set(style="ticks", palette="bright")


class SRSFGANModel(BaseModel):
    def __init__(self,
                 epochs,
                 bands,
                 rgb_c,
                 dataname
                 ):
        super().__init__(
                 epochs,
                 bands,
                 rgb_c,
                 dataname
                 )
        self.automatic_optimization = False

        self.rgb_c = rgb_c
        self.generator = Generator(bands)
        self.dnet = Discriminator(bands)
        self.ContentLoss = ReconstructionLoss()
        self.AdversarialLoss = AdversarialLoss()
        self.Perceptualloss = ContentLoss()
        self.mse = torch.nn.L1Loss()
        self.criterion_d = torch.nn.BCELoss()
        self.sobelloss = ReconstructionLoss()
        self.bands = bands

        self.reset_metrics()
        self.save_hyperparameters()

        self.visual_idx = [i for i in range(20)]

        # HR_t1 = torch.randn(1, self.bands, 256, 256)
        # LR_t2 = torch.randn(1, self.bands, 256, 256)
        
        # from thop import profile
        # from thop import clever_format
        # macs, params = profile(self.generator, inputs=(HR_t1, LR_t2))
        # macs, params = clever_format([macs, params], "%.3f")
        # print("macs:", macs) 
        # print("params", params) 
        # macs, params = profile(self.dnet, inputs=HR_t1)
        # macs, params = clever_format([macs, params], "%.3f")
        # print("macs:", macs) 
        # print("params", params) 
        
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=2e-4)
        opt_d = torch.optim.Adam(self.dnet.parameters(), lr=2e-3)
        return [opt_g, opt_d], []

    def forward(self, LR_t1, LR_t2, HR_t1, HR_t2=None):
        pred, outs = self.generator(HR_t1, LR_t2)
        out = dict_of(pred, outs)
        return out

    def training_step(self, batch, batch_idx):
        LR_t1, LR_t2, HR_t1, gt = batch["LR_t1"], batch["LR_t2"], batch["HR_t1"], batch["HR_t2"]
        self.generator.zero_grad()
        self.dnet.zero_grad()
        opt = self.optimizers()
        opt_g, opt_d = opt[0], opt[1]

        B = LR_t1.shape[0]
        W = LR_t1.shape[-1]
        device = LR_t1.device
        
        self.real_label = torch.ones([B, 1, W, W]).to(device)
        self.real_label_val = torch.ones([B, 1, W, W]).to(device)
        self.fake_label = torch.zeros([B, 1, W, W]).to(device)

        # D
        out = self.forward(LR_t1, LR_t2, HR_t1)
        fake_img = out["pred"]
        outs = out["outs"]

        real_out = self.dnet(gt)
        fake_out = self.dnet(fake_img.detach())
        loss_d = 0.5 * torch.mean((real_out - self.real_label) ** 2) + 0.5 * torch.mean(
            (fake_out - self.fake_label) ** 2)
        self.manual_backward(loss_d)
        opt_d.step()
        
        # G
        loss_g = 0.01 * self.AdversarialLoss(self.dnet(fake_img), self.real_label) + self.mse(fake_img, gt) + self.sobelloss(fake_img, gt) + self.sobelloss(outs, gt)
        self.manual_backward(loss_g)
        opt_g.step()

        log_dict = {
            "d_loss": loss_d.item(),
            "g_loss": loss_g.item(),
            "mse": F.mse_loss(fake_img.detach(), gt).item()
        }

        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)


    def on_train_epoch_end(self):
        pass
