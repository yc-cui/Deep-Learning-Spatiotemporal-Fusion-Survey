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
from LibSTFv1.loss.MLFF_GAN import MLFF_GAN_L1, GANLoss
from LibSTFv1.metric import cross_correlation
from LibSTFv1.model.MLFF_GAN.network import CombinFeatureGenerator, NLayerDiscriminator
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


class MLFFGANModel(BaseModel):
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
        self.generator = CombinFeatureGenerator(bands=bands, ifAdaIN=True, ifAttention=True,ifTwoInput = False)
        self.nlayerdiscriminator = NLayerDiscriminator(input_nc = bands * 2, getIntermFeat = True)
        self.pd_loss = GANLoss()

        self.bands = bands

        self.reset_metrics()
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=2e-4)
        opt_d = torch.optim.Adam(self.nlayerdiscriminator.parameters(), lr=2e-4)
        def lambda_rule(epoch):
            lr_l = 1.0
            return lr_l
        g_scheduler = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda=lambda_rule)
        d_scheduler = torch.optim.lr_scheduler.LambdaLR(opt_d, lr_lambda=lambda_rule)
        return [opt_g, opt_d], [g_scheduler, d_scheduler]

    def forward(self, LR_t1, LR_t2, HR_t1, HR_t2=None):
        pred = self.generator(LR_t1, LR_t2, HR_t1)
        out = dict_of(pred)
        return out

    def training_step(self, batch, batch_idx):
        LR_t1, LR_t2, HR_t1, gt = batch["LR_t1"], batch["LR_t2"], batch["HR_t1"], batch["HR_t2"]
        self.generator.zero_grad()
        self.nlayerdiscriminator.zero_grad()
        opt = self.optimizers()
        opt_g, opt_d = opt[0], opt[1]

        # D
        out = self.forward(LR_t1, LR_t2, HR_t1)
        pred = out["pred"]
        pred_fake = self.nlayerdiscriminator(torch.cat((pred.detach(), LR_t2), dim=1))
        pred_real1 = self.nlayerdiscriminator(torch.cat((gt, LR_t2), dim=1))
        pd_loss = (self.pd_loss(pred_fake,False) +
                   self.pd_loss(pred_real1,True))  * 0.5
        
        self.manual_backward(pd_loss)
        opt_d.step()
        
        # G
        out = self.forward(LR_t1, LR_t2, HR_t1)
        pred = out["pred"]
        pred_fake = self.nlayerdiscriminator(torch.cat((pred, LR_t2), dim=1))
        loss_G_GAN = self.pd_loss(pred_fake, True) * 1e-2
        loss_G_l1 = MLFF_GAN_L1(pred, gt, 1, 1, 1)
        g_loss = loss_G_l1 + loss_G_GAN
        self.manual_backward(g_loss)
        opt_g.step()

        log_dict = {
            "d_loss": pd_loss.item(),
            "g_loss": loss_G_GAN.item(),
            "g_total_loss": g_loss.item(),
            "mse": F.mse_loss(pred.detach(), gt).item()
        }

        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)


    def on_train_epoch_end(self):
        sche_pf = self.lr_schedulers()
        sche_pf[0].step()
        sche_pf[1].step()
