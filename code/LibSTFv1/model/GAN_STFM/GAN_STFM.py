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

from LibSTFv1.metric import cross_correlation

from LibSTFv1.model.GAN_STFM.network import AutoEncoder, MSDiscriminator, ReconstructionLoss, SFFusion
from LibSTFv1.model.GAN_STFM.utils import load_pretrained
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
from torchgan.losses import LeastSquaresDiscriminatorLoss, LeastSquaresGeneratorLoss


class GANSTFMModel(BaseModel):
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
        self.bands = bands  # 移到前面，先保存通道数
        
        # 使用实际的通道数初始化网络
        self.generator = SFFusion(in_channels=bands)
        self.discriminator = MSDiscriminator(num_bands=bands)
        self.pretrained = AutoEncoder(in_channels=bands)
        
        # 加载预训练权重时处理通道数不匹配的问题
        try:
            load_pretrained(self.pretrained, 'LibSTFv1/model/GAN_STFM/autoencoder.pth')
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}. Using random initialization.")

        self.criterion = ReconstructionLoss(self.pretrained)
        self.g_loss = LeastSquaresGeneratorLoss()
        self.d_loss = LeastSquaresDiscriminatorLoss()

        self.reset_metrics()
        self.save_hyperparameters()

        self.visual_idx = [i for i in range(20)]

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=1e-3)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3)
        return [opt_g, opt_d], []

    def forward(self, LR_t1, LR_t2, HR_t1, HR_t2=None):
        pred = self.generator(LR_t1, LR_t2, HR_t1)
        out = dict_of(pred)
        return out

    def training_step(self, batch, batch_idx):
        LR_t1, LR_t2, HR_t1, gt = batch["LR_t1"], batch["LR_t2"], batch["HR_t1"], batch["HR_t2"]
        self.generator.zero_grad()
        self.discriminator.zero_grad()
        opt = self.optimizers()
        opt_g, opt_d = opt[0], opt[1]

        # D
        out = self.forward(LR_t1, LR_t2, HR_t1)
        pred = out["pred"]
        d_loss = (self.d_loss(self.discriminator(torch.cat((gt, LR_t1), 1)),
                  self.discriminator(torch.cat((pred.detach(), LR_t1), 1))))
        self.manual_backward(d_loss)
        opt_d.step()
        
        # G
        g_loss = (self.criterion(pred, gt) + 1e-3 *
                      self.g_loss(self.discriminator(torch.cat((pred, LR_t1), 1))))
            
        self.manual_backward(g_loss)
        opt_g.step()

        log_dict = {
            "d_loss": d_loss.item(),
            "g_total_loss": g_loss.item(),
            "mse": F.mse_loss(pred.detach(), gt).item()
        }

        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)


    def on_train_epoch_end(self):
        pass
