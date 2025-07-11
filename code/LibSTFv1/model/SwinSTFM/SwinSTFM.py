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
from LibSTFv1.model.SwinSTFM.loss import GeneratorLoss
from LibSTFv1.model.SwinSTFM.swinstfm import SwinSTFM
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


class SwinSTFMModel(BaseModel):
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
        self.model = SwinSTFM(bands, 512)
        
        self.cri_pix = GeneratorLoss()

        self.bands = bands

        self.reset_metrics()
        self.save_hyperparameters()


    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs, eta_min=1e-7) # original: ReduceLROnPlateau
        return [opt], [scheduler]

    def forward(self, LR_t1, LR_t2, HR_t1, HR_t2=None):
        pred = self.model(LR_t1, HR_t1, LR_t2)
        out = dict_of(pred)
        return out

    def training_step(self, batch, batch_idx):
        LR_t1, LR_t2, HR_t1, gt = batch["LR_t1"], batch["LR_t2"], batch["HR_t1"], batch["HR_t2"]
        opt = self.optimizers()
        opt.zero_grad()

        out = self.forward(LR_t1, LR_t2, HR_t1) 
        pred = out["pred"]

        l_total = self.cri_pix(pred, gt, is_ds=False)

        self.manual_backward(l_total)
        opt.step()

        # 记录日志
        log_dict = {
            "loss": l_total.item(),
            "mse": F.mse_loss(pred.detach(), gt).item()
        }

        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)


    def on_train_epoch_end(self):
        sche_pf = self.lr_schedulers()
        sche_pf.step()
