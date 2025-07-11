
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
from LibSTFv1.metric.cross_correlation import cross_correlation
from LibSTFv1.model.STFDiff.diffusion import GaussianDiffusion
from LibSTFv1.model.STFDiff.pred_resnet import PredNoiseNet
from LibSTFv1.util.misc import check_and_make, regularize_inputs
from LibSTFv1.loss.l1_loss import l1_loss
from sorcery import dict_of
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
from io import BytesIO
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
from ema_pytorch import EMA

plt.rcParams['mathtext.fontset'] = 'cm'
sns.set(style="ticks", palette="bright")


class STFDiffModel(BaseModel):
    def __init__(self,
                 epochs,
                 bands,
                 rgb_c,
                 dataname,
                 ):
        super().__init__(
                 epochs,
                 bands,
                 rgb_c,
                 dataname,)
        self.automatic_optimization = False

        self.rgb_c = rgb_c
        self.model = GaussianDiffusion(
                    model=PredNoiseNet(dim=64, channels=bands, out_dim=bands, dim_mults=(1, 2, 4)),
                    image_size=512,
                    timesteps=100,
                    sampling_timesteps=50,
                    objective="pred_x0",
                    ddim_sampling_eta=0.0,
                )
        self.ema = EMA(self.model, beta=0.995, update_every=1)
        self.loss = l1_loss
        self.dataname = dataname
        self.bands = bands

        self.reset_metrics()
        self.save_hyperparameters()

        self.visual_idx = [i for i in range(20)]

        # LR_t1 = torch.randn(1, self.bands, 256, 256)
        # LR_t2 = torch.randn(1, self.bands, 256, 256)
        # HR_t1 = torch.randn(1, self.bands, 256, 256)
        # HR_t2 = torch.randn(1, self.bands, 256, 256)
        
        # from thop import profile
        # from thop import clever_format
        # macs, params = profile(self.model, inputs=(LR_t1, LR_t2, HR_t1, HR_t2))
        # macs, params = clever_format([macs, params], "%.3f")
        # print("macs:", macs) 
        # print("params", params) 
        
        
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=2e-4)
        sche_opt = StepLR(opt, step_size=100, gamma=0.8)
        return [opt], [sche_opt]

    def forward(self, LR_t1, LR_t2, HR_t1, HR_t2=None):
        loss = None
        pred = None
        if HR_t2 is not None:
            loss = self.model(LR_t1, LR_t2, HR_t1, HR_t2)
        else:
            pred = self.ema.ema_model.sample(LR_t1, LR_t2, HR_t1)
        out = dict_of(pred, loss)
        return out

    def on_train_epoch_end(self):
        sche_pf = self.lr_schedulers()
        sche_pf.step()
        self.ema.update()
    
    def training_step(self, batch, batch_idx):
        LR_t1, LR_t2, HR_t1, gt = batch["LR_t1"], batch["LR_t2"], batch["HR_t1"], batch["HR_t2"]
        out = self.forward(LR_t1, LR_t2, HR_t1, gt)
        total_loss = out["loss"]
        log_dict = {"total_loss": total_loss}
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(total_loss)
        opt.step()

        log_dict["lr"] = opt.param_groups[0]["lr"]
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)

 