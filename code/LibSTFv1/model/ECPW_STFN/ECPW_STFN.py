
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
from LibSTFv1.model.ECPW_STFN.model_ParallelWavelet_withrefinement import CompoundLoss, FusionNet_2, Pretrained
from LibSTFv1.model.ECPW_STFN.model_pretrain import AutoEncoder, VisionLoss
from LibSTFv1.model.STFDiff.diffusion import GaussianDiffusion
from LibSTFv1.model.STFDiff.pred_resnet import PredNoiseNet
from LibSTFv1.util.misc import check_and_make, regularize_inputs
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


class ECPW_STFNModel(BaseModel):
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
        self.model = FusionNet_2(bands)
        self.pretrained = AutoEncoder(bands)
        self.loss = CompoundLoss(self.pretrained)  # the pretrained loss may be different from the paper
        self.loss_pre = VisionLoss()
        self.dataname = dataname
        self.bands = bands
        self.pre_epoch = 200 #test

        self.reset_metrics()
        self.save_hyperparameters()

        self.visual_idx = [i for i in range(20)]
        
        
        # LR_t1 = torch.randn(1, self.bands, 256, 256)
        # LR_t2 = torch.randn(1, self.bands, 256, 256)
        # HR_t1 = torch.randn(1, self.bands, 256, 256)
        # HR_t2 = torch.randn(1, self.bands, 256, 256)
        
        # from thop import profile
        # from thop import clever_format
        # macs, params = profile(self.model, inputs=(LR_t1, LR_t2, HR_t1))
        # macs, params = clever_format([macs, params], "%.3f")
        # print("macs:", macs) 
        # print("params", params) 
        # macs, params = profile(self.pretrained, inputs=HR_t1)
        # macs, params = clever_format([macs, params], "%.3f")
        # print("macs:", macs) 
        # print("params", params) 

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-6)
        opt_pre = torch.optim.Adam(self.pretrained.parameters(), lr=1e-3, weight_decay=1e-6)
        sche_opt = StepLR(opt, step_size=100, gamma=0.8)
        sche_opt_pre = StepLR(opt, step_size=100, gamma=0.8)
        return [opt, opt_pre], [sche_opt, sche_opt_pre]

    def on_train_epoch_end(self):
        sche_pf = self.lr_schedulers()
        sche_pf[0].step()
        sche_pf[1].step()

    def forward(self, LR_t1, LR_t2, HR_t1, HR_t2=None):
        if self.current_epoch < self.pre_epoch:
            pred = self.pretrained(HR_t1)
        else:
            pred = self.model(LR_t1, LR_t2, HR_t1)
        out = dict_of(pred)
        return out


    def training_step(self, batch, batch_idx):
        LR_t1, LR_t2, HR_t1, gt = batch["LR_t1"], batch["LR_t2"], batch["HR_t1"], batch["HR_t2"]
        out = self.forward(LR_t1, LR_t2, HR_t1)
        pred = out["pred"]
        opts = self.optimizers()
        opt, opt_pre = opts[0], opts[1]
        log_dict = {"pre_loss": 0, "loss": 0}
        if self.current_epoch < self.pre_epoch:
            total_loss = self.loss_pre(pred, gt)
            opt_pre.zero_grad()
            self.manual_backward(total_loss)
            opt_pre.step()
            log_dict["pre_loss"] = total_loss.item()
        else:
            total_loss = self.loss(pred, gt)
            opt.zero_grad()
            self.manual_backward(total_loss)
            opt.step()
            
            # ! bug: truncated gradients
            # for name, param in self.model.Landsat_encoder.named_parameters():
            #     if param.grad is not None:
            #         print(f"Layer: {name}, Gradient: {param.grad}")
        
            # for param in self.model.RFEncoder.parameters():
                # print(param)

            log_dict["loss"] = total_loss.item()

        log_dict["lr"] = opt.param_groups[0]["lr"]
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        #self.log("train/loss", total_loss, prog_bar=True, on_epoch=True, sync_dist=True)
 