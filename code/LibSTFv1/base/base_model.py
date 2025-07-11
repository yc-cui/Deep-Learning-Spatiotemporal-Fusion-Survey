from matplotlib.patheffects import Stroke, Normal
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import os
import torchmetrics.functional.image as MF
from torchmetrics.functional.regression import r2_score
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import pandas as pd
import time
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from LibSTFv1.metric.cross_correlation import cross_correlation
from LibSTFv1.util.misc import check_and_make, regularize_inputs
from LibSTFv1.loss.l1_loss import l1_loss
from sorcery import dict_of
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
from io import BytesIO
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
plt.rcParams['mathtext.fontset'] = 'cm'
sns.set(style="ticks", palette="bright")


class BaseModel(pl.LightningModule):
    def __init__(self,
                 epochs,
                 bands,
                 rgb_c,
                 dataname,
                 ):
        super().__init__()
        self.automatic_optimization = False

        self.rgb_c = rgb_c
        self.model = None
        self.loss = l1_loss
        self.dataname = dataname
        self.bands = bands

        self.reset_metrics()
        self.save_hyperparameters()

        self.visual_idx = [i for i in range(5)]
        
        self.metric_ranges = {
            'MAE': (0, 1),
            'SSIM': (-1, 1),
            'RMSE': (0, 2),
            'ERGAS': (0, 1000),
            'SAM': (0, np.pi),
            'RASE': (0, 1000),
            'PSNR': (0, 100),
            'UQI': (-1, 1),
            'SCC': (-1, 1),
            'CC': (-1, 1),
            'R2': (-10, 1),
            'Time': (0, 100)
        }

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        sche_opt = StepLR(opt, step_size=100, gamma=0.8)
        return [opt], [sche_opt]

    def forward(self, LR_t1, LR_t2, HR_t1, HR_t2):
        pred = self.model(LR_t1, LR_t2, HR_t1)
        out = dict_of(pred)
        return out

    def training_step(self, batch, batch_idx):
        LR_t1, LR_t2, HR_t1, gt = batch["LR_t1"], batch["LR_t2"], batch["HR_t1"], batch["HR_t2"]
        out = self.forward(LR_t1, LR_t2, HR_t1, gt)
        pred = out["pred"]
        opt = self.optimizers()

        opt.zero_grad()
        total_loss, log_dict = self.loss(gt, pred)
        self.manual_backward(total_loss)
        opt.step()

        log_dict["lr"] = opt.param_groups[0]["lr"]
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)

    def on_train_epoch_end(self):
        sche_pf = self.lr_schedulers()
        sche_pf.step()

    def validation_step(self, batch, batch_idx):
        LR_t1, LR_t2, HR_t1, gt = batch["LR_t1"], batch["LR_t2"], batch["HR_t1"], batch["HR_t2"]
        out = self.forward(LR_t1, LR_t2, HR_t1, None)
        pred = out["pred"]
        # pred = self.trainer.datamodule.dataset_val.inv_transform_ms(pred.cpu()).to(self.device)
        # print(self.trainer.datamodule.dataset_val)

        pred, gt = regularize_inputs(pred, gt)
        
        self.save_full_ref(pred, gt)
        # self.save_full_ref(pred, gt)
        if batch_idx in self.visual_idx:
            channel_indices = torch.tensor(self.rgb_c, device=self.device)
            LR_t1_rgb = torch.index_select(LR_t1, 1, channel_indices)
            LR_t2_rgb = torch.index_select(LR_t2, 1, channel_indices)
            HR_t1_rgb = torch.index_select(HR_t1, 1, channel_indices)
            gt_rgb = torch.index_select(gt, 1, channel_indices)
            pred_rgb = torch.index_select(pred, 1, channel_indices)
            err_rgb = torch.abs(pred - gt).mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            err_rgb /= torch.max(err_rgb)
            rgb_imgs = torch.cat([
                LR_t1_rgb,
                LR_t2_rgb,
                HR_t1_rgb,
                gt_rgb,
                pred_rgb,
                err_rgb], dim=0)

            if self.visual is None:
                self.visual = rgb_imgs
            else:
                self.visual = torch.cat([self.visual, rgb_imgs], dim=0)

    def on_validation_epoch_end(self):
        model_name = self.__class__.__name__
        eval_results = {"method": model_name}
        for metric in self.eval_metrics:
            mean = np.mean(self.metrics_all[metric])
            std = np.std(self.metrics_all[metric])
            eval_results[f'{metric}_mean'] = round(mean, 10)
            eval_results[f'{metric}_std'] = round(std, 10)
        filtered_dict = {k: v for k, v in eval_results.items() if isinstance(v, np.float64) and np.isnan(v) == False}
        self.log_dict(filtered_dict)
        filtered_dict["epoch"] = self.current_epoch
        csv_path = os.path.join(self.logger.save_dir, "metrics.csv")
        pd.DataFrame.from_dict(
            [filtered_dict]).to_csv(
            csv_path,
            mode="a",
            index=False,
            header=False if os.path.exists(csv_path) else True)

        grid = make_grid(self.visual, nrow=6, padding=2, normalize=False, scale_each=False, pad_value=0)
        image_grid = grid.permute(1, 2, 0).cpu().numpy()
        check_and_make(f"visual/{model_name}")
        save_path = f"visual/{model_name}/{self.current_epoch}.jpg"
        plt.imsave(save_path, image_grid)
        # self.logger.log_image(key="visual", images=[save_path])
        self.reset_metrics()

    def test_step(self, batch, batch_idx):
        LR_t1, LR_t2, HR_t1, gt = batch["LR_t1"], batch["LR_t2"], batch["HR_t1"], batch["HR_t2"]
        t_start = time.time()
        out = self.forward(LR_t1, LR_t2, HR_t1, None)
        t_end = time.time()
        pred = out["pred"]
        len_patch = self.trainer.datamodule.dataset_test.LRs_arr.shape[1]
        self.save_full_ref(pred, gt, "test")
        self.record_metrics('Time', torch.tensor(t_end - t_start), "test")
        pred, gt = regularize_inputs(pred, gt)
        self.save_RGB(pred, gt, batch_idx, self.rgb_c)
        if (batch_idx+1) % len_patch == 0:
            self.save_metric((batch_idx+1) // len_patch)

        self.save_GT(LR_t1, LR_t2, HR_t1, gt, batch_idx, self.rgb_c)

    def on_test_epoch_start(self):
        self.reset_metrics("test")

    def save_metric(self, id_img=-1):
        model_name = self.__class__.__name__
        eval_results = {"method": model_name}
        print(id_img, self.metrics_all)
        for metric in self.eval_metrics:
            mean = np.mean(self.metrics_all[metric])
            std = np.std(self.metrics_all[metric])
            eval_results[f'{metric}_mean'] = round(mean, 10)
            eval_results[f'{metric}_std'] = round(std, 10)

        filtered_dict = {k: v for k, v in eval_results.items() if isinstance(
            v, np.float64) and np.isnan(v) == False and "std" not in k}
        print(filtered_dict)
        filtered_dict["epoch"] = f"testing-{id_img}"
        # os.makedirs(os.path.join(self.logger.save_dir, self.sensor), exist_ok=True)
        csv_path = os.path.join(self.logger.save_dir, f"test.csv")
        pd.DataFrame.from_dict(
            [filtered_dict]).to_csv(
            csv_path,
            mode="a",
            index=False,
            header=False if os.path.exists(csv_path) else True)
        self.reset_metrics("test")

    def save_GT(self, LR_t1, LR_t2, HR_t1, gt, idx, RGB):
        def _linear_scale(gray, mi=0, ma=255):
            min_value = np.percentile(gray, 2)
            max_value = np.percentile(gray, 98)
            truncated_gray = np.clip(gray, a_min=min_value, a_max=max_value)
            processed_gray = ((truncated_gray - min_value) / (max_value - min_value)) * (ma - mi)
            return processed_gray

        def _normlize_uint8(img):
            # max, min = np.max(img, axis=(0, 1)), np.min(img, axis=(0, 1))
            # img = np.float32(img - min) / (max - min)
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
            return img

        def _get_RGB(img, RGB):
            pred_rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            pred_rgb[:, :, 0] = _linear_scale(img[:, :, RGB[0]]).astype(np.uint8)
            pred_rgb[:, :, 1] = _linear_scale(img[:, :, RGB[1]]).astype(np.uint8)
            pred_rgb[:, :, 2] = _linear_scale(img[:, :, RGB[2]]).astype(np.uint8)
            return pred_rgb

        LR_t1 = LR_t1.squeeze().permute(1, 2, 0).clone().cpu().numpy()
        LR_t2 = LR_t2.squeeze().permute(1, 2, 0).clone().cpu().numpy()
        HR_t1 = HR_t1.squeeze().permute(1, 2, 0).clone().cpu().numpy()
        gt = gt.squeeze().permute(1, 2, 0).clone().cpu().numpy()

        LR_t1_dir = os.path.join(f"viz_test_RGB-GT/{self.dataname}", "LR_t1")
        LR_t2_dir = os.path.join(f"viz_test_RGB-GT/{self.dataname}","LR_t2")
        HR_t1_dir = os.path.join(f"viz_test_RGB-GT/{self.dataname}","HR_t1")
        gt_dir = os.path.join(f"viz_test_RGB-GT/{self.dataname}","gt")
        os.makedirs(LR_t1_dir, exist_ok=True)
        os.makedirs(LR_t2_dir, exist_ok=True)
        os.makedirs(HR_t1_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        plt.imsave(os.path.join(LR_t1_dir, str(idx) + "_" + self.__class__.__name__ + ".png"), _get_RGB(LR_t1, RGB))
        plt.imsave(os.path.join(LR_t2_dir, str(idx) + "_" + self.__class__.__name__ + ".png"), _get_RGB(LR_t2, RGB))
        plt.imsave(os.path.join(HR_t1_dir, str(idx) + "_" + self.__class__.__name__ + ".png"), _get_RGB(HR_t1, RGB))
        plt.imsave(os.path.join(gt_dir, str(idx) + "_" + self.__class__.__name__ + ".png"), _get_RGB(gt, RGB))
        err_0 = _normlize_uint8(np.mean(np.abs(gt - gt), axis=2))
        err_LR_t2 = _normlize_uint8(np.mean(np.abs(gt - LR_t2), axis=2))
        err_HR_t1 = _normlize_uint8(np.mean(np.abs(gt - HR_t1), axis=2))
        err_0_dir = os.path.join(f"viz_test_RGB-GT/{self.dataname}", "err0")
        err_LR_t2_dir = os.path.join(f"viz_test_RGB-GT/{self.dataname}", "err_LR_t2")
        err_HR_t1_dir = os.path.join(f"viz_test_RGB-GT/{self.dataname}", "err_HR_t1")
        os.makedirs(err_0_dir, exist_ok=True)
        os.makedirs(err_LR_t2_dir, exist_ok=True)
        os.makedirs(err_HR_t1_dir, exist_ok=True)
        plt.imsave(os.path.join(err_0_dir,"err0.png"), err_0, cmap="turbo")
        plt.imsave(os.path.join(err_LR_t2_dir, str(idx) + "_" + self.__class__.__name__ + ".png"), err_LR_t2, cmap="turbo")
        plt.imsave(os.path.join(err_HR_t1_dir, str(idx) + "_" + self.__class__.__name__ + ".png"), err_HR_t1, cmap="turbo")



    def save_RGB(self, pred, gt, idx, RGB):
        # return

        def _linear_scale(gray, mi=0, ma=255):
            min_value = np.percentile(gray, 2)
            max_value = np.percentile(gray, 98)
            truncated_gray = np.clip(gray, a_min=min_value, a_max=max_value)
            processed_gray = ((truncated_gray - min_value) / (max_value - min_value)) * (ma - mi)
            return processed_gray

        def _normlize_uint8(img):
            # max, min = np.max(img, axis=(0, 1)), np.min(img, axis=(0, 1))
            # img = np.float32(img - min) / (max - min)
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
            return img

        pred = pred.squeeze().permute(1, 2, 0).clone().cpu().numpy()
        gt = gt.squeeze().permute(1, 2, 0).clone().cpu().numpy()
        pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        pred_rgb[:, :, 0] = _linear_scale(pred[:, :, RGB[0]]).astype(np.uint8)
        pred_rgb[:, :, 1] = _linear_scale(pred[:, :, RGB[1]]).astype(np.uint8)
        pred_rgb[:, :, 2] = _linear_scale(pred[:, :, RGB[2]]).astype(np.uint8)
        err = _normlize_uint8(np.mean(np.abs(pred - gt), axis=2))
        pred_dir = os.path.join(f"viz_test_RGB/{self.dataname}", "pred")
        err_dir = os.path.join(f"viz_test_RGB/{self.dataname}", "err")
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(err_dir, exist_ok=True)

        plt.imsave(os.path.join(pred_dir, str(idx) + "_" + self.__class__.__name__ + ".png"), pred_rgb)
        plt.imsave(os.path.join(err_dir, str(idx) + "_" + self.__class__.__name__ + ".png"), err, cmap="turbo")

    def save_full_ref(self, pred, gt, split="val"):
        data_range = (0., 1.)
        self.record_metrics('MAE', F.l1_loss(pred, gt), split)
        self.record_metrics('SSIM', MF.structural_similarity_index_measure(pred, gt, data_range=data_range), split)
        self.record_metrics('RMSE', MF.root_mean_squared_error_using_sliding_window(pred, gt), split)
        self.record_metrics('ERGAS', MF.error_relative_global_dimensionless_synthesis(pred, gt) / 16., split)
        self.record_metrics('SAM', MF.spectral_angle_mapper(pred, gt), split)
        self.record_metrics('RASE', MF.relative_average_spectral_error(pred, gt), split)
        self.record_metrics('PSNR', MF.peak_signal_noise_ratio(pred, gt, data_range=data_range), split)
        self.record_metrics('UQI', MF.universal_image_quality_index(pred, gt), split)
        self.record_metrics('SCC', MF.spatial_correlation_coefficient(pred, gt), split)
        self.record_metrics('CC', cross_correlation(pred, gt), split)
        r2 = r2_score(pred.view(-1), gt.view(-1))
        self.record_metrics('R2', r2, split)



    def reset_metrics(self, split="val"):
        self.eval_metrics = ['MAE', 'SCC', 'SAM', 'RMSE', 'ERGAS', 'PSNR', 'SSIM', 'RASE',
                             'UQI', "CC", "R2", "Time"]  # 添加了 R2 指标
        self.eval_metrics = [f"{split}/" + i for i in self.eval_metrics]
        tmp_results = {}
        for metric in self.eval_metrics:
            tmp_results.setdefault(metric, [])

        self.metrics_all = tmp_results
        self.visual = None

    def is_outlier_iqr(self, values, new_value, k=1.5):
        """使用IQR方法检测异常值"""
        if len(values) < 4:  # 需要足够的数据点来计算IQR
            return False
        
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        if iqr == 0:  # 如果IQR为0，使用标准差方法
            return self.is_outlier_std(values, new_value)
        
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        
        return new_value < lower_bound or new_value > upper_bound

    def is_outlier_std(self, values, new_value, n_std=3):
        """使用标准差方法检测异常值"""
        if len(values) < 2:
            return False
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return False
        
        return abs(new_value - mean) > n_std * std

    def record_metrics(self, k, v, split="val"):
        """记录指标值，包含异常值检测"""
        try:
            # 首先检查值是否为有限值
            if not torch.isfinite(v):
                print(f"Warning: {k} has non-finite value: {v.item()}, skipping...")
                return
            
            v_item = v.item()
            metric_key = f'{split}/' + k
            
            # 检查是否在预定义的合理范围内
            if k in self.metric_ranges:
                min_val, max_val = self.metric_ranges[k]
                if v_item < min_val or v_item > max_val:
                    print(f"Warning: {k} value {v_item} is outside expected range [{min_val}, {max_val}], skipping...")
                    return
            
            # 如果已有历史数据，进行异常值检测
            if len(self.metrics_all[metric_key]) >= 10:  # 至少有10个数据点才进行异常检测
                # 使用IQR方法检测
                if self.is_outlier_iqr(self.metrics_all[metric_key], v_item, k=2.0):
                    print(f"Warning: {k} value {v_item} detected as outlier (IQR method), skipping...")
                    return
                
                # 对于某些重要指标，额外使用标准差方法
                if k in ['PSNR', 'SSIM', 'MAE', 'RMSE']:
                    if self.is_outlier_std(self.metrics_all[metric_key], v_item, n_std=4):
                        print(f"Warning: {k} value {v_item} detected as outlier (STD method), skipping...")
                        return
            
            # 通过所有检查，记录值
            self.metrics_all[metric_key].append(v_item)
            
        except Exception as e:
            print(f"Error recording metric {k}: {str(e)}")
            # 即使出错也不影响程序运行