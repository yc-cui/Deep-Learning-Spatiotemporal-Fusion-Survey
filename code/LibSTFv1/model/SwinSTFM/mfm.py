import torch
import torch.nn as nn

from .swin_transformer import BasicLayer, BasicLayer3


class FineUp(nn.Module):
    def __init__(self, in_dim=64, img_size=512, down_scale=2, depths=(2, 2, 6, 2), output_bands=6):
        super(FineUp, self).__init__()
        self.down_scale = down_scale
        self.img_size = img_size
        self.up1 = FineUpBlock(in_dim * 8 * 2, in_dim * 4, img_size // (down_scale*8), depths[3], 1)
        self.up2 = FineUpBlock(in_dim * 4 * 2, in_dim * 2, img_size // (down_scale*4), depths[2], 2)
        self.up3 = FineUpBlock(in_dim * 2 * 2, in_dim, img_size // (down_scale*2), depths[1], 4)
        self.up4 = FineUpBlock(in_dim * 2, in_dim, img_size // down_scale, depths[0], 8)

        self.outc = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(in_dim, output_bands, 3, 1, 1),
            nn.Tanh()
        )

        self.layer = BasicLayer3(dim=in_dim * 8, input_resolution=(img_size // (down_scale*8*2), img_size // (down_scale*8*2)),
                                 depth=1, num_heads=in_dim * 8 // 32, window_size=8, mlp_ratio=1,
                                 qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.,
                                 norm_layer=nn.LayerNorm)

    def forward(self, diff0_features, diff1_features, fine_features, coarse_features):

        x0 = fine_features[4]
        x0 = self.layer(diff1_features[4], diff0_features[4], x0, coarse_features[4]) + x0

        x1 = self.up1(fine_features[3], diff0_features[3], coarse_features[3], diff1_features[3], x0)
        x2 = self.up2(fine_features[2], diff0_features[2], coarse_features[2], diff1_features[2], x1)
        x3 = self.up3(fine_features[1], diff0_features[1], coarse_features[1], diff1_features[1], x2)
        x4 = self.up4(fine_features[0], diff0_features[0], coarse_features[0], diff1_features[0], x3)

        B, L, C = x4.shape

        x4 = x4.transpose(1, 2).view(B, C, self.img_size // self.down_scale, self.img_size // self.down_scale)
        output_fine = self.outc(x4)

        return output_fine


class FineUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resolution, cur_depth, gaussian_kernel_size):
        super(FineUpBlock, self).__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.up = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2 * 4, 3, 1, 1),
            nn.PixelShuffle(2)
        )

        self.layer = BasicLayer3(dim=in_channels // 2, input_resolution=(resolution, resolution),
                                 depth=2, num_heads=in_channels // 2 // 32, window_size=8, mlp_ratio=1,
                                 qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.,
                                 norm_layer=nn.LayerNorm)

        self.layer2 = BasicLayer(dim=out_channels, input_resolution=(resolution, resolution),
                                 depth=cur_depth, num_heads=out_channels // 32, window_size=8, mlp_ratio=1,
                                 qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.,
                                 norm_layer=nn.LayerNorm)

        self.proj1 = nn.Linear(in_channels // 2 * 3, out_channels)

    def forward(self, x_fine0, x_diff0, x_coarse0, x_diff1, x_fine1):
        """
        :param x_fine0: 下采样的精细特征
        :param x_diff0: 下采样的差异特征
        :param x_coarse0: 下采样的粗糙特征
        :param x_fine1: 上一阶段上采样的精细特征
        :return:
        """

        B, L, C = x_fine1.shape

        x_f1 = x_fine1.transpose(1, 2).view(B, C, self.resolution // 2, self.resolution // 2)
        x_f1 = self.up(x_f1).flatten(2).transpose(1, 2)

        x_f0 = self.layer(x_diff1, x_diff0, x_fine0, x_coarse0) + x_fine0
        x = torch.cat([x_coarse0, x_f0, x_f1], dim=2)
        x = self.proj1(x)
        x = self.layer2(x)

        return x