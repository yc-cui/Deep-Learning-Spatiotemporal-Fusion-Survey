import torch.nn as nn

from .swin_transformer import PatchEmbed, PatchMerging, BasicLayer


class Down(nn.Module):
    def __init__(self, bands, img_size, down_scale=2, in_dim=64, depths=(2, 2, 6, 2)):
        super(Down, self).__init__()
        self.inc = PatchEmbed(img_size=img_size, patch_size=down_scale, in_chans=bands, embed_dim=in_dim,
                              norm_layer=nn.LayerNorm)
        self.down1 = DownBlock(in_channels=in_dim, out_channels=in_dim * 2, resolution=img_size // down_scale,
                               downsample=PatchMerging, cur_depth=depths[0])
        self.down2 = DownBlock(in_channels=in_dim * 2, out_channels=in_dim * 4, resolution=img_size // (down_scale*2),
                               downsample=PatchMerging, cur_depth=depths[1])
        self.down3 = DownBlock(in_channels=in_dim * 4, out_channels=in_dim * 8, resolution=img_size // (down_scale*4),
                               downsample=PatchMerging, cur_depth=depths[2])
        self.down4 = DownBlock(in_channels=in_dim * 8, out_channels=in_dim * 8, resolution=img_size // (down_scale*8),
                               downsample=PatchMerging, cur_depth=depths[3])

    def forward(self, x):   
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x1, x2, x3, x4, x5


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resolution, downsample, cur_depth):
        super(DownBlock, self).__init__()
        self.layer = BasicLayer(dim=in_channels,
                                input_resolution=(resolution, resolution),
                                depth=cur_depth,
                                num_heads=in_channels // 32,
                                window_size=8,
                                mlp_ratio=1,
                                qkv_bias=True, qk_scale=None,
                                drop=0., attn_drop=0.,
                                drop_path=0.,
                                norm_layer=nn.LayerNorm)

        if downsample is not None:
            self.downsample = downsample((resolution, resolution), in_channels, out_channels)
        else:
            self.downsample = None

    def forward(self, x):
        x_o = self.layer(x)

        if self.downsample is not None:
            x_o = self.downsample(x_o)

        return x_o

