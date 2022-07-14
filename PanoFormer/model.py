"""
## PanoFormer: Panorama Transformer for Indoor 360 Depth Estimation
## Zhijie Shen, Chunyu Lin, Kang Liao, Lang Nie, Zishuo Zheng, Yao Zhao
## https://arxiv.org/abs/2203.09283
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
import time
from torch import einsum
from PanoFormer.PSA import *

from PanoFormer.equisamplingpoint import genSamplingPattern

########### LeFF #############

class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=0),
            act_layer())

        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))

    def forward(self, x, H, W):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = H

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh * 2)
        
        x = F.pad(x, (1, 1, 0, 0), mode='circular')  # width
        x = F.pad(x, (0, 0, 1, 1))

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh * 2)

        x = self.linear2(x)

        return x

#########################################
# Downsample Block
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel, input_resolution=None):
        super(Downsample, self).__init__()
        self.input_resolution = input_resolution
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=0),

        )

    def forward(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H, W = self.input_resolution
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = F.pad(x, (1, 1, 0, 0), mode='circular')  # width
        x = F.pad(x, (0, 0, 1, 1))
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out


# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel, input_resolution=None):
        super(Upsample, self).__init__()
        self.input_resolution = input_resolution
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out


# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None, act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=0),
            act_layer()
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = F.pad(x, (3 // 2, 3 // 2, 0, 0), mode='circular')  # width
        x = F.pad(x, (0, 0, 3 // 2, 3 // 2))
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x


# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=1, kernel_size=3, stride=1, norm_layer=None, act_layer=None,
                 input_resolution=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.transpose(1, 2).view(B, C, H, W)
        #x = F.interpolate(x, scale_factor=2, mode='nearest')#for 1024*512
        #x = F.pad(x, (3 // 2, 3 // 2, 0, 0), mode='circular')  # width
        #x = F.pad(x, (0, 0, 3 // 2, 3 // 2))
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PSTBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=4,
                 mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, ref_point=None, flag=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.ref_point = ref_point #generate_ref_points(self.input_resolution[1], self.input_resolution[0])
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)

        self.dattn = PanoSelfAttention(num_heads, dim, k=9, last_feat_height=self.input_resolution[0], last_feat_width=self.input_resolution[1], scales=1, dropout=0, need_attn=False)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop, flag=flag)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        # W-MSA/SW-MSA
        x = self.dattn(x, x.unsqueeze(0), self.ref_point.repeat(B, 1, 1, 1, 1))  # nW*B, win_size*win_size, C

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

########### Basic layer of Uformer ################
class PanoFormerLayer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., drop=0., drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False, ref_point = None, flag = 0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = nn.ModuleList([
            PSTBlock(dim=dim, input_resolution=input_resolution,
                                  num_heads=num_heads, win_size=win_size,
                                  shift_size=0 if (i % 2 == 0) else win_size // 2,
                                  mlp_ratio=mlp_ratio,
                                  drop=drop,
                                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                  norm_layer=norm_layer, ref_point=ref_point)
            for i in range(depth)])

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


########### Panoformer ################
class PanoFormer(nn.Module):
    def __init__(self, img_size=256, in_chans=3,
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False,dowsample=Downsample, upsample=Upsample, **kwargs):
        super().__init__()

        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size
        self.ref_point256x512 = genSamplingPattern(256, 512, 3, 3).cuda()#torch.load("network6/Equioffset256x512.pth")
        self.ref_point128x256 = genSamplingPattern(128, 256, 3, 3).cuda()#torch.load("network6/Equioffset128x256.pth")
        self.ref_point64x128 = genSamplingPattern(64, 128, 3, 3).cuda()#torch.load("network6/Equioffset64x128.pth")
        self.ref_point32x64 = genSamplingPattern(32, 64, 3, 3).cuda()##torch.load("network6/Equioffset32x64.pth")
        self.ref_point16x32 = genSamplingPattern(16, 32, 3, 3).cuda()#torch.load("network6/Equioffset16x32.pth")

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]

        # build layers

        # Input/Output
        self.input_proj = InputProj(in_channel=in_chans, out_channel=embed_dim, kernel_size=3, stride=1,
                                    act_layer=nn.GELU)#stride = 2 for 1024*512
        self.output_proj = OutputProj(in_channel=2 * embed_dim, out_channel=1, kernel_size=3, stride=1,
                                      input_resolution=(img_size, img_size * 2))

        # Encoder
        self.encoderlayer_0 = PanoFormerLayer(dim=embed_dim,
                                                output_dim=embed_dim,
                                                input_resolution=(img_size, img_size * 2),
                                                depth=depths[0],
                                                num_heads=num_heads[0],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop=drop_rate,
                                                drop_path=enc_dpr[int(sum(depths[:0])):int(sum(depths[:1]))],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                ref_point=self.ref_point256x512, flag=0)
        self.dowsample_0 = dowsample(embed_dim, embed_dim * 2, input_resolution=(img_size, img_size * 2))
        self.encoderlayer_1 = PanoFormerLayer(dim=embed_dim * 2,
                                                output_dim=embed_dim * 2,
                                                input_resolution=(img_size // 2, img_size * 2 // 2),
                                                depth=depths[1],
                                                num_heads=num_heads[1],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop=drop_rate,
                                                drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                ref_point=self.ref_point128x256, flag=0)
        self.dowsample_1 = dowsample(embed_dim * 2, embed_dim * 4, input_resolution=(img_size // 2, img_size * 2 // 2))
        self.encoderlayer_2 = PanoFormerLayer(dim=embed_dim * 4,
                                                output_dim=embed_dim * 4,
                                                input_resolution=(img_size // (2 ** 2), img_size * 2 // (2 ** 2)),
                                                depth=depths[2],
                                                num_heads=num_heads[2],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop=drop_rate,
                                                drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                ref_point=self.ref_point64x128, flag=0)
        self.dowsample_2 = dowsample(embed_dim * 4, embed_dim * 8,
                                     input_resolution=(img_size // (2 ** 2), img_size * 2 // (2 ** 2)))
        self.encoderlayer_3 = PanoFormerLayer(dim=embed_dim * 8,
                                                output_dim=embed_dim * 8,
                                                input_resolution=(img_size // (2 ** 3), img_size * 2 // (2 ** 3)),
                                                depth=depths[3],
                                                num_heads=num_heads[3],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop=drop_rate,
                                                drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                ref_point=self.ref_point32x64, flag=0)
        self.dowsample_3 = dowsample(embed_dim * 8, embed_dim * 16,
                                     input_resolution=(img_size // (2 ** 3), img_size * 2 // (2 ** 3)))

        # Bottleneck
        self.conv = PanoFormerLayer(dim=embed_dim * 16,
                                      output_dim=embed_dim * 16,
                                      input_resolution=(img_size // (2 ** 4), img_size * 2 // (2 ** 4)),
                                      depth=depths[4],
                                      num_heads=num_heads[4],
                                      win_size=win_size,
                                      mlp_ratio=self.mlp_ratio,
                                      drop=drop_rate,
                                      drop_path=conv_dpr,
                                      norm_layer=norm_layer,
                                      use_checkpoint=use_checkpoint,
                                      ref_point=self.ref_point16x32, flag=0)

        # Decoder
        self.upsample_0 = upsample(embed_dim * 16, embed_dim * 8,
                                   input_resolution=(img_size // (2 ** 4), img_size * 2 // (2 ** 4)))
        self.decoderlayer_0 = PanoFormerLayer(dim=embed_dim * 16,
                                                output_dim=embed_dim * 16,
                                                input_resolution=(img_size // (2 ** 3), img_size * 2 // (2 ** 3)),
                                                depth=depths[5],
                                                num_heads=num_heads[5],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop=drop_rate,
                                                drop_path=dec_dpr[:depths[5]],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                ref_point=self.ref_point32x64, flag=1)
        self.upsample_1 = upsample(embed_dim * 16, embed_dim * 4,
                                   input_resolution=(img_size // (2 ** 3), img_size * 2 // (2 ** 3)))
        self.decoderlayer_1 = PanoFormerLayer(dim=embed_dim * 8,
                                                output_dim=embed_dim * 8,
                                                input_resolution=(img_size // (2 ** 2), img_size * 2 // (2 ** 2)),
                                                depth=depths[6],
                                                num_heads=num_heads[6],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop=drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                ref_point=self.ref_point64x128,flag=1)
        self.upsample_2 = upsample(embed_dim * 8, embed_dim * 2,
                                   input_resolution=(img_size // (2 ** 2), img_size * 2 // (2 ** 2)))
        self.decoderlayer_2 = PanoFormerLayer(dim=embed_dim * 4,
                                                output_dim=embed_dim * 4,
                                                input_resolution=(img_size // 2, img_size * 2 // 2),
                                                depth=depths[7],
                                                num_heads=num_heads[7],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop=drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                ref_point=self.ref_point128x256, flag=1)
        self.upsample_3 = upsample(embed_dim * 4, embed_dim, input_resolution=(img_size // 2, img_size * 2 // 2))
        self.decoderlayer_3 = PanoFormerLayer(dim=embed_dim * 2,
                                                output_dim=embed_dim * 2,
                                                input_resolution=(img_size, img_size * 2),
                                                depth=depths[8],
                                                num_heads=num_heads[8],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop=drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                ref_point=self.ref_point256x512, flag=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def forward(self, x):
        # Input Projection
        #y = self.pre_block(x)
        y = self.input_proj(x)
        y = self.pos_drop(y)

        # Encoder
        conv0 = self.encoderlayer_0(y)
        pool0 = self.dowsample_0(conv0)
        conv1 = self.encoderlayer_1(pool0)
        pool1 = self.dowsample_1(conv1)
        conv2 = self.encoderlayer_2(pool1)
        pool2 = self.dowsample_2(conv2)
        conv3 = self.encoderlayer_3(pool2)
        pool3 = self.dowsample_3(conv3)

        # Bottleneck
        conv4 = self.conv(pool3)

        # Decoder
        up0 = self.upsample_0(conv4)
        deconv0 = torch.cat([up0, conv3], -1)
        deconv0 = self.decoderlayer_0(deconv0)

        up1 = self.upsample_1(deconv0)
        deconv1 = torch.cat([up1, conv2], -1)
        deconv1 = self.decoderlayer_1(deconv1)

        up2 = self.upsample_2(deconv1)
        deconv2 = torch.cat([up2, conv1], -1)
        deconv2 = self.decoderlayer_2(deconv2)

        up3 = self.upsample_3(deconv2)
        deconv3 = torch.cat([up3, conv0], -1)
        deconv3 = self.decoderlayer_3(deconv3)

        # Output Projection
        
        y = self.output_proj(deconv3)
        outputs = {}
        outputs["pred_depth"] = y
        return outputs



