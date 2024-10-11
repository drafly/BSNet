import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2
# from pretrained_pth.hardnet_68 import hardnet
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

from lib.BG import Spade
import numpy as np
import cv2
from lib.DB import diffdetect


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class conv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_relu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_relu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)

        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x0 * self.sigmoid(out)


class DWConv_Mulit(nn.Module):
    def __init__(self, dim=768):
        super(DWConv_Mulit, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv_Mulit(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, g, H, W):
        B, N, C = x.shape
        q = self.q(g).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, q, k


class Block(nn.Module):

    def __init__(self, dim, num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, g, H, W):
        msa, q, k = self.attn(self.norm1(x), self.norm2(g), H, W)

        x = x + g + msa

        x = x + self.mlp(self.norm2(x), H, W)

        return x, q, k


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class agg(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.up_1 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up_2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_2_1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.c1 = conv2d(in_c[0], out_c, kernel_size=1, padding=0)
        self.c2 = conv2d(in_c[1], out_c, kernel_size=1, padding=0)
        self.c3 = conv2d(in_c[2], out_c, kernel_size=1, padding=0)

        self.c12_11 = conv2d(out_c * 2, out_c)
        self.c12_12 = conv2d(out_c, out_c, kernel_size=1, padding=0)

        self.c12_21 = conv2d(out_c * 2, out_c)
        self.c12_22 = conv2d(out_c, out_c, kernel_size=1, padding=0)

        self.c22 = conv2d(2 * out_c, out_c)
        self.c23 = conv2d(out_c, out_c)

    def forward(self, x1, x2, x3):
        x1 = self.up_1(x1)  # 对最底层特征上采样四倍
        x1 = self.c1(x1)  # 卷积成64通道

        x2 = self.c2(x2)  # 卷积成64通道
        x2 = self.up_2(x2)  # 将到第二层特征上采样二倍
        x12 = torch.cat([x1, x2], 1)  # 通道连接
        x12 = self.up_2_1(x12)  # 上采样二倍

        x12_1 = self.c12_11(x12)  # 对后两层合并的特征卷积为64通道
        # x12_1 = self.c12_12(x12_1)  # 1*1卷积  # 1*1卷积

        x3 = self.up_3(x3)  # 将到第三层特征上采样二倍

        x = torch.cat([x12_1, x3], 1)
        x = self.c23(self.c22(x))
        return x
        # 底层两特征调整相同维度通道聚合，然后再次转为64通道的两个特征，一个和上一层特征相乘，然后和剩下的相加，结果再和第一层特征相加


class BSNet(nn.Module):
    def __init__(self, channel=64):
        super(BSNet, self).__init__()

        self.backbone = pvt_v2_b2()
        path = 'pretrained/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.edge_d = diffdetect()

        self.rfb2_1 = RFB_modified(128, channel)
        self.rfb3_1 = RFB_modified(320, channel)
        self.rfb4_1 = RFB_modified(512, channel)

        self.con4_3 = conv2d(64, 64)
        self.con3_2 = conv2d(64, 64)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.spade1 = Spade(64, 64)
        self.spade2 = Spade(64, 64)
        self.spade3 = Spade(64, 64)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.MFA = agg([64, 64, 64], 64)

        self.out1 = nn.Conv2d(64, 1, 1)
        self.out2 = nn.Conv2d(64, 1, 1)
        self.out3 = nn.Conv2d(64, 1, 1)
        self.out4 = nn.Conv2d(64, 1, 1)
        self.out5 = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        image_shape = x.size()[2:]
        pvt = self.backbone(x)
        x_1 = pvt[0]
        x_2 = pvt[1]
        x_3 = pvt[2]
        x_4 = pvt[3]

        x_2 = self.rfb2_1(x_2)
        x_3 = self.rfb3_1(x_3)
        x_4 = self.rfb4_1(x_4)

        edge_map = self.edge_d(x_1, x_2, x_3, x_4)
        edge_out = F.interpolate(edge_map, size=image_shape, mode='bilinear')

        x4 = F.interpolate(self.con4_3(x_4), scale_factor=2, mode='bilinear')
        x_3 = x_3 + x4

        x3 = F.interpolate(self.con3_2(x_3), scale_factor=2, mode='bilinear')
        x_2 = x_2 + x3

        # edge_map = self.edge_d(x_1, x_2, x_3, x_4)
        # edge_out = F.interpolate(edge_map, size=image_shape, mode='bilinear')

        edge_map_2 = edge_map
        edge_map_3 = F.interpolate(edge_map, size=(x_3.shape[2], x_3.shape[3]), mode='bilinear', align_corners=True)
        edge_map_4 = F.interpolate(edge_map, size=(x_4.shape[2], x_4.shape[3]), mode='bilinear', align_corners=True)


        x_2 = self.spade1(x_2, edge_map_2)
        x_3 = self.spade2(x_3, edge_map_3)
        x_4 = self.spade3(x_4, edge_map_4)

        x_out = self.MFA(x_4, x_3, x_2)

        x_out = F.interpolate(self.out5(x_out), scale_factor=4, mode='bilinear')
        prediction1_4 = F.interpolate(self.out1(x_1), scale_factor=4, mode='bilinear')
        prediction2_8 = F.interpolate(self.out2(x_2), scale_factor=8, mode='bilinear')
        prediction3_16 = F.interpolate(self.out3(x_3), scale_factor=16, mode='bilinear')
        prediction4_32 = F.interpolate(self.out4(x_4), scale_factor=32, mode='bilinear')

        return x_out, prediction1_4, prediction2_8, prediction3_16, prediction4_32, edge_out


if __name__ == '__main__':
    model = BSNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction1, prediction2 = model(input_tensor)
    print(prediction1.size(), prediction2.size())
