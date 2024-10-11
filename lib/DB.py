import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.EAM import get_sobel, run_sobel, EAM

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


class DB(nn.Module):
    def __init__(self, in_channels, guidance_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(DB, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.conv1 = conv2d(guidance_channels, in_channels, kernel_size=3, padding=1)
        self.theta = theta
        self.guidance_channels = guidance_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        # initialize for skipped features
        x_initial = torch.randn(in_channels, 1, kernel_size, kernel_size)
        x_initial[:, :, int(kernel_size / 2), int(kernel_size / 2)] = -1
        self.x_kernel_diff = nn.Parameter(x_initial)
        self.x_kernel_diff[:, :, int(kernel_size / 2), int(kernel_size / 2)].detach()

        self.sobel_x1, self.sobel_y1 = get_sobel(in_channels, 1)
        # initialize for guidance features
        guidance_initial = torch.randn(in_channels, 1, kernel_size, kernel_size)
        guidance_initial[:, :, int(kernel_size / 2), int(kernel_size / 2)] = -1
        self.guidance_kernel_diff = nn.Parameter(guidance_initial)
        self.guidance_kernel_diff[:, :, int(kernel_size / 2), int(kernel_size / 2)].detach()

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x, guidance):
        guidance = F.interpolate(guidance, scale_factor=2, mode='bilinear')
        in_channels = self.in_channels

        guidance = self.conv1(guidance)

        s1 = run_sobel(self.sobel_x1, self.sobel_y1, guidance)
        s1 = 1 / (s1 ** 2 + 1)
        x_diff = F.conv2d(input=x, weight=self.x_kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                          padding=1, groups=in_channels)

        out = self.conv(x_diff * s1)
        boundary = self.relu(self.bn(out))
        # boundary = self.bn(out)
        out = boundary + x
        return out


class diffdetect(nn.Module):
    def __init__(self):
        super(diffdetect, self).__init__()

        self.db1 = DB(64, 64)
        self.db2 = DB(64, 64)
        self.db3 = DB(64, 64)

        self.edge_conv_cat = BasicConv2d(64 * 3, 64, kernel_size=3, padding=1)
        self.edge_linear = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x1, x2, x3, x4):
        d1 = self.db1(x1, x2)
        d2 = self.db2(x2, x3)
        d3 = self.db3(x3, x4)

        d3 = F.interpolate(d3, scale_factor=4, mode='bilinear')
        d2 = F.interpolate(d2, scale_factor=2, mode='bilinear')

        edge_cat = self.edge_conv_cat(torch.cat((d1, d2, d3), dim=1))
        edge_map = self.edge_linear(edge_cat)

        return edge_map
