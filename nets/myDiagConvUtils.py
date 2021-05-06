import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.diagConv.diagConv import diagConv2d_p, diagConv2d_n


class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes + i * growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        # return F.avg_pool2d(out, 2)
        return out


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class DiagConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        """out_planes就是growth_rate"""
        super(DiagConvBlock, self).__init__()
        self.droprate = dropRate

        assert out_planes % 4 == 0
        inter_planes = out_planes // 4
        # 1. 水平卷积
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=[1, 3], stride=1, padding=[0, 1], bias=False)
        # 2. 垂直卷积
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_planes, inter_planes, kernel_size=[3, 1], stride=1, padding=[1, 0], bias=False)
        # 3. 主对角线卷积
        self.bn3 = nn.BatchNorm2d(in_planes)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = diagConv2d_p(in_planes, inter_planes, kernel_size=3, stride=1, padding=1, bias=False)
        # 4. 反对角线卷积
        self.bn4 = nn.BatchNorm2d(in_planes)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv4 = diagConv2d_n(in_planes, inter_planes, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out1 = self.conv1(self.relu1(self.bn1(x)))
        out2 = self.conv2(self.relu2(self.bn2(x)))
        out3 = self.conv3(self.relu3(self.bn3(x)))
        out4 = self.conv4(self.relu4(self.bn4(x)))

        return torch.cat([x, out1, out2, out3, out4], 1)
