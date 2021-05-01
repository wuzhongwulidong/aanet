import torch
import torch.nn as nn


class diagConv2d_p(nn.Module):
    """主对角线卷积"""
    def __init__(self, inChannels, outChannels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(diagConv2d_p, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size, stride, padding, dilation, groups, bias)

        self.register_buffer('myMask', torch.eye(kernel_size))
        # self.eyeTenor = torch.eye(kernel_size)  # 创建对角矩阵n*n
        # self.eyeTenor.requires_grad = False

        self.conv1.weight.data = self.conv1.weight.data * self.myMask  # 卷积核矩阵的维度为：[输出通道，输入通道，Kh，Kw]
        self.conv1.weight.register_hook(self.tensor_hook)  # tensor_hook会接受到的参数维度为：[输出通道，输入通道，Kh，Kw]

    def tensor_hook(self, grad):
        grad *= self.myMask
        return grad

    def forward(self, x):
        assert self.conv1.weight.data[0, 0, 0, 1] == 0  # 确认只有主对角线上的权重非0，其他权重都为0
        x = self.conv1(x)
        return x


class diagConv2d_n(nn.Module):
    """反对角线卷积"""
    def __init__(self, inChannels, outChannels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(diagConv2d_n, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size, stride, padding, dilation, groups, bias)

        self.register_buffer('myMask', torch.eye(kernel_size).flip([0]).contiguous())
        # self.eyeTenor = torch.eye(kernel_size).flip([0]).contiguous()  # 创建对角矩阵n*n
        # self.eyeTenor.requires_grad = False

        self.conv1.weight.data = self.conv1.weight.data * self.myMask  # 卷积核矩阵的维度为：[输出通道，输入通道，Kh，Kw]
        self.conv1.weight.register_hook(self.tensor_hook)  # tensor_hook会接受到的参数维度为：[输出通道，输入通道，Kh，Kw]

    def tensor_hook(self, grad):
        grad *= self.myMask
        return grad

    def forward(self, x):
        assert self.conv1.weight.data[0, 0, 0, 0] == 0  # 确认只有反对角线上的权重非0，其他权重都为0.
        x = self.conv1(x)
        return x