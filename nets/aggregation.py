import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax

from nets.deform import SimpleBottleneck, DeformSimpleBottleneck


def conv3d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   bias=False, groups=groups),
                         nn.BatchNorm3d(out_channels),
                         nn.LeakyReLU(0.2, inplace=True))


# Used in PSMNet
def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
        nn.BatchNorm3d(out_planes))


def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   bias=False, groups=groups),
                         nn.BatchNorm2d(out_channels),
                         nn.LeakyReLU(0.2, inplace=True))


def conv1x1(in_planes, out_planes):
    """1x1 convolution, used for pointwise conv"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                                   padding=0, bias=False),
                         nn.BatchNorm2d(out_planes),
                         nn.LeakyReLU(0.2, inplace=True))


# Used for StereoNet feature extractor
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, with_bn_relu=False):
    """3x3 convolution with padding"""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
    if with_bn_relu:
        conv = nn.Sequential(conv,
                             nn.BatchNorm2d(out_planes),
                             nn.ReLU(inplace=True))
    return conv


# Used for GCNet for aggregation
def conv3x3_3d(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=3,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   groups=groups, bias=False),
                         nn.BatchNorm3d(out_planes),
                         nn.ReLU(inplace=True))


def trans_conv3x3_3d(in_channels, out_channels, stride=1, groups=1, dilation=1):
    return nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3,
                                            stride=stride, padding=dilation,
                                            output_padding=dilation,
                                            groups=groups, dilation=dilation,
                                            bias=False),
                         nn.BatchNorm3d(out_channels),
                         nn.ReLU(inplace=True))


class StereoNetAggregation(nn.Module):
    def __init__(self, in_channels=32):
        super(StereoNetAggregation, self).__init__()

        aggregation_modules = nn.ModuleList()

        # StereoNet uses four 3d conv
        for _ in range(4):
            aggregation_modules.append(conv3d(in_channels, in_channels))
        self.aggregation_layer = nn.Sequential(*aggregation_modules)

        self.final_conv = nn.Conv3d(in_channels, 1, kernel_size=3, stride=1,
                                    padding=1, bias=True)

    def forward(self, cost_volume):
        assert cost_volume.dim() == 5  # [B, C, D, H, W]

        out = self.aggregation_layer(cost_volume)
        out = self.final_conv(out)  # [B, 1, D, H, W]
        out = out.squeeze(1)  # [B, D, H, W]

        return out


class PSMNetBasicAggregation(nn.Module):
    """12 3D conv"""

    def __init__(self, max_disp):
        super(PSMNetBasicAggregation, self).__init__()
        self.max_disp = max_disp

        conv0 = convbn_3d(64, 32, 3, 1, 1)
        conv1 = convbn_3d(32, 32, 3, 1, 1)

        final_conv = nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)

        self.dres0 = nn.Sequential(conv0,
                                   nn.ReLU(inplace=True),
                                   conv1,
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(conv1,
                                   nn.ReLU(inplace=True),
                                   conv1)

        self.dres2 = nn.Sequential(conv1,
                                   nn.ReLU(inplace=True),
                                   conv1)

        self.dres3 = nn.Sequential(conv1,
                                   nn.ReLU(inplace=True),
                                   conv1)

        self.dres4 = nn.Sequential(conv1,
                                   nn.ReLU(inplace=True),
                                   conv1)

        self.classify = nn.Sequential(conv1,
                                      nn.ReLU(inplace=True),
                                      final_conv)

    def forward(self, cost):
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0
        cost0 = self.dres2(cost0) + cost0
        cost0 = self.dres3(cost0) + cost0
        cost0 = self.dres4(cost0) + cost0

        cost = self.classify(cost0)  # [B, 1, 48, H/4, W/4]
        cost = F.interpolate(cost, scale_factor=4, mode='trilinear')

        cost = torch.squeeze(cost, 1)  # [B, 192, H, W]

        return [cost]


# PSMNet Hourglass network
class PSMNetHourglass(nn.Module):
    def __init__(self, inplanes):
        super(PSMNetHourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


class PSMNetHGAggregation(nn.Module):
    """22 3D conv"""

    def __init__(self, max_disp):
        super(PSMNetHGAggregation, self).__init__()
        self.max_disp = max_disp

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),  # [in_planes, out_planes, kernel_size, stride, pad]
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))  # [B, 32, D/4, H/4, W/4]

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))  # [B, 32, D/4, H/4, W/4]

        self.dres2 = PSMNetHourglass(32)

        self.dres3 = PSMNetHourglass(32)

        self.dres4 = PSMNetHourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

    def forward(self, cost):
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        cost3 = F.interpolate(cost3, scale_factor=4, mode='trilinear')
        cost3 = torch.squeeze(cost3, 1)

        if self.training:
            cost1 = F.interpolate(cost1, scale_factor=4, mode='trilinear')
            cost2 = F.interpolate(cost2, scale_factor=4, mode='trilinear')

            cost1 = torch.squeeze(cost1, 1)
            cost2 = torch.squeeze(cost2, 1)

            return [cost1, cost2, cost3]

        return [cost3]


class GCNetAggregation(nn.Module):
    def __init__(self):
        super(GCNetAggregation, self).__init__()
        self.conv1 = nn.Sequential(conv3x3_3d(64, 32),
                                   conv3x3_3d(32, 32))  # H/2

        self.conv2a = conv3x3_3d(64, 64, stride=2)  # H/4
        self.conv2b = nn.Sequential(conv3x3_3d(64, 64),
                                    conv3x3_3d(64, 64))  # H/4

        self.conv3a = conv3x3_3d(64, 64, stride=2)  # H/8
        self.conv3b = nn.Sequential(conv3x3_3d(64, 64),
                                    conv3x3_3d(64, 64))  # H/8

        self.conv4a = conv3x3_3d(64, 64, stride=2)  # H/16
        self.conv4b = nn.Sequential(conv3x3_3d(64, 64),
                                    conv3x3_3d(64, 64))  # H/16

        self.conv5a = conv3x3_3d(64, 128, stride=2)  # H/32
        self.conv5b = nn.Sequential(conv3x3_3d(128, 128),
                                    conv3x3_3d(128, 128))  # H/32

        self.trans_conv1 = trans_conv3x3_3d(128, 64, stride=2)  # H/16
        self.trans_conv2 = trans_conv3x3_3d(64, 64, stride=2)  # H/8
        self.trans_conv3 = trans_conv3x3_3d(64, 64, stride=2)  # H/4
        self.trans_conv4 = trans_conv3x3_3d(64, 32, stride=2)  # H/2
        self.trans_conv5 = nn.ConvTranspose3d(32, 1, kernel_size=3,
                                              stride=2, padding=1,
                                              groups=1, dilation=1,
                                              bias=False)  # H

    def forward(self, cost_volume):
        conv1 = self.conv1(cost_volume)  # H/2
        conv2a = self.conv2a(cost_volume)  # H/4
        conv2b = self.conv2b(conv2a)  # H/4
        conv3a = self.conv3a(conv2a)  # H/8
        conv3b = self.conv3b(conv3a)  # H/8
        conv4a = self.conv4a(conv3a)  # H/16
        conv4b = self.conv4b(conv4a)  # H/16
        conv5a = self.conv5a(conv4a)  # H/32
        conv5b = self.conv5b(conv5a)  # H/32
        trans_conv1 = self.trans_conv1(conv5b)  # H/16
        trans_conv2 = self.trans_conv2(trans_conv1 + conv4b)  # H/8
        trans_conv3 = self.trans_conv3(trans_conv2 + conv3b)  # H/4
        trans_conv4 = self.trans_conv4(trans_conv3 + conv2b)  # H/2
        trans_conv5 = self.trans_conv5(trans_conv4 + conv1)  # H

        out = torch.squeeze(trans_conv5, 1)  # [B, D, H, W]

        return out


# Adaptive intra-scale aggregation & adaptive cross-scale aggregation
class AdaptiveAggregationModule(nn.Module):
    def __init__(self, num_scales, num_output_branches, max_disp,
                 num_blocks=1,
                 simple_bottleneck=False,
                 deformable_groups=2,
                 mdconv_dilation=2):
        super(AdaptiveAggregationModule, self).__init__()

        self.num_scales = num_scales
        self.num_output_branches = num_output_branches
        self.max_disp = max_disp
        self.num_blocks = num_blocks

        self.branches = nn.ModuleList()

        # Adaptive intra-scale aggregation
        for i in range(self.num_scales):
            num_candidates = max_disp // (2 ** i)
            branch = nn.ModuleList()
            for j in range(num_blocks):
                if simple_bottleneck:
                    branch.append(SimpleBottleneck(num_candidates, num_candidates))
                else:
                    branch.append(DeformSimpleBottleneck(num_candidates, num_candidates, modulation=True,
                                                         mdconv_dilation=mdconv_dilation,
                                                         deformable_groups=deformable_groups))

            self.branches.append(nn.Sequential(*branch))

        self.fuse_layers = nn.ModuleList()

        # Adaptive cross-scale aggregation
        # For each output branch
        for i in range(self.num_output_branches):
            self.fuse_layers.append(nn.ModuleList())
            # For each branch (different scale)
            for j in range(self.num_scales):
                if i == j:
                    # Identity
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    self.fuse_layers[-1].append(
                        nn.Sequential(nn.Conv2d(max_disp // (2 ** j), max_disp // (2 ** i),
                                                kernel_size=1, bias=False),
                                      nn.BatchNorm2d(max_disp // (2 ** i)),
                                      ))
                elif i > j:
                    layers = nn.ModuleList()
                    for k in range(i - j - 1):
                        layers.append(nn.Sequential(nn.Conv2d(max_disp // (2 ** j), max_disp // (2 ** j),
                                                              kernel_size=3, stride=2, padding=1, bias=False),
                                                    nn.BatchNorm2d(max_disp // (2 ** j)),
                                                    nn.LeakyReLU(0.2, inplace=True),
                                                    ))

                    layers.append(nn.Sequential(nn.Conv2d(max_disp // (2 ** j), max_disp // (2 ** i),
                                                          kernel_size=3, stride=2, padding=1, bias=False),
                                                nn.BatchNorm2d(max_disp // (2 ** i))))
                    self.fuse_layers[-1].append(nn.Sequential(*layers))

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        assert len(self.branches) == len(x)

        for i in range(len(self.branches)):
            branch = self.branches[i]
            for j in range(self.num_blocks):
                dconv = branch[j]
                x[i] = dconv(x[i])

        if self.num_scales == 1:  # without fusions
            return x

        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    exchange = self.fuse_layers[i][j](x[j])
                    if exchange.size()[2:] != x_fused[i].size()[2:]:
                        exchange = F.interpolate(exchange, size=x_fused[i].size()[2:],
                                                 mode='bilinear', align_corners=False)
                    x_fused[i] = x_fused[i] + exchange

        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused


# Stacked AAModules
class AdaptiveAggregation(nn.Module):
    def __init__(self, max_disp, num_scales=3, num_fusions=6,
                 num_stage_blocks=1,
                 num_deform_blocks=2,
                 intermediate_supervision=True,
                 deformable_groups=2,
                 mdconv_dilation=2):
        super(AdaptiveAggregation, self).__init__()

        self.max_disp = max_disp  # 最高分辨率代价体的最大视差
        self.num_scales = num_scales
        self.num_fusions = num_fusions
        self.intermediate_supervision = intermediate_supervision

        fusions = nn.ModuleList()
        for i in range(num_fusions):
            # 共使用6个AAModules，其中最后三个使用了变形卷积
            if self.intermediate_supervision:
                num_out_branches = self.num_scales
            else:
                num_out_branches = 1 if i == num_fusions - 1 else self.num_scales

            if i >= num_fusions - num_deform_blocks:
                simple_bottleneck_module = False
            else:
                simple_bottleneck_module = True

            fusions.append(AdaptiveAggregationModule(num_scales=self.num_scales,
                                                     num_output_branches=num_out_branches,
                                                     max_disp=max_disp,
                                                     num_blocks=num_stage_blocks,
                                                     mdconv_dilation=mdconv_dilation,
                                                     deformable_groups=deformable_groups,
                                                     simple_bottleneck=simple_bottleneck_module))

        self.fusions = nn.Sequential(*fusions)

        self.final_conv = nn.ModuleList()
        for i in range(self.num_scales):
            in_channels = max_disp // (2 ** i)

            self.final_conv.append(nn.Conv2d(in_channels, max_disp // (2 ** i), kernel_size=1))

            if not self.intermediate_supervision:
                break

    def forward(self, cost_volume):
        assert isinstance(cost_volume, list)

        for i in range(self.num_fusions):
            fusion = self.fusions[i]
            cost_volume = fusion(cost_volume)

        # Make sure the final output is in the first position
        out = []  # H/3, H/6, H/12
        for i in range(len(self.final_conv)):
            out = out + [self.final_conv[i](cost_volume[i])]

        return out


# 尝试用对角线卷积做代价聚合
class myAttentionCostAggregation(nn.Module):
    def __init__(self, max_disp, num_scales=3, num_fusions=2,  # 共多少级处理
                 num_stage_blocks=1,
                 num_deform_blocks=2,  # 在num_fusions级中，共有多少个是特殊模块（Attention代价聚合）
                 intermediate_supervision=True,
                 deformable_groups=2,
                 mdconv_dilation=2):
        super(myAttentionCostAggregation, self).__init__()
        # 在这里调节参数
        num_scales = 3
        num_stage_blocks = 1
        intermediate_supervision = True
        deformable_groups = 2
        mdconv_dilation = 2  # 无用参数

        # 需要调节的参数
        num_fusions = 4  # 共多少级处理
        num_attention_blocks = 2  # 在num_fusions级中，使用多少个Attention代价聚合模块
        num_deform_blocks = 2  # 在num_fusions级中，使用多少个变形卷积模块

        self.max_disp = max_disp  # 最高分辨率代价体的最大视差
        self.num_scales = num_scales
        self.num_fusions = num_fusions  # 共多少级处理
        self.intermediate_supervision = intermediate_supervision

        fusions = nn.ModuleList()
        for i in range(num_fusions):
            # 共使用num_fusions级处理，其中最后三个使用了变形卷积
            if self.intermediate_supervision:
                num_out_branches = self.num_scales
            else:
                num_out_branches = 1 if i == num_fusions - 1 else self.num_scales

            # 1.Attention代价聚合。2.变形卷积
            if i < num_attention_blocks:
                # num_fusions级处理中，前面的num_attention_blocks级使用Attention代价聚合（simple_bottleneck_module=1）
                simple_bottleneck_module = 1
            else:
                # num_fusions级处理中，最后的num_deform_blocks级使用变形卷积（simple_bottleneck_module=2）
                simple_bottleneck_module = 2

            fusions.append(myAttentionCostAggModule(num_scales=self.num_scales,
                                                    num_output_branches=num_out_branches,
                                                    max_disp=max_disp,
                                                    num_blocks=num_stage_blocks,
                                                    mdconv_dilation=mdconv_dilation,
                                                    deformable_groups=deformable_groups,
                                                    simple_bottleneck=simple_bottleneck_module))

        self.fusions = nn.Sequential(*fusions)

        self.final_conv = nn.ModuleList()
        for i in range(self.num_scales):
            in_channels = max_disp // (2 ** i)

            self.final_conv.append(nn.Conv2d(in_channels, max_disp // (2 ** i), kernel_size=1))

            if not self.intermediate_supervision:
                break

    def forward(self, cost_volume, left_feature, right_feature=None):
        """
        cost_volume：多尺度代价体；
        left_feature：
        right_feature：降维后的左右图特征向量，以左图为例：[[尺度1：query,key],[尺度2：query,key],[尺度3：query,key]]
        """
        # cost_volume[B, D, H, W]和left_feature[B, C, H, W]都是三尺度的3D代价体：H/3,H/6,H/12, D=64,32,16, C=128,128,128
        assert isinstance(cost_volume, list)

        # 把cost_volume和Feature送入第一个Attention聚合模块，并将结果送入第二个Attention聚合模块，依次类推。
        for i in range(self.num_fusions):  # 共经过self.num_fusions级处理
            fusion = self.fusions[i]
            cost_volume = fusion(cost_volume, left_feature, right_feature)

        # Make sure the final output is in the first position
        out = []  # H/3, H/6, H/12
        for i in range(len(self.final_conv)):
            out = out + [self.final_conv[i](cost_volume[i])]

        return out


# 使用Attention进行代价聚合：在这里真正发生
class myAttentionCostAggModule(nn.Module):
    def __init__(self, num_scales, num_output_branches, max_disp,
                 num_blocks=1,
                 simple_bottleneck=0,
                 deformable_groups=2,
                 mdconv_dilation=2):
        super(myAttentionCostAggModule, self).__init__()

        self.num_scales = num_scales
        self.max_disp = max_disp
        self.num_blocks = num_blocks
        self.num_output_branches = num_output_branches
        self.feature_channels = [64, 48, 32]  # 特征的通道数: 已在进入代价聚合模块之前，进行了降维
        self.simple_bottleneck = simple_bottleneck

        # 基于Attention的尺度内代价聚合
        self.branches = nn.ModuleList()  # 一个尺度，一个branch
        for i in range(self.num_scales):
            disp_candidates = max_disp // (2 ** i)  # 本尺度下的视差范围
            branch = nn.ModuleList()  # 本尺度下的处理流程
            for j in range(self.num_blocks):  # 本尺度下的处理流程，包含多少个Block
                # 1.Attention代价聚合。2.变形卷积
                if simple_bottleneck == 1:
                    # Attention代价聚合模块: 不考虑A/C相似性
                    # branch.append(feature_Attention_CostAgg_Module(self.feature_channels[i], disp_candidates))
                    # warp Attention代价聚合模块：考虑A/C相似性
                    branch.append(warp_feature_Attention_CostAgg_Module(self.feature_channels[i], disp_candidates))
                elif simple_bottleneck == 2:
                    # 变形卷积代价聚合模块
                    branch.append(DeformSimpleBottleneck(disp_candidates, disp_candidates, modulation=True,
                                                         mdconv_dilation=mdconv_dilation,
                                                         deformable_groups=deformable_groups))
                    # branch.append(feature_Attention_CostAgg_Module(self.feature_channels[i], disp_candidates))
                    # branch.append(SimpleBottleneck(num_candidates, num_candidates))
                else:
                    raise NotImplementedError

            self.branches.append(nn.Sequential(*branch))  # 一个尺度，一个branch

        # 尺度间代价聚合
        # 尺度间代价体上下采样，并融合
        # [尺度i,尺度j]，数值越小，分辨率越高。如下的上下采样操作，都是为了把尺度j的代价体，变成和尺度i一致！！！
        self.fuse_layers = nn.ModuleList()  # （一个输出分支）一个尺度，一个fuse_layer
        for i in range(self.num_output_branches):  # 遍历所有的（输出分支）输出尺度：[尺度i,*]
            self.fuse_layers.append(nn.ModuleList())
            # For each branch (different scale)
            for j in range(self.num_scales):  # 遍历所有的（输出分支）输出尺度,形成尺度对：[尺度i,尺度j]
                if i == j:
                    # 同一尺度：尺度i = 尺度j, 无需上下采样，identity即可
                    # Identity
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    # 尺度i分辨率 > 尺度j分辨率, 需对尺度j代价体上采样（插值），故先对尺度j卷积一下
                    self.fuse_layers[-1].append(
                        nn.Sequential(nn.Conv2d(max_disp // (2 ** j), max_disp // (2 ** i),
                                                kernel_size=1, bias=False),
                                      nn.BatchNorm2d(max_disp // (2 ** i)),
                                      ))
                elif i > j:
                    # 尺度i分辨率 < 尺度j分辨率, 需对尺度j代价体下采样，
                    # 故对j代价体进行“卷积下采样”（ij相差两个以上的尺度则使用LeakyReLU，否则不使用LeakyReLU）
                    layers = nn.ModuleList()
                    for k in range(i - j - 1):
                        layers.append(nn.Sequential(nn.Conv2d(max_disp // (2 ** j), max_disp // (2 ** j),
                                                              kernel_size=3, stride=2, padding=1, bias=False),
                                                    nn.BatchNorm2d(max_disp // (2 ** j)),
                                                    nn.LeakyReLU(0.2, inplace=True),
                                                    ))

                    layers.append(nn.Sequential(nn.Conv2d(max_disp // (2 ** j), max_disp // (2 ** i),
                                                          kernel_size=3, stride=2, padding=1, bias=False),
                                                nn.BatchNorm2d(max_disp // (2 ** i))))
                    self.fuse_layers[-1].append(nn.Sequential(*layers))

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, cost_volume, left_feature, right_feature=None):
        # cost_volume: list, 三个尺度的代价体, H/3, H/6, H/12, 通道数(视差维度)分别为：D=64,32,16
        # left_feature：list, 三个尺度的特征, H/3, H/6, H/12, 通道数都为128
        assert len(self.branches) == len(cost_volume) and len(cost_volume) == len(left_feature)

        # 基于Attention的尺度内代价聚合: 一个尺度，一个branch
        for i in range(len(self.branches)):
            branch = self.branches[i]  # 当前尺度i的branch
            for j in range(self.num_blocks):
                dconv = branch[j]  # 当前branch的流程块
                # 1.Attention代价聚合。2.变形卷积
                if self.simple_bottleneck == 1:
                    cost_volume[i] = dconv(cost_volume[i],
                                           left_feature[i])  # cost_volume, left_feature, right_feature=None
                elif self.simple_bottleneck == 2:
                    cost_volume[i] = dconv(cost_volume[i])

        if self.num_scales == 1:  # without fusions
            return cost_volume

        x_fused = []  # 一个尺度的输出是x_fused的一个元素
        for i in range(len(self.fuse_layers)):  # 遍历所有的（输出分支）输出尺度：[尺度i,*]
            for j in range(len(self.branches)):  # 遍历所有的（输出分支）输出尺度,形成尺度对：[尺度i,尺度j],数值越小，分辨率越高
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](cost_volume[0]))
                else:
                    exchange = self.fuse_layers[i][j](cost_volume[j])
                    if exchange.size()[2:] != x_fused[i].size()[2:]:
                        exchange = F.interpolate(exchange, size=x_fused[i].size()[2:],
                                                 mode='bilinear', align_corners=False)
                    x_fused[i] = x_fused[i] + exchange

        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused


def INF(B, H, W):
    # [H] -> [H, H]->[1, H, H] -> [BW, H, H]
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class feature_Attention_CostAgg_Module(nn.Module):
    def __init__(self, feature_channels, disp_candidates, recurrence=2):
        super(feature_Attention_CostAgg_Module, self).__init__()

        # TODO: 在此处调节递归Attention的递归次数
        self.recurrence = recurrence

        # 对代价体的卷积
        self.conva = nn.Sequential(nn.Conv2d(disp_candidates, disp_candidates, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(disp_candidates))

        self.cca = CostAgg_CrissCrossAttention(feature_channels, disp_candidates)

        # 对代价体的卷积
        self.convb = nn.Sequential(nn.Conv2d(disp_candidates, disp_candidates, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(disp_candidates))

    def forward(self, cost_volume, left_feature, right_feature=None):
        # cost_volume: 单尺度代价体
        # left_feature：单尺度特征
        output = self.conva(cost_volume)

        for i in range(self.recurrence):
            output = self.cca(output, left_feature, right_feature)
        output = self.convb(output)

        return output


class CostAgg_CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, f_qurey_chls, value_chls):
        super(CostAgg_CrissCrossAttention, self).__init__()

        # 通道数需要调节：特征已在特征提取模块进行了降维
        self.query_conv = nn.Conv2d(in_channels=f_qurey_chls, out_channels=f_qurey_chls, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=f_qurey_chls, out_channels=f_qurey_chls, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=value_chls, out_channels=value_chls, kernel_size=1)

        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, cost_volume, left_feature, right_feature=None):
        # [B, C, H, W]
        m_batchsize, _, height, width = cost_volume.size()

        # proj_query = self.query_conv(left_feature)
        proj_query = self.query_conv(left_feature[0])
        # [B, C, H, W] -> [B, W, C, H] -> [BW, C, H] -> [BW, H, C]
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        # [B, C, H, W] -> [B, H, C, W] -> [BH, C, W] -> [BH, W, C]
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)

        # proj_key = self.key_conv(left_feature)
        proj_key = self.key_conv(left_feature[1])
        # [B, C, H, W] -> [B, W, C, H] -> [BW, C, H]
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        # [B, C, H, W] -> [B, H, C, W] -> [BH, C, W]
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_value = self.value_conv(cost_volume)
        # [B, C, H, W] -> [B, W, C, H] -> [BW, C, H]
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        # [B, C, H, W] -> [B, H, C, W] -> [BH, C, W]
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        # 负无穷的作用是：十字交叉Attention对于像素与其自身会计算两层Attention，故需去掉一个。加上负无穷，在Softmax的时候，其权重就会变成0.
        # [BW, H, C] * [BW, C, H] = [BW, H, H]: H维度（垂直方向）上的像素之间的Attention  + [BW, H, H]对角负无穷矩阵  -> [B, H, W, H]
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(
            # [BW, H, H] -> [B, W, H, H] -> [B, H, W, H]
            m_batchsize, width, height, height).permute(0, 2, 1, 3)
        # [BH, W, C] * [BH, C, W] = [BH, W, W] -> [B, H, W, W]
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        # [B, H, W, H] || [B, H, W, W] -> [B, H, W, H+W] -> 在最后一维上做Softmax
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        # [B, H, W, H + W]取出[B, H, W, 0:H]->[B, W, H, H]-> [BW, H, H]
        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # [B, H, W, H + W]取出[B, H, W, H : W+H]->[BH, W, W]
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)

        # value[BW, C, H] * (Attention[BW, H, H]->[BW, H, H]) -> [B, W, C, H]-> [B, C, H, W]
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        # value[BH, C, W] * (Attention[BH, W, W]->[BH, W, W]) -> [B, H, C, W]-> [B, C, H, W]
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + cost_volume


class FeatureShrinkModule(nn.Module):
    """
    给用于计算Attention的Feature降维，
    防止出现带着大体积的Feature进Forward的情况，减少显存占用
    """
    def __init__(self, num_scales=3):
        super(FeatureShrinkModule, self).__init__()

        self.in_channels = [128, 128, 128]  # AANet的特征提取模块，固定为128通道
        self.num_scales = num_scales
        # TODO: 在此处调节Attention的通道数, 目的是降低后续在计算Attention时计算量过大
        self.query_channels = [64, 48, 32]

        self.query_conv_s = nn.ModuleList()
        self.key_conv_s = nn.ModuleList()
        for i in range(self.num_scales):
            self.query_conv_s.append(nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels[i], out_channels=self.query_channels[i], kernel_size=1),
                nn.BatchNorm2d(self.query_channels[i]),
                nn.ReLU(inplace=True)))
            self.key_conv_s.append(nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels[i], out_channels=self.query_channels[i], kernel_size=1),
                nn.BatchNorm2d(self.query_channels[i]),
                nn.ReLU(inplace=True)))

    def forward(self, left_feature, right_feature=None):

        lft_rslt = []
        right_rlst = []
        for i in range(self.num_scales):
            left = []
            left.append(self.query_conv_s[i](left_feature[i]))
            left.append(self.key_conv_s[i](left_feature[i]))
            lft_rslt.append(left)

            right = []
            right.append(self.query_conv_s[i](right_feature[i]))
            right.append(self.key_conv_s[i](right_feature[i]))
            right_rlst.append(right)

        # 降维后的左右图特征向量，以左图为例：[[尺度1：query,key],[尺度2：query,key],[尺度3：query,key]]
        return lft_rslt, right_rlst


class warp_feature_Attention_CostAgg_Module(nn.Module):
    def __init__(self, feature_channels, disp_candidates, recurrence=2):
        super(warp_feature_Attention_CostAgg_Module, self).__init__()

        # TODO: 在此处调节递归Attention的递归次数
        self.recurrence = recurrence

        # 对代价体的卷积
        self.conva = nn.Sequential(nn.Conv2d(disp_candidates, disp_candidates, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(disp_candidates))

        self.cca = warp_CostAgg_CrissCrossAttention(feature_channels, disp_candidates)

        # 对代价体的卷积
        self.convb = nn.Sequential(nn.Conv2d(disp_candidates, disp_candidates, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(disp_candidates))

    def forward(self, cost_volume, left_feature, right_feature=None):
        # cost_volume: 单尺度代价体
        # left_feature：单尺度特征
        output = self.conva(cost_volume)

        for i in range(self.recurrence):
            output = self.cca(output, left_feature, right_feature)
        output = self.convb(output)

        return output


class warp_CostAgg_CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, f_qurey_chls, value_chls):
        super(warp_CostAgg_CrissCrossAttention, self).__init__()

        self.feature_similarity == 'difference'
        self.value_chls = value_chls

        # 通道数需要调节：特征已在特征提取模块进行了降维
        self.query_conv = nn.Conv2d(in_channels=f_qurey_chls, out_channels=f_qurey_chls, kernel_size=1)

        self.left_key_conv = nn.Conv2d(in_channels=f_qurey_chls, out_channels=f_qurey_chls, kernel_size=1)
        self.right_key_conv = nn.Conv2d(in_channels=f_qurey_chls, out_channels=f_qurey_chls, kernel_size=1)

        self.value_conv = nn.Conv2d(in_channels=value_chls, out_channels=value_chls, kernel_size=1)

        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        # self.gamma = nn.Parameter(torch.zeros(1))

        self.gammaList = nn.ParameterList([nn.Parameter(torch.zeros(1)) for i in range(self.value_chls)])

    def forward(self, cost_volume, left_feature, right_feature=None):
        """
        cost_volume： 单尺度代价体
        left_feature, right_feature：降维后的左/右图特征向量(单尺度的)，以左图为例：[query特征, key特征]
        """
        # TODO: working on ...
        left_key_feature = self.left_key_conv(left_feature[1])
        right_key_feature = self.right_key_conv(right_feature[1])

        b, c, h, w = left_key_feature.size()
        D_max = cost_volume.size(1)
        assert D_max == self.value_chls, 'D_max == self.value_chls Must holds!'

        # proj_query = self.query_conv(left_feature)
        proj_query = self.query_conv(left_feature[0])
        # [B, C, H, W] -> [B, W, C, H] -> [BW, C, H] -> [BW, H, C]
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)
        # [B, C, H, W] -> [B, H, C, W] -> [BH, C, W] -> [BH, W, C]
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)

        proj_value = self.value_conv(cost_volume)
        # [B, C, H, W] -> [B, W, C, H] -> [BW, C, H]
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
        # [B, C, H, W] -> [B, H, C, W] -> [BH, C, W]
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

        # 1. warp right_key_feature
        if self.feature_similarity == 'difference':
            warped_right_key_features = right_key_feature.new_zeros(b, c, D_max, h, w)  # [B, C, D, H, W] D=192/3

            for i in range(D_max):
                if i > 0:
                    warped_right_key_features[:, :, i, :, i:] = right_key_feature[:, :, :, :-i]
                else:
                    warped_right_key_features[:, :, i, :, :] = right_key_feature

        # 2. feature_mix
        # [B, C, D, H, W] + ([B, C, H, W] -> [B, C, 1, H, W]) -> [B, C, D, H, W]
        # TODO：两者特征直接相加，经过Softmax之后就相当于相乘，貌似比较合理了。还有更好的方法吗？经过卷积？
        mixed_features = warped_right_key_features + left_key_feature.unsqueeze(2)

        # 3. 计算Attention
        for i in range(D_max):
            # 3.1 针对每一个视差值, 计算Attention。因为不同的视差下，右特征图的warp偏移不一样。
            slice_feature = mixed_features[:, :, i, :, :]  # [B, C, D=i, H, W] ->[B, C, H, W]
            # mixed_proj_key = self.left_key_conv(slice_feature)
            # [B, C, H, W] -> [B, W, C, H] -> [BW, C, H]
            mixed_proj_key_H = slice_feature.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
            # [B, C, H, W] -> [B, H, C, W] -> [BH, C, W]
            mixed_proj_key_W = slice_feature.permute(0, 2, 1, 3).contiguous().view(b * w, -1, w)

            # 负无穷的作用是：十字交叉Attention对于像素与其自身会计算两层Attention，故需去掉一个。加上负无穷，在Softmax的时候，其权重就会变成0.
            # [BW, H, C] * [BW, C, H] = [BW, H, H]: H维度（垂直方向）上的像素之间的Attention  + [BW, H, H]对角负无穷矩阵  -> [B, H, W, H]
            energy_H = (torch.bmm(proj_query_H, mixed_proj_key_H) + self.INF(b, h, w)).view(
                # [BW, H, H] -> [B, W, H, H] -> [B, H, W, H]
                b, w, h, h).permute(0, 2, 1, 3)
            # [BH, W, C] * [BH, C, W] = [BH, W, W] -> [B, H, W, W]
            energy_W = torch.bmm(proj_query_W, mixed_proj_key_W).view(b, h, w, w)
            # [B, H, W, H] || [B, H, W, W] -> [B, H, W, H+W] -> 在最后一维上做Softmax
            concate = self.softmax(torch.cat([energy_H, energy_W], 3))

            # [B, H, W, H + W]取出[B, H, W, 0:H]->[B, W, H, H]-> [BW, H, H]
            att_H = concate[:, :, :, 0:h].permute(0, 2, 1, 3).contiguous().view(b * w, h, h)
            # [B, H, W, H + W]取出[B, H, W, H : W+H]->[BH, W, W]
            att_W = concate[:, :, :, h:h + w].contiguous().view(b * h, w, w)

            # 3.2 对代价体的每一个视差Slice，进行加权聚合
            # value[BW, C=1, H] * (Attention[BW, H, H]->[BW, H, H]) -> [B, W, C=1, H]-> [B, C=1, H, W]
            out_H = torch.bmm(proj_value_H[:, i:i + 1, :], att_H.permute(0, 2, 1)).view(b, w, -1, h).permute(0, 2, 3, 1)
            # value[BH, C=1, W] * (Attention[BH, W, W]->[BH, W, W]) -> [B, H, C=1, W]-> [B, C=1, H, W]
            out_W = torch.bmm(proj_value_W[:, i:i + 1, :], att_W.permute(0, 2, 1)).view(b, h, -1, w).permute(0, 2, 1, 3)

            # 3.3 保存聚合后的代价体Slice
            # cost_volume[B, C/D=i, H, W]
            cost_volume[:, i, :, :] = self.gammaList[i] * (out_H + out_W) + cost_volume[:, i, :, :]

        return cost_volume

        # # 对代价体的每一个Slice进行处理：[B, D=i, H, W]
        # for i in range(D_max):
        #     warped_right_feature = warpFeature(right_feature, disparity)
        #     mixed_feature = feature_mix(left_feature, warped_right_feature)
        #     计算Attention
        #     对代价体的当前Slice进行加权[B, D=i, H, W]，得到新的代价体Slice[B, D=i, H, W]
        #
        #
        #
        #
        # # [B, C, H, W]
        # m_batchsize, _, height, width = cost_volume.size()
        #
        # # proj_query = self.query_conv(left_feature)
        # proj_query = self.query_conv(left_feature[0])
        # # [B, C, H, W] -> [B, W, C, H] -> [BW, C, H] -> [BW, H, C]
        # proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        # # [B, C, H, W] -> [B, H, C, W] -> [BH, C, W] -> [BH, W, C]
        # proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        #
        # # proj_key = self.key_conv(left_feature)
        # left_proj_key = self.left_key_conv(left_feature[1])
        # # [B, C, H, W] -> [B, W, C, H] -> [BW, C, H]
        # left_proj_key_H = left_proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        # # [B, C, H, W] -> [B, H, C, W] -> [BH, C, W]
        # left_proj_key_W = left_proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        #
        # proj_value = self.value_conv(cost_volume)
        # # [B, C, H, W] -> [B, W, C, H] -> [BW, C, H]
        # proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        # # [B, C, H, W] -> [B, H, C, W] -> [BH, C, W]
        # proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        #
        # # 负无穷的作用是：十字交叉Attention对于像素与其自身会计算两层Attention，故需去掉一个。加上负无穷，在Softmax的时候，其权重就会变成0.
        # # [BW, H, C] * [BW, C, H] = [BW, H, H]: H维度（垂直方向）上的像素之间的Attention  + [BW, H, H]对角负无穷矩阵  -> [B, H, W, H]
        # energy_H = (torch.bmm(proj_query_H, left_proj_key_H) + self.INF(m_batchsize, height, width)).view(
        #                                                         # [BW, H, H] -> [B, W, H, H] -> [B, H, W, H]
        #                                                         m_batchsize, width, height, height).permute(0, 2, 1, 3)
        # # [BH, W, C] * [BH, C, W] = [BH, W, W] -> [B, H, W, W]
        # energy_W = torch.bmm(proj_query_W, left_proj_key_W).view(m_batchsize,height,width,width)
        # # [B, H, W, H] || [B, H, W, W] -> [B, H, W, H+W] -> 在最后一维上做Softmax
        # concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        #
        # # [B, H, W, H + W]取出[B, H, W, 0:H]->[B, W, H, H]-> [BW, H, H]
        # att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        # # [B, H, W, H + W]取出[B, H, W, H : W+H]->[BH, W, W]
        # att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        #
        # # value[BW, C, H] * (Attention[BW, H, H]->[BW, H, H]) -> [B, W, C, H]-> [B, C, H, W]
        # out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        # # value[BH, C, W] * (Attention[BH, W, W]->[BH, W, W]) -> [B, H, C, W]-> [B, C, H, W]
        # out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        # #print(out_H.size(),out_W.size())
        # return self.gamma * (out_H + out_W) + cost_volume

# class CrissCrossAttention(nn.Module):
#     """ Criss-Cross Attention Module"""
#     def __init__(self, in_chs):
#         super(CrissCrossAttention,self).__init__()
#         self.chanel_in = in_chs
#         self.query_conv = nn.Conv2d(in_channels = in_chs, out_channels = in_chs // 8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels = in_chs, out_channels = in_chs // 8, kernel_size= 1)
#         self.value_conv = nn.Conv2d(in_channels = in_chs, out_channels = in_chs, kernel_size= 1)
#
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self,x):
#         proj_query = self.query_conv(x)
#         proj_key = self.key_conv(x)
#         proj_value = self.value_conv(x)
#
#
#         energy = ca_weight(proj_query, proj_key)
#         attention = F.softmax(energy, 1)
#         out = ca_map(attention, proj_value)
#         out = self.gamma*out + x
#
#         return out


# class CrissCrossAttention_PurePython(nn.Module):
#     """ Criss-Cross Attention Module"""
#     def __init__(self, in_dim):
#         super(CrissCrossAttention_PurePython,self).__init__()
#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#
#         self.softmax = nn.Softmax(dim=3)
#         self.INF = INF
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         # [B, C, H, W]
#         m_batchsize, _, height, width = x.size()
#
#         proj_query = self.query_conv(x)
#         # [B, C, H, W] -> [B, W, C, H] -> [BW, C, H] -> [BW, H, C]
#         proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
#         # [B, C, H, W] -> [B, H, C, W] -> [BH, C, W] -> [BH, W, C]
#         proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
#
#         proj_key = self.key_conv(x)
#         # [B, C, H, W] -> [B, W, C, H] -> [BW, C, H]
#         proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
#         # [B, C, H, W] -> [B, H, C, W] -> [BH, C, W]
#         proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
#
#         proj_value = self.value_conv(x)
#         # [B, C, H, W] -> [B, W, C, H] -> [BW, C, H]
#         proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
#         # [B, C, H, W] -> [B, H, C, W] -> [BH, C, W]
#         proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
#
#         # 负无穷的作用是：十字交叉Attention对于像素与其自身会计算两层Attention，故需去掉一个。加上负无穷，在Softmax的时候，其权重就会变成0.
#         # [BW, H, C] * [BW, C, H] = [BW, H, H]: H维度（垂直方向）上的像素之间的Attention  + [BW, H, H]对角负无穷矩阵  -> [B, H, W, H]
#         energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(
#                                                                 # [BW, H, H] -> [B, W, H, H] -> [B, H, W, H]
#                                                                 m_batchsize, width, height, height).permute(0, 2, 1, 3)
#         # [BH, W, C] * [BH, C, W] = [BH, W, W] -> [B, H, W, W]
#         energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
#         # [B, H, W, H] || [B, H, W, W] -> [B, H, W, H+W] -> 在最后一维上做Softmax
#         concate = self.softmax(torch.cat([energy_H, energy_W], 3))
#
#         # [B, H, W, H + W]取出[B, H, W, 0:H]->[B, W, H, H]-> [BW, H, H]
#         att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
#         # [B, H, W, H + W]取出[B, H, W, H : W+H]->[BH, W, W]
#         att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
#
#
#         # value[BW, C, H] * (Attention[BW, H, H]->[BW, H, H]) -> [B, W, C, H]-> [B, C, H, W]
#         out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
#         # value[BH, C, W] * (Attention[BH, W, W]->[BH, W, W]) -> [B, H, C, W]-> [B, C, H, W]
#         out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
#         #print(out_H.size(),out_W.size())
#         return self.gamma*(out_H + out_W) + x
