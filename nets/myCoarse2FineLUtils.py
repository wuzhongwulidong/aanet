import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.deform import SimpleBottleneck
from nets.warp import disp_warp


def myRegressNarrowDisp(cost_volume, dispRangeCenter, candidateNum, max_disp, regress_type='exp_avg',
                        similarityOrCost='sim'):
    assert cost_volume.dim() == 4  # [B, D, H, W]
    assert dispRangeCenter.dim() == 4  # [B, 1, H, W], 1: 可能的视差范围的中点 [min_disp + max_disp]/2
    assert cost_volume.size(1) == candidateNum

    if regress_type == 'exp_avg':
        # Matching similarity or matching cost
        cost_volume = cost_volume if similarityOrCost == 'sim' else -cost_volume
        prob_volume = F.softmax(cost_volume, dim=1)  # [B, D, H, W]

        dispCandiates = []
        for i in range(candidateNum):
            dispCandiates.append(dispRangeCenter[:, :, :, :] + (i - candidateNum // 2))

        dispCandiates = torch.cat(dispCandiates, dim=1)  # [B, candidateDisp, H, W]
        dispCandiates = torch.clamp(dispCandiates, min=0, max=max_disp)  # 超出视差范围的候选视差，置为上下限
        disp = torch.sum(prob_volume * dispCandiates, dim=1, keepdim=False)  # [B, D, H, W]->[B, H, W]
    else:
        raise NotImplementedError("regress_type not supported!")

    return disp


def mySampleCostVolume(preDisp, left_feature, right_feature, max_disp, sampleCount, subPixel=False):
    """
    :param preDisp: [B, 1, H, W]
    :param left_feature: [B, C, H, W]
    :param right_feature: [B, C, H, W]
    :return:
    """
    assert preDisp.size(2) == left_feature.size(2) and preDisp.size(3) == left_feature.size(3)
    b, c, h, w = left_feature.size()

    cost_volume = left_feature.new_zeros(b, sampleCount, h, w)  # [B, D ,H, W]
    # 采样每个像素的以preDisp([B, 1, H, W])为中点的sampleCount个视差，并估计其代价
    # 使用整型视差。当前认为，不必要太早地进行亚像素精度的视差估计。
    if not subPixel:
        preDisp = preDisp.round()

    dispCandiates = []
    for i in range(sampleCount):
        dispCandiate = torch.clamp(preDisp[:, :, :, :] + (i - sampleCount // 2), min=0, max=max_disp)
        right_warped_img = disp_warp(right_feature, dispCandiate)
        cost_volume[:, i, :, :] = (left_feature[:, :, :, :] *  # left_feature.shape=[B, C, H, W]
                                    right_warped_img[:, :, :, :]).mean(dim=1)

        dispCandiates.append(dispCandiate)

    cost_volume = cost_volume.contiguous()  # [B, sampleCount, H, W]
    dispCandiates = torch.cat(dispCandiates, dim=1)  # [B, sampleCount, H, W]

    return cost_volume, dispCandiates


def myRegressSampledDisp(cost_volume, dispCandiates, regress_type='exp_avg', similarityOrCost='sim'):
    """

    :param cost_volume: # [B, sampleCount, H/n, W/n]
    :param dispCandiates: , [B, sampleCount, H/n, W/n]
    :param regress_type:
    :param similarityOrCost:
    :return:
    """

    assert cost_volume.dim() == 4  # [B, sampleCount, H, W]
    assert cost_volume.size() == dispCandiates.size()

    if regress_type == 'exp_avg':
        # Matching similarity or matching cost
        cost_volume = cost_volume if similarityOrCost == 'sim' else -cost_volume
        prob_volume = F.softmax(cost_volume, dim=1)  # [B, D, H, W]

        disp = torch.sum(prob_volume * dispCandiates, 1, keepdim=True)  # [B, D, H, W]->[B, 1, H, W]
    else:
        raise NotImplementedError("regress_type not supported!")

    return disp  # [B, 1, H, W]


def myRegressInitDisp(cost_volume, max_disp, regress_type='exp_avg', similarityOrCost='sim'):
    assert cost_volume.dim() == 4  # [B, D, H, W]

    if regress_type == 'exp_avg':
        # Matching similarity or matching cost
        cost_volume = cost_volume if similarityOrCost == 'sim' else -cost_volume
        prob_volume = F.softmax(cost_volume, dim=1)  # [B, D, H, W]

        disp_candidates = torch.arange(0, max_disp).type_as(prob_volume)  # torch.arange() returns a 1-D tensor
        disp_candidates = disp_candidates.view(1, cost_volume.size(1), 1, 1)
        disp = torch.sum(prob_volume * disp_candidates, 1, keepdim=True)  # [B, D, H, W]->[B, 1, H, W]
    else:
        raise NotImplementedError("regress_type not supported!")

    return disp # [B, 1, H, W]


def myDispUpsample(disp, upScale=2):
    assert disp.dim() == 4  # [B, 1, H, W]

    upDisp = F.interpolate(disp, scale_factor=upScale, mode='bilinear', align_corners=False) * upScale

    return upDisp


class CostAggregation(nn.Module):
    """
    """
    def __init__(self, in_c, out_c, resblk_num=2):
        super(CostAggregation, self).__init__()
        self.conv0 = BasicConv2d(in_c, out_c, 1, 1, 0, 1)

        resblks = nn.ModuleList()
        for i in range(resblk_num):
            resblks.append(SimpleBottleneck(out_c, out_c))
        self.resblocks = nn.Sequential(*resblks)

        self.lastconv = nn.Conv2d(out_c, out_c, 1, 1, 0, 1, bias=False)

    def forward(self, costVolume):
        assert len(costVolume.size()) == 4  # [B, D, H, W]

        costVolume = self.conv0(costVolume)
        costVolume = self.resblocks(costVolume)
        costVolume = self.lastconv(costVolume)

        return costVolume


class coarse2fine_module(nn.Module):
    def __init__(self, max_disp, feature_similarity='correlation'):
        super(coarse2fine_module, self).__init__()
        self.max_disp = max_disp
        self.feature_similarity = feature_similarity

        # =======================1/16==================
        # cost_volume_16x:[B, D/16 ,H/16, W/16], feature: [B, C=32, H/16, W/16]
        self.max_disp_16x = self.max_disp // 16  # 192/16 = 12
        self.sCount_16x = self.max_disp_16x  # 12
        self.cost_volume_module_16x = CostVolume(self.max_disp_16x, feature_similarity)
        self.cost_aggregration_16x = CostAggregation(self.sCount_16x, self.sCount_16x, 2)  # in_c, out_c, resblk_num

        # =======================1/8==================
        self.max_disp_8x = self.max_disp // 8   # 192/8 = 24
        self.sCount_8x = self.max_disp_8x // 1  # 24
        # self.cost_volume_module_8x = CostVolume(max_disp // (2 ** 3), feature_similarity)
        self.cost_aggregration_8x = CostAggregation(self.sCount_8x, self.sCount_8x, 2)  # in_c, out_c, hid_c, resblk_num

        # =======================1/4==================
        self.max_disp_4x = self.max_disp // 4   # 192/4 = 48
        self.sCount_4x = self.max_disp_4x // 2  # 24
        # self.cost_volume_module_4x = CostVolume(max_disp // (2 ** 2), feature_similarity)
        self.cost_aggregration_4x = CostAggregation(self.sCount_4x, self.sCount_4x, 2)  # in_c, out_c, resblk_num

        # =======================1/2==================
        self.max_disp_2x = self.max_disp // 2   # 192/2 = 96
        self.sCount_2x = self.max_disp_2x // 4  # 24
        # self.cost_volume_module_2x = CostVolume(max_disp // (2 ** 1), feature_similarity)
        self.cost_aggregration_2x = CostAggregation(self.sCount_2x, self.sCount_2x, 2)  # in_c, out_c, resblk_num

        # =======================1/1==================
        self.max_disp_1x = self.max_disp // 1   # 192/1 = 192
        self.sCount_1x = self.max_disp_1x // 16  # 16
        # self.cost_volume_module_1x = CostVolume(max_disp // (2 ** 0), feature_similarity)
        self.cost_aggregration_1x = CostAggregation(self.sCount_1x, self.sCount_1x, 2)  # in_c, out_c, hid_c, resblk_num

    def forward(self, left_feature_pyramid, right_feature_pyramid):
        """
        coarse2fine_cost_construction
        :param left_feature_pyramid: # hitnet_feature H：1/16, 1/8, 1/4, 1/2, 1/1。C：32, 24, 24, 16, 16
        :param right_feature_pyramid:
        :param max_disp: 全尺寸图上的最大视差。
        :param feature_similarity: 构建代价体的方式。
        :return: 不同尺度的代价体。
        """
        # cost_volume_pyramid = []
        # =======================1/16==================
        # 注意这里CostVolume层的定义：因为CostVolume层中没有需要训练的参数，所以不必要把它保存在self中。否则，必须保存在self中。下同
        # cost_volume_16x:[B, D/16 ,H/16, W/16], feature: [B, C=32, H/16, W/16]
        cost_volume_16x = self.cost_volume_module_16x(left_feature_pyramid[0], right_feature_pyramid[0])  # H/16
        cost_volume_16x = self.cost_aggregration_16x(cost_volume_16x)
        disp_16x = myRegressInitDisp(cost_volume_16x, self.max_disp_16x)

        preDisp_8x = myDispUpsample(disp_16x, upScale=2)

        # =======================1/8==================
        cost_volume_8x, dispCandiates_8x = mySampleCostVolume(preDisp_8x, left_feature_pyramid[1], right_feature_pyramid[1],
                                            self.max_disp_8x, self.sCount_8x)  # [B, sampleCount, H/8, W/8], [B, sampleCount, H/8, W/8]
        cost_volume_8x = self.cost_aggregration_8x(cost_volume_8x)
        disp_8x = myRegressSampledDisp(cost_volume_8x, dispCandiates_8x)

        preDisp_4x = myDispUpsample(disp_8x, upScale=2)

        # =======================1/4==================
        cost_volume_4x, dispCandiates_4x = mySampleCostVolume(preDisp_4x, left_feature_pyramid[2], right_feature_pyramid[2],
                                            self.max_disp_4x, self.sCount_4x)  # [B, sampleCount, H/8, W/8], [B, sampleCount, H/8, W/8]
        cost_volume_4x = self.cost_aggregration_4x(cost_volume_4x)

        disp_4x = myRegressSampledDisp(cost_volume_4x, dispCandiates_4x)

        preDisp_2x = myDispUpsample(disp_4x, upScale=2)

        # =======================1/2==================
        cost_volume_2x, dispCandiates_2x = mySampleCostVolume(preDisp_2x, left_feature_pyramid[3], right_feature_pyramid[3],
                                            self.max_disp_2x, self.sCount_2x)  # [B, sampleCount, H/8, W/8], [B, sampleCount, H/8, W/8]
        cost_volume_2x = self.cost_aggregration_2x(cost_volume_2x)
        disp_2x = myRegressSampledDisp(cost_volume_2x, dispCandiates_2x)

        preDisp_1x = myDispUpsample(disp_2x, upScale=2)

        # =======================1/1==================
        cost_volume_1x, dispCandiates_1x = mySampleCostVolume(preDisp_1x, left_feature_pyramid[4], right_feature_pyramid[4],
                                            self.max_disp_1x, self.sCount_1x, subPixel=True)  # [B, sampleCount, H/8, W/8], [B, sampleCount, H/8, W/8]
        cost_volume_1x = self.cost_aggregration_1x(cost_volume_1x)
        disp_1x = myRegressSampledDisp(cost_volume_1x, dispCandiates_1x)

        disp_16x = disp_16x.squeeze(1)  # [B, H, W]
        disp_8x = disp_8x.squeeze(1)
        disp_4x = disp_4x.squeeze(1)
        disp_2x = disp_2x.squeeze(1)
        disp_1x = disp_1x.squeeze(1)

        return disp_16x, disp_8x, disp_4x, disp_2x, disp_1x  # 1/16, 1/8, 1/4, 1/2, 1/1


class CostVolume(nn.Module):
    def __init__(self, max_disp, feature_similarity='correlation'):
        """Construct cost volume based on different
        similarity measures

        Args:
            max_disp: max disparity candidate
            feature_similarity: type of similarity measure
        """
        super(CostVolume, self).__init__()

        self.max_disp = max_disp
        self.feature_similarity = feature_similarity

    def forward(self, left_feature, right_feature):
        b, c, h, w = left_feature.size()

        if self.feature_similarity == 'difference':
            cost_volume = left_feature.new_zeros(b, c, self.max_disp, h, w)  # [B, C, D, H, W] D=192/3

            for i in range(self.max_disp):
                if i > 0:
                    cost_volume[:, :, i, :, i:] = left_feature[:, :, :, i:] - right_feature[:, :, :, :-i]
                else:
                    cost_volume[:, :, i, :, :] = left_feature - right_feature

        elif self.feature_similarity == 'concat':
            cost_volume = left_feature.new_zeros(b, 2 * c, self.max_disp, h, w)  # [B, 2C, D, H, W]
            for i in range(self.max_disp):
                if i > 0:
                    cost_volume[:, :, i, :, i:] = torch.cat((left_feature[:, :, :, i:], right_feature[:, :, :, :-i]),
                                                            dim=1)
                else:
                    cost_volume[:, :, i, :, :] = torch.cat((left_feature, right_feature), dim=1)

        elif self.feature_similarity == 'correlation':
            cost_volume = left_feature.new_zeros(b, self.max_disp, h, w)  # [B, D ,H, W] D=192/3

            for i in range(self.max_disp):
                if i > 0:
                    cost_volume[:, i, :, i:] = (left_feature[:, :, :, i:] *  # left_feature.shape=[B, C, H, W]
                                                right_feature[:, :, :, :-i]).mean(dim=1)
                else:
                    cost_volume[:, i, :, :] = (left_feature * right_feature).mean(dim=1)

        else:
            raise NotImplementedError

        cost_volume = cost_volume.contiguous()  # [B, C, D, H, W] or [B, D, H, W]

        return cost_volume


def normalize_coords(grid):
    """Normalize coordinates of image scale to [-1, 1]
    Args:
        grid: [B, 2, H, W]
    """
    assert grid.size(1) == 2
    h, w = grid.size()[2:]
    grid[:, 0, :, :] = 2 * (grid[:, 0, :, :].clone() / (w - 1)) - 1  # x: [-1, 1]
    grid[:, 1, :, :] = 2 * (grid[:, 1, :, :].clone() / (h - 1)) - 1  # y: [-1, 1]
    grid = grid.permute((0, 2, 3, 1))  # [B, H, W, 2]
    return grid


def meshgrid(img, homogeneous=False):
    """Generate meshgrid in image scale
    Args:
        img: [B, _, H, W]
        homogeneous: whether to return homogeneous coordinates
    Return:
        grid: [B, 2, H, W]
    """
    b, _, h, w = img.size()

    x_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(img)  # [1, H, W]
    y_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(img)

    grid = torch.cat((x_range, y_range), dim=0)  # [2, H, W], grid[:, i, j] = [j, i]
    grid = grid.unsqueeze(0).expand(b, 2, h, w)  # [B, 2, H, W]

    if homogeneous:
        ones = torch.ones_like(x_range).unsqueeze(0).expand(b, 1, h, w)  # [B, 1, H, W]
        grid = torch.cat((grid, ones), dim=1)  # [B, 3, H, W]
        assert grid.size(1) == 3
    return grid


def disp_warp(img, disp, padding_mode='border'):
    """Warping by disparity
    Args:
        img: [B, 3, H, W]
        disp: [B, 1, H, W], positive
        padding_mode: 'zeros' or 'border'
    Returns:
        warped_img: [B, 3, H, W]
        valid_mask: [B, 3, H, W]ffind
    """
    assert disp.min() >= 0

    grid = meshgrid(img)  # [B, 2, H, W] in image scale
    # Note that -disp here
    offset = torch.cat((-disp, torch.zeros_like(disp)), dim=1)  # [B, 2, H, W]
    sample_grid = grid + offset
    sample_grid = normalize_coords(sample_grid)  # [B, H, W, 2] in [-1, 1]
    # warped_img = F.grid_sample(img, sample_grid, mode='bilinear', padding_mode=padding_mode)
    warped_img = F.grid_sample(img, sample_grid, mode='bilinear', padding_mode=padding_mode, align_corners=True)

    return warped_img


def BasicConv2d(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                  padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True, negative_slope=0.2),
    )


class ResBlock(nn.Module):
    """
    Residual Block without BN but with dilation
    """

    def __init__(self, inplanes, out_planes, hid_planes, add_relu=True):
        super(ResBlock, self).__init__()
        self.add_relu = add_relu
        self.conv1 = nn.Sequential(nn.Conv2d(inplanes, hid_planes, 3, 1, 1, 1),
                                   nn.LeakyReLU(inplace=True, negative_slope=0.2))

        self.conv2 = nn.Conv2d(hid_planes, out_planes, 3, 1, 1, 1)
        if add_relu:
            self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += x
        if self.add_relu:
            out = self.relu(out)
        return out





