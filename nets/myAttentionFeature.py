import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class myRawFeature(nn.Module):
    def __init__(self, layers=[3, 4, 6, 3]):
        super(myRawFeature, self).__init__()
        self.inplanes = 64
        self.firstConv = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # H/2
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(True))
        self.layer1 = self._make_layer(Bottleneck, 64, layers[0])  # block, planes, blocks, stride=1, dilation=1
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)  # H/4
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(Bottleneck, 512, layers[3], stride=1, dilation=4)  # H/4

        # self.attentionBlocks = myAttentionBlock(in_channels=512, key_channels=256, value_channels=512)

        self.parameter_initialization()  # 权重初始化

    def parameter_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.firstConv(x)  # H/2
        x = self.layer1(x)
        x = self.layer2(x)  # H/4
        x = self.layer3(x)  # H/4

        return x

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride,
                            dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                stride=1, dilation=dilation))
        return nn.Sequential(*layers)


class PixelAttentionBlock_(nn.Module):
    def __init__(self, in_channels, key_channels):
        super(PixelAttentionBlock_, self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.f_key = nn.Sequential(nn.Conv2d(in_channels, key_channels, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm2d(key_channels),
                                   nn.ReLU(True))
        self.parameter_initialization()
        self.f_query = self.f_key  # 使用同一个线性映射，生成query和key

    def parameter_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward_att(self, x, y):
        b = x.size(0)
        query = self.f_query(x).view(b, self.key_channels, -1).permute(0, 2, 1)  # [B, HW, C]
        key = self.f_key(y).view(b, self.key_channels, -1)  # [B, C ,HW]
        sim_map = torch.matmul(query, key)   # [B, HW, HW]
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        return sim_map

    def forward(self, x):
        raise NotImplementedError


class AttentionBlock_(PixelAttentionBlock_):
    def __init__(self, in_channels, key_channels, value_channels):
        super(AttentionBlock_, self).__init__(in_channels, key_channels)
        self.value_channels = value_channels

        self.f_value = nn.Sequential(
            nn.Conv2d(in_channels, value_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(value_channels, value_channels, kernel_size=1, stride=1),
            nn.ReLU(inplace=True))

        self.parameter_initialization()

    def parameter_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, y):
        assert x.size() == y.size()

        b, c, h, w = x.size()
        sim_map = self.forward_att(x, y)

        value = self.f_value(y).view(b, self.value_channels, -1).permute(0, 2, 1)  # [B, C, H, W] ->[B, HW, C]
        context = torch.matmul(sim_map, value).permute(0, 2, 1).contiguous()  # [B, HW, HW] * [B, HW, C] = [B, HW, C]->[B, C, HW]

        context = context.view(b, self.value_channels, h, w)

        # return context, sim_map
        return context


class myAttentionBlock(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, layer_names=None):
        super(myAttentionBlock, self).__init__()
        if layer_names is None:
            layer_names = ["self", "cross"]
        self.names = layer_names

        self.layers = nn.ModuleList([AttentionBlock_(in_channels, key_channels, value_channels)
                                     for _ in range(len(layer_names))])

    def forward(self, x, y):
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                delta_x, delta_y = layer(x, y), layer(y, x)
            elif name == 'self':
                delta_x, delta_y = layer(x, x), layer(y, y)
            else:
                raise Exception("Error, Please specify: cross OR self Attention!")

            x, y = (x + delta_x), (y + delta_y)

        return x, y


class multiScaleAttention(nn.Module):
    def __init__(self, in_channels, scale_num=3, layer_names=None):
        super(multiScaleAttention, self).__init__()

        self.scale_num = scale_num

        if layer_names is None:
            layer_names = ["self", "cross"]

        # self.feature_pyramid_network = feature_pyramid_network时：in_channels=[128, 128, 128]
        # self.feature_pyramid_network ！= feature_pyramid_network时：in_channels=[32, 64, 128]
        attentionList = nn.ModuleList()
        attentionList.append(myAttentionBlock(in_channels=in_channels[0], key_channels=16, value_channels=in_channels[0], layer_names=layer_names))
        attentionList.append(myAttentionBlock(in_channels=in_channels[1], key_channels=32, value_channels=in_channels[1], layer_names=layer_names))
        attentionList.append(myAttentionBlock(in_channels=in_channels[2], key_channels=64, value_channels=in_channels[2], layer_names=layer_names))

        self.attentionLayers = attentionList

        shrinkChannel = in_channels[0] * 2
        self.shrinkHW = nn.Sequential(nn.Conv2d(in_channels[0], shrinkChannel, kernel_size=3, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(shrinkChannel),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(shrinkChannel, in_channels[0], kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(in_channels[0]),
                                  nn.LeakyReLU(0.2, inplace=True))
        self.fuse = nn.Conv2d(in_channels=in_channels[0] + in_channels[0], out_channels=in_channels[0], kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, lft_feature, rht_feature):
        assert len(lft_feature) == len(rht_feature) == 3  # 高分辨率->低分辨率
        # 为了节约显存和计算量，先将最高分辨率的左右图，尺寸缩小一半，再做Attention
        shnk_lft, shnk_rht = self.attentionLayers[0](self.shrinkHW(lft_feature[0]), self.shrinkHW(rht_feature[0]))

        expd_lft, expd_rht = \
            F.upsample(shnk_lft, scale_factor=2, mode='bilinear', align_corners=True), \
            F.upsample(shnk_rht, scale_factor=2, mode='bilinear', align_corners=True)
        lft_feature[0], rht_feature[0] = \
             self.fuse(torch.cat((lft_feature[0], expd_lft), dim=1)), self.fuse(torch.cat((rht_feature[0], expd_rht), dim=1))

        # for i in range(self.scale_num):
        for i in [1, 2]:
            lft_feature[i], rht_feature[i] = self.attentionLayers[i](lft_feature[i], rht_feature[i])

        return lft_feature, rht_feature


class PAMAttentionBlock_(nn.Module):
    def __init__(self, in_channels, channel_shrink, key_channels, value_channels):
        super(PAMAttentionBlock_, self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        srnk_cnls = in_channels // channel_shrink
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, srnk_cnls, 1, 1, 0, bias=True),
            nn.BatchNorm2d(srnk_cnls),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(srnk_cnls, srnk_cnls, 3, 1, 1, bias=True),
            nn.BatchNorm2d(srnk_cnls),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(srnk_cnls, srnk_cnls, 3, 1, 1, bias=True),
            nn.BatchNorm2d(srnk_cnls),
            nn.LeakyReLU(0.1, inplace=True))  # TODO: 这里可以考虑再增加一两层卷积

        self.query = nn.Sequential(
            nn.Conv2d(srnk_cnls, key_channels, 1, 1, 0, bias=True),
            nn.BatchNorm2d(key_channels))

        self.key = nn.Sequential(
            nn.Conv2d(srnk_cnls, key_channels, 1, 1, 0, bias=True),
            nn.BatchNorm2d(key_channels))

        self.value = nn.Sequential(
            nn.Conv2d(srnk_cnls, value_channels, 1, 1, 0, bias=True),
            nn.BatchNorm2d(value_channels))
    #     self.parameter_initialization()
    #
    # def parameter_initialization(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias.data, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()

    def forward(self, x, y):
        """
        :param x:      features from the left image  (B * C * H * W)
        :param y:     features from the right image (B * C * H * W)
        """
        assert x.size() == y.size()
        b, c, h, w = x.shape

        fea_x = self.head(x)
        fea_y = self.head(y)

        # 行内Attention
        Q = self.query(fea_x).permute(0, 2, 3, 1).contiguous()       # B * H * W * C
        K = self.key(fea_y).permute(0, 2, 1, 3) .contiguous()        # B * H * C * W
        V = self.value(fea_y).permute(0, 2, 3, 1).contiguous()       # B * H * W * C

        attRow_y2x = torch.matmul(Q, K) * (self.key_channels**-.5)  # M(B->A): B * H * W * W     # scale the matching cost
        # attCol_y2x = attCol_y2x + cost[0]
        attRow_y2x = F.softmax(attRow_y2x, dim=-1)
        ctxtRow_x = torch.matmul(attRow_y2x, V).permute(0, 3, 1, 2).contiguous()  # [B, H, W, W] * [B, H, W, C] = [B, H, W, C] -> [B, C, H, W]

        # 列内Attention
        Q = Q.permute(0, 2, 1, 3).contiguous()       # B * H * W * C -> B * W * H * C
        K = K.permute(0, 3, 2, 1).contiguous()        # B * H * C * W -> B * W * C * H
        V = V.permute(0, 2, 1, 3).contiguous()       # B * H * W * C -> B * W * H * C

        attCol_y2x = torch.matmul(Q, K) * (self.key_channels**-.5)  # [B, W, H, C] * [B, W, C, H] -> [B, W, H, H]
        attCol_y2x = F.softmax(attCol_y2x, dim=-1)
        ctxtCol_x = torch.matmul(attCol_y2x, V).permute(0, 3, 2, 1).contiguous()  # [B, W, H, H] * [B, W, H, C] -> [B, W, H, C] -> [B, C, H, W]

        # return context, sim_map
        # return torch.cat([ctxtRow_x, ctxtCol_x], dim=1)
        return ctxtRow_x + ctxtCol_x


class myFuseBlock_(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(myFuseBlock_, self).__init__()

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        return self.fuse(x)


class myPAMAttentionBlock(nn.Module):
    def __init__(self, in_channels, channel_shrink, key_channels, value_channels, layer_names=None):
        super(myPAMAttentionBlock, self).__init__()
        assert layer_names is not None

        self.names = layer_names

        self.layers = nn.ModuleList([PAMAttentionBlock_(in_channels, channel_shrink, key_channels, value_channels)
                                     for _ in range(len(layer_names))])

        # self.fuse_x = nn.ModuleList([myFuseBlock_(value_channels * 2, value_channels)
        #                             for _ in range(len(layer_names))])
        # self.fuse_y = nn.ModuleList([myFuseBlock_(value_channels * 2, value_channels)
        #                             for _ in range(len(layer_names))])

    def forward(self, x, y):
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                ctxt_x, ctxt_y = layer(x, y), layer(y, x)
            elif name == 'self':
                ctxt_x, ctxt_y = layer(x, x), layer(y, y)
            else:
                raise Exception("Error, Please specify: cross OR self Attention!")

            x, y = x + ctxt_x, y + ctxt_y

        return x, y

    # def forward(self, x, y):
    #     for layer, name, x_fuse, y_fuse in zip(self.layers, self.names, self.fuse_x, self.fuse_y):
    #         if name == 'cross':
    #             ctxt_x, ctxt_y = layer(x, y), layer(y, x)
    #         elif name == 'self':
    #             ctxt_x, ctxt_y = layer(x, x), layer(y, y)
    #         else:
    #             raise Exception("Error, Please specify: cross OR self Attention!")
    #
    #         x, y = x_fuse(torch.cat([x, ctxt_x], dim=1)), y_fuse(torch.cat([y, ctxt_y], dim=1))
    #
    #     return x, y


class multiScalePAMAttention(nn.Module):
    def __init__(self, in_channels, feature_pyramid_network, scale_num=3, layer_names=None):
        super(multiScalePAMAttention, self).__init__()

        self.scale_num = scale_num

        if layer_names is None:
            layer_names = ["self", "cross"]

        # self.feature_pyramid_network = feature_pyramid_network时：in_channels=[128, 128, 128]
        # self.feature_pyramid_network ！= feature_pyramid_network时：in_channels=[32, 64, 128]
        cnl_s = [4, 2, 1] if feature_pyramid_network else [1, 1, 1]  # 用于压缩通道数
        lists = nn.ModuleList()
        lists.append(
            myPAMAttentionBlock(in_channels=in_channels[0], channel_shrink=cnl_s[0], key_channels=16, value_channels=in_channels[0],
                             layer_names=layer_names))
        lists.append(
            myPAMAttentionBlock(in_channels=in_channels[1], channel_shrink=cnl_s[1], key_channels=32, value_channels=in_channels[1],
                             layer_names=layer_names))
        lists.append(
            myPAMAttentionBlock(in_channels=in_channels[2], channel_shrink=cnl_s[2], key_channels=64, value_channels=in_channels[2],
                             layer_names=layer_names))

        self.pamAttentionLayers = lists

    def forward(self, lft_feature, rht_feature):
        assert len(lft_feature) == len(rht_feature) == 3  # 高分辨率->低分辨率

        # for i in [0, 1, 2]:
        for i in range(self.scale_num):
            lft_feature[i], rht_feature[i] = self.pamAttentionLayers[i](lft_feature[i], rht_feature[i])

        return lft_feature, rht_feature