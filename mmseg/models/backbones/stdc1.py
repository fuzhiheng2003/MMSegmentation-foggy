# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/MichaelFan01/STDC-Seg."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, ModuleList, Sequential

from mmseg.registry import MODELS
from ..utils import resize
from .bisenetv1 import AttentionRefinementModule


class STDCModule(BaseModule):
    """STDCModule.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels before scaling.
        stride (int): The number of stride for the first conv layer.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): The activation config for conv layers.
        num_convs (int): Numbers of conv layers.
        fusion_type (str): Type of fusion operation. Default: 'add'.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 norm_cfg=None,
                 act_cfg=None,
                 num_convs=4,
                 fusion_type='add',
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert num_convs > 1
        assert fusion_type in ['add', 'cat']
        self.stride = stride
        self.with_downsample = True if self.stride == 2 else False
        self.fusion_type = fusion_type

        self.layers = ModuleList()
        conv_0 = ConvModule(
            in_channels, out_channels // 2, kernel_size=1, norm_cfg=norm_cfg)

        if self.with_downsample:
            self.downsample = ConvModule(
                out_channels // 2,
                out_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=out_channels // 2,
                norm_cfg=norm_cfg,
                act_cfg=None)

            if self.fusion_type == 'add':
                self.layers.append(nn.Sequential(conv_0, self.downsample))
                self.skip = Sequential(
                    ConvModule(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        groups=in_channels,
                        norm_cfg=norm_cfg,
                        act_cfg=None),
                    ConvModule(
                        in_channels,
                        out_channels,
                        1,
                        norm_cfg=norm_cfg,
                        act_cfg=None))
            else:
                self.layers.append(conv_0)
                self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.layers.append(conv_0)

        for i in range(1, num_convs):
            out_factor = 2**(i + 1) if i != num_convs - 1 else 2**i
            self.layers.append(
                ConvModule(
                    out_channels // 2**i,
                    out_channels // out_factor,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def forward(self, inputs):
        if self.fusion_type == 'add':
            out = self.forward_add(inputs)
        else:
            out = self.forward_cat(inputs)
        return out

    def forward_add(self, inputs):
        layer_outputs = []
        x = inputs.clone()
        for layer in self.layers:
            x = layer(x)
            layer_outputs.append(x)
        if self.with_downsample:
            inputs = self.skip(inputs)

        return torch.cat(layer_outputs, dim=1) + inputs

    def forward_cat(self, inputs):
        x0 = self.layers[0](inputs)

        layer_outputs = [x0]
        for i, layer in enumerate(self.layers[1:]):
            if i == 0:
                if self.with_downsample:
                    x = layer(self.downsample(x0))
                else:
                    x = layer(x0)
            else:
                x = layer(x)
            layer_outputs.append(x)
        if self.with_downsample:
            layer_outputs[0] = self.skip(x0)
        return torch.cat(layer_outputs, dim=1)


class FeatureFusionModule(BaseModule):
    """Feature Fusion Module. This module is different from FeatureFusionModule
    in BiSeNetV1. It uses two ConvModules in `self.attention` whose inter
    channel number is calculated by given `scale_factor`, while
    FeatureFusionModule in BiSeNetV1 only uses one ConvModule in
    `self.conv_atten`.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        scale_factor (int): The number of channel scale factor.
            Default: 4.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): The activation config for conv layers.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        channels = out_channels // scale_factor
        self.conv0 = ConvModule(
            in_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(
                out_channels,
                channels,
                1,
                norm_cfg=None,
                bias=False,
                act_cfg=act_cfg),
            ConvModule(
                channels,
                out_channels,
                1,
                norm_cfg=None,
                bias=False,
                act_cfg=None), nn.Sigmoid())

    def forward(self, spatial_inputs, context_inputs):
        inputs = torch.cat([spatial_inputs, context_inputs], dim=1)
        x = self.conv0(inputs)
        attn = self.attention(x)
        x_attn = x * attn
        return x_attn + x


@MODELS.register_module()
class STDCNet1(BaseModule):
    """This backbone is the implementation of `Rethinking BiSeNet For Real-time
    Semantic Segmentation <https://arxiv.org/abs/2104.13188>`_.

    Args:
        stdc_type (int): The type of backbone structure,
            `STDCNet1` and`STDCNet2` denotes two main backbones in paper,
            whose FLOPs is 813M and 1446M, respectively.
        in_channels (int): The num of input_channels.
        channels (tuple[int]): The output channels for each stage.
        bottleneck_type (str): The type of STDC Module type, the value must
            be 'add' or 'cat'.
        norm_cfg (dict): Config dict for normalization layer.
        act_cfg (dict): The activation config for conv layers.
        num_convs (int): Numbers of conv layer at each STDC Module.
            Default: 4.
        with_final_conv (bool): Whether add a conv layer at the Module output.
            Default: True.
        pretrained (str, optional): Model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.

    Example:
        >>> import torch
        >>> stdc_type = 'STDCNet1'
        >>> in_channels = 3
        >>> channels = (32, 64, 256, 512, 1024)
        >>> bottleneck_type = 'cat'
        >>> inputs = torch.rand(1, 3, 1024, 2048)
        >>> self = STDCNet(stdc_type, in_channels,
        ...                 channels, bottleneck_type).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 256, 128, 256])
        outputs[1].shape = torch.Size([1, 512, 64, 128])
        outputs[2].shape = torch.Size([1, 1024, 32, 64])
    """

    arch_settings = {
        'STDCNet1': [(2, 1), (2, 1), (2, 1)],
        'STDCNet2': [(2, 1, 1, 1), (2, 1, 1, 1, 1), (2, 1, 1)]
    }

    def __init__(self,
                 stdc_type,
                 in_channels,
                 channels,
                 bottleneck_type,
                 norm_cfg,
                 act_cfg,
                 num_convs=4,
                 with_final_conv=False,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert stdc_type in self.arch_settings, \
            f'invalid structure {stdc_type} for STDCNet.'
        assert bottleneck_type in ['add', 'cat'],\
            f'bottleneck_type must be `add` or `cat`, got {bottleneck_type}'

        assert len(channels) == 5,\
            f'invalid channels length {len(channels)} for STDCNet.'

        self.in_channels = in_channels
        self.channels = channels
        self.stage_strides = self.arch_settings[stdc_type]
        self.prtrained = pretrained
        self.num_convs = num_convs
        self.with_final_conv = with_final_conv
        # self.enhance = DualEnhanceModule()
        self.stages = ModuleList([
            ConvModule(
                self.in_channels,
                self.channels[0],
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                self.channels[0],
                self.channels[1],
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        ])
        # `self.num_shallow_features` is the number of shallow modules in
        # `STDCNet`, which is noted as `Stage1` and `Stage2` in original paper.
        # They are both not used for following modules like Attention
        # Refinement Module and Feature Fusion Module.
        # Thus they would be cut from `outs`. Please refer to Figure 4
        # of original paper for more details.
        self.num_shallow_features = len(self.stages)

        for strides in self.stage_strides:
            idx = len(self.stages) - 1
            self.stages.append(
                self._make_stage(self.channels[idx], self.channels[idx + 1],
                                 strides, norm_cfg, act_cfg, bottleneck_type))
        # After appending, `self.stages` is a ModuleList including several
        # shallow modules and STDCModules.
        # (len(self.stages) ==
        # self.num_shallow_features + len(self.stage_strides))
        if self.with_final_conv:
            self.final_conv = ConvModule(
                self.channels[-1],
                max(1024, self.channels[-1]),
                1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

    def _make_stage(self, in_channels, out_channels, strides, norm_cfg,
                    act_cfg, bottleneck_type):
        layers = []
        for i, stride in enumerate(strides):
            layers.append(
                STDCModule(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    stride,
                    norm_cfg,
                    act_cfg,
                    num_convs=self.num_convs,
                    fusion_type=bottleneck_type))
        return Sequential(*layers)

    def forward(self, x):
        # denormalize_and_save(x)
        # x = self.enhance(x)
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        if self.with_final_conv:
            outs[-1] = self.final_conv(outs[-1])
        outs = outs[self.num_shallow_features:]
        return tuple(outs)


# 在stdc.py中添加以下模块
# @MODELS.register_module()
# 新增模块定义在stdc.py文件中

import torch


# @MODELS.register_module()
class LightEnhanceModule(BaseModule):
    """轻量级图像增强模块，包含亮度调整、残差连接和Sobel边缘增强"""
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        # 亮度调整（深度可分离卷积）
        self.conv_bright = Sequential(
            ConvModule(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                norm_cfg=norm_cfg,
                act_cfg=None),
            ConvModule(
                in_channels,
                out_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU'))
        )
        # 固定Sobel卷积核
        self.sobel_conv = nn.Conv2d(out_channels, 2, kernel_size=3, padding=1, bias=False)
        sobel_kernel = self._get_sobel_kernel(out_channels)
        self.sobel_conv.weight.data = sobel_kernel
        self.sobel_conv.weight.requires_grad = False
        # 边缘特征转换
        self.conv_edge = ConvModule(
            2, out_channels, kernel_size=1, norm_cfg=norm_cfg, act_cfg=dict(type='ReLU'))

    def _get_sobel_kernel(self, channels):
        kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        kernel = torch.stack([kernel_x, kernel_y], dim=0)  # shape(2,3,3)
        kernel = kernel.unsqueeze(1).repeat(1, channels, 1, 1)  # shape(2,channels,3,3)
        return kernel

    def forward(self, x):
        # 亮度调整+残差
        # t = x[0].clone()
        # Sobel边缘增强
        bright = self.conv_bright(x)
        res = x + bright
        edge = self.sobel_conv(res)
        edge = self.conv_edge(edge)
        # res = x + edge
        # denormalize_and_save(res)

        # res = x
        # denormalize_and_save(res)
        # t = torch.clip(t, 0, 255)
        # t = edge
        # t = t.reshape(-1, 1024, 1820)
        # # t = torch.clip(t, 0, 255)
        # toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
        # pic = toPIL(t)
        # pic.save('D:\\random.jpg')
        # denormalize_and_save(res + edge)
        return res + edge

 # 亮度调整+残差
 #        bright = self.conv_bright(x)
 #        res = x + bright
 #        # Sobel边缘增强
 #        edge = self.sobel_conv(res)
 #        edge = self.conv_edge(edge)
 #        return res + edge
class FeatureFusionModule3(BaseModule):
    """Improved Feature Fusion Module with ASPP-based Channel Attention.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        scale_factor (int): The number of channel scale factor. Default: 4.
        norm_cfg (dict): Config dict for normalization layer. Default: BN.
        act_cfg (dict): The activation config. Default: ReLU.
        init_cfg (dict, optional): Initialization config. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        channels = out_channels // scale_factor

        # Main fusion convolution
        self.conv0 = ConvModule(
            in_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # ASPP-based attention module
        self.aspp_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ASPPAttentionBlock(
                in_channels=out_channels,
                mid_channels=channels,
                out_channels=out_channels,
                dilation_rates=(1, 3, 6)),
            nn.Sigmoid()
        )
    def forward(self, spatial_inputs, context_inputs):
        inputs = torch.cat([spatial_inputs, context_inputs], dim=1)
        x = self.conv0(inputs)
        attn = self.aspp_attention(x)
        x_attn = x * attn
        return x_attn + x




# 修改后的FogEnhanceModule类

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model import constant_init, kaiming_init
from mmengine.registry import MODELS
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

class LightEnhanceModule1(BaseModule):
    """轻量级可学习图像增强模块：
    - Sobel 梯度分支 + 深度可分离卷积融合
    输入输出均为 (B, C, H, W)，保持原始通道数不变。
    """
    def __init__(self, in_channels=3, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        # Sobel X 和 Sobel Y 深度可分离卷积
        self.sobel_x = nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=3,
                                 padding=1,
                                 groups=in_channels,
                                 bias=False)
        self.sobel_y = nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=3,
                                 padding=1,
                                 groups=in_channels,
                                 bias=False)
        # 深度可分离卷积融合：先 depthwise 再 pointwise
        self.dwconv = ConvModule(
            in_channels * 2,
            in_channels * 2,
            kernel_size=3,
            padding=1,
            groups=in_channels * 2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.pwconv = ConvModule(
            in_channels * 2,
            in_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        # 初始化 Sobel 权重为经典 Sobel 核
        sobel_kernel = torch.tensor([[1., 0., -1.],
                                     [2., 0., -2.],
                                     [1., 0., -1.]], dtype=torch.float32)
        sobel_xk = sobel_kernel.unsqueeze(0).unsqueeze(0)  # [1,1,3,3]
        sobel_yk = sobel_kernel.t().unsqueeze(0).unsqueeze(0)
        sobel_xk = sobel_xk.repeat(in_channels, 1, 1, 1)
        sobel_yk = sobel_yk.repeat(in_channels, 1, 1, 1)
        with torch.no_grad():
            self.sobel_x.weight.copy_(sobel_xk)
            self.sobel_y.weight.copy_(sobel_yk)

    def forward(self, x):
        # Sobel 梯度
        gx = self.sobel_x(x)
        gy = self.sobel_y(x)


        # grad = torch.hypot(gx, gy)
        grad = torch.sqrt(gx * gx + gy * gy + 1e-6)
        # 拼接原图 + 梯度


        # denormalize_and_save(grad)
        z = torch.cat([x, grad], dim=1)
        # z = torch.cat([x, x], dim=1)

        # 深度可分离卷积融合通道信息
        z = self.dwconv(z)
        out = self.pwconv(z)
        denormalize_and_save(out)
        return out


from PIL import Image
# 存储可视化特征图，
def denormalize_and_save(input_tensor, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
                         save_path="D:\\random.jpg"):
    """
    对均值化/标准化的图像张量进行还原并保存（针对STDCNet输入格式）
    :param input_tensor: 输入张量，形状为 (B, 3, H, W) 或 (3, H, W)
    :param mean: 预处理时使用的均值（RGB顺序，范围0-255）
    :param std: 预处理时使用的标准差（RGB顺序，范围0-255）
    :param save_path: 还原后图像的保存路径
    """
    # 处理输入维度：确保包含batch维度（B=1）
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)  # 转为 (1, 3, H, W)
    elif input_tensor.dim() != 4:
        raise ValueError("Input tensor must be 3D (C,H,W) or 4D (B,C,H,W)")

    # 转换至CPU并分离梯度（避免计算图污染）
    input_tensor = input_tensor.detach().cpu()

    # 逆标准化：image = x * std + mean（注意通道顺序为RGB）
    mean_tensor = torch.tensor(mean).view(1, 3, 1, 1)  # 形状 (1,3,1,1) 以适配广播
    std_tensor = torch.tensor(std).view(1, 3, 1, 1)
    restored_tensor = input_tensor * std_tensor + mean_tensor  # 逆操作核心步骤

    # 转换为HWC格式（PIL要求）并裁剪至0-255
    restored_np = restored_tensor.squeeze(0).permute(1, 2, 0).numpy()  # 形状 (H,W,C)
    restored_np = np.clip(restored_np, 0, 255).astype(np.uint8)  # 确保像素值合法

    # 保存图像
    Image.fromarray(restored_np).save(save_path)
    print(f"Successfully restored and saved image to {save_path}")

class ASPPAttentionBlock(nn.Module):
    """Lightweight ASPP block for attention generation."""

    def __init__(self, in_channels, mid_channels, out_channels, dilation_rates=(1, 3, 6)):
        super().__init__()
        self.branches = ModuleList()
        for rate in dilation_rates:
            self.branches.append(
                Sequential(
                    ConvModule(in_channels, mid_channels, 3,
                               padding=rate, dilation=rate,
                               groups=mid_channels,  # Depthwise conv
                               norm_cfg=None,
                               act_cfg=dict(type='ReLU')),
                    ConvModule(mid_channels, mid_channels, 1,
                               norm_cfg=None,
                               act_cfg=None)
                )
            )
        self.global_branch = Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(in_channels, mid_channels, 1,
                       norm_cfg=None,
                       act_cfg=dict(type='ReLU'))
        )
        self.fusion = ConvModule(
            mid_channels * (len(dilation_rates) + 1), out_channels,
            1, norm_cfg=None, act_cfg=None)

    def forward(self, x):
        bs, c, _, _ = x.size()
        out = []
        for branch in self.branches:
            out.append(branch(x))
        # Global feature
        global_feat = self.global_branch(x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='nearest')
        out.append(global_feat)

        # Concatenate and fuse
        out = torch.cat(out, dim=1)
        return self.fusion(out)

class LightEnhanceModule2(BaseModule):
    """固定预设Sobel核 + 可学习卷积的轻量级图像增强模块"""

    def __init__(self,  in_channels=3, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels

        # 注册不可学习的 sobel_x 和 sobel_y 权重（缓冲区）
        sobel_x_weight = torch.tensor(
            [[[[0.1749, 0.0064, -0.1703],
               [0.2215, -0.0737, -0.3559],
               [0.1902, 0.0006, -0.1701]]],

             [[[0.1825, 0.0204, -0.1109],
               [0.2869, -0.0154, -0.2818],
               [0.1521, -0.0180, -0.1749]]],

             [[[0.1804, 0.0282, -0.1110],
               [0.2845, -0.0023, -0.2779],
               [0.1731, 0.0068, -0.1568]]]
             ], dtype=torch.float32)  # shape [3,1,3,3]

        sobel_y_weight = torch.tensor(
            [[[[0.1072, 0.2545, 0.1688],
               [-0.0025, -0.0188, 0.0041],
               [-0.1178, -0.2750, -0.1358]]],

             [[[0.1236, 0.2706, 0.0766],
               [0.0135, 0.0495, -0.0224],
               [-0.1313, -0.2233, -0.1778]]],

             [[[0.1254, 0.2443, 0.1174],
               [-0.0158, -0.0327, -0.0388],
               [-0.1333, -0.2854, -0.1687]]]
             ], dtype=torch.float32)  # shape [3,1,3,3]

        # 注册为 buffer，不更新
        self.register_buffer('sobel_kernel_x', sobel_x_weight)
        self.register_buffer('sobel_kernel_y', sobel_y_weight)

        # 深度可分离卷积
        self.dwconv = ConvModule(
            in_channels * 2,
            in_channels * 2,
            kernel_size=3,
            padding=1,
            groups=in_channels * 2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.pwconv = ConvModule(
            in_channels * 2,
            in_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        # 将提供的 sobel_kernel 按通道数复制扩展
        sobel_x = self.sobel_kernel_x.repeat(self.in_channels, 1, 1, 1)
        sobel_y = self.sobel_kernel_y.repeat(self.in_channels, 1, 1, 1)

        # 分组卷积（每个通道独立应用）
        gx = F.conv2d(x, sobel_x, bias=None, padding=1, groups=self.in_channels)
        gy = F.conv2d(x, sobel_y, bias=None, padding=1, groups=self.in_channels)

        grad = torch.sqrt(gx * gx + gy * gy + 1e-6)
        denormalize_and_save(grad)
        # 拼接原图和梯度特征
        z = torch.cat([x, grad], dim=1)

        # 深度可分离卷积融合
        z = self.dwconv(z)
        out = self.pwconv(z)
        return out

class LightEnhanceModule3(BaseModule):
    """轻量级固定Sobel+可学习卷积的图像增强模块：
    - Sobel梯度（固定参数）+ 深度可分离卷积融合
    - 输入输出均为 (B, C, H, W)，保持通道数不变。
    """
    def __init__(self, in_channels=3, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        # 创建 Sobel X 和 Sobel Y 的固定卷积
        self.register_buffer('sobel_kernel_x', torch.tensor([[1., 0., -1.],
                                                              [2., 0., -2.],
                                                              [1., 0., -1.]]).unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_kernel_y', torch.tensor([[1., 2., 1.],
                                                              [0., 0., 0.],
                                                              [-1., -2., -1.]]).unsqueeze(0).unsqueeze(0))

        self.in_channels = in_channels

        # 深度可分离卷积融合
        self.dwconv = ConvModule(
            in_channels * 2,
            in_channels * 2,
            kernel_size=3,
            padding=1,
            groups=in_channels * 2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.pwconv = ConvModule(
            in_channels * 2,
            in_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        # 对每个输入通道应用相同的 sobel_x, sobel_y
        sobel_kernel_x = self.sobel_kernel_x.repeat(self.in_channels, 1, 1, 1)
        sobel_kernel_y = self.sobel_kernel_y.repeat(self.in_channels, 1, 1, 1)

        gx = F.conv2d(x, sobel_kernel_x, bias=None, padding=1, groups=self.in_channels)
        gy = F.conv2d(x, sobel_kernel_y, bias=None, padding=1, groups=self.in_channels)
        grad = torch.sqrt(gx * gx + gy * gy + 1e-6)
        # grad = torch.hypot(gx, gy)
        # denormalize_and_save(grad)
        # 拼接原图和梯度图
        z = torch.cat([x, grad], dim=1)

        # 深度可分离卷积融合
        z = self.dwconv(z)
        out = self.pwconv(z)
        # denormalize_and_save(out)
        return out

@MODELS.register_module()
class STDCContextPathNet1(BaseModule):
    """STDCNet with Context Path. The `outs` below is a list of three feature
    maps from deep to shallow, whose height and width is from small to big,
    respectively. The biggest feature map of `outs` is outputted for
    `STDCHead`, where Detail Loss would be calculated by Detail Ground-truth.
    The other two feature maps are used for Attention Refinement Module,
    respectively. Besides, the biggest feature map of `outs` and the last
    output of Attention Refinement Module are concatenated for Feature Fusion
    Module. Then, this fusion feature map `feat_fuse` would be outputted for
    `decode_head`. More details please refer to Figure 4 of original paper.

    Args:
        backbone_cfg (dict): Config dict for stdc backbone.
        last_in_channels (tuple(int)), The number of channels of last
            two feature maps from stdc backbone. Default: (1024, 512).
        out_channels (int): The channels of output feature maps.
            Default: 128.
        ffm_cfg (dict): Config dict for Feature Fusion Module. Default:
            `dict(in_channels=512, out_channels=256, scale_factor=4)`.
        upsample_mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``.
        align_corners (str): align_corners argument of F.interpolate. It
            must be `None` if upsample_mode is ``'nearest'``. Default: None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.

    Return:
        outputs (tuple): The tuple of list of output feature map for
            auxiliary heads and decoder head.
    """

    def __init__(self,
                 backbone_cfg,
                 last_in_channels=(1024, 512),
                 out_channels=128,
                 # enhance_cfg=dict(type='DualPathEnhance', out_channels=3),
                 ffm_cfg=dict(
                     in_channels=512, out_channels=256, scale_factor=4),
                 upsample_mode='nearest',
                 align_corners=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        # self.enhance = LearnableFogEnhance()
        # self.enhance = MODELS.build(enhance_cfg)
        # self.enhance = FogEnhanceModule()

        # self.enhance = LightEnhanceModule()
        # self.enhance = LearnableSobelEnhancer(in_channels=3)
        # self.enhance =FogAdaptiveEnhance()
        # self.enhance =FogEnhanceModule1(in_channels=3)

        # self.enhance =FogEnhanceModule(in_channels=3, norm_cfg=norm_cfg,)
        # 图像增强模块1为可学习权重策略，3为固定权重策略
        self.enhance = LightEnhanceModule1()
        # self.enhance = LightEnhanceModule1()
        # self.enhance =ImageEnhancementModule() # 待测
        self.ffm = FeatureFusionModule(**ffm_cfg)
        # 3为替换金字塔结构特征融合模块
        # self.ffm = FeatureFusionModule3(**ffm_cfg)

        # self.enhance =ImageEnhanceModule(in_channels=3)
        self.backbone = MODELS.build(backbone_cfg)
        self.arms = ModuleList()
        self.convs = ModuleList()
        for channels in last_in_channels:
            self.arms.append(AttentionRefinementModule(channels, out_channels))
            self.convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    norm_cfg=norm_cfg))
        self.conv_avg = ConvModule(
            last_in_channels[0], out_channels, 1, norm_cfg=norm_cfg)



        self.upsample_mode = upsample_mode
        self.align_corners = align_corners

    def forward(self, x):

        # 边缘细节增强
        x = self.enhance(x)  # 先进行图像增强

        outs = list(self.backbone(x))
        avg = F.adaptive_avg_pool2d(outs[-1], 1)
        avg_feat = self.conv_avg(avg)

        feature_up = resize(
            avg_feat,
            size=outs[-1].shape[2:],
            mode=self.upsample_mode,
            align_corners=self.align_corners)
        arms_out = []
        for i in range(len(self.arms)):
            x_arm = self.arms[i](outs[len(outs) - 1 - i]) + feature_up
            feature_up = resize(
                x_arm,
                size=outs[len(outs) - 1 - i - 1].shape[2:],
                mode=self.upsample_mode,
                align_corners=self.align_corners)
            feature_up = self.convs[i](feature_up)
            arms_out.append(feature_up)

        feat_fuse = self.ffm(outs[0], arms_out[1])

        # The `outputs` has four feature maps.
        # `outs[0]` is outputted for `STDCHead` auxiliary head.
        # Two feature maps of `arms_out` are outputted for auxiliary head.
        # `feat_fuse` is outputted for decoder head.
        outputs = [outs[0]] + list(arms_out) + [feat_fuse]
        return tuple(outputs)
