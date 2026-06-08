# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/MichaelFan01/STDC-Seg."""
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


class FeatureFusionModule1(BaseModule):
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


class ASPPChannelAttention(BaseModule):
    """ASPP-based Channel Attention Module.

    Args:
        in_channels (int): Input channels.
        scale_factor (int): Scale factor to reduce channels.
        norm_cfg (dict): Normalization config.
        act_cfg (dict): Activation config.
    """

    def __init__(self, in_channels, scale_factor=4, norm_cfg=None, act_cfg=dict(type='ReLU')):
        super().__init__()
        self.in_channels = in_channels
        self.scale_factor = scale_factor
        self.reduced_channels = in_channels // scale_factor

        # Define multi-branch convolutions with different dilations
        self.branches = ModuleList([
            # Branch 1: 1x1 convolution
            ConvModule(
                in_channels,
                self.reduced_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                bias=False),
            # Branch 2: 1x1 conv + 3x3 dilated conv (d=2)
            Sequential(
                ConvModule(
                    in_channels,
                    self.reduced_channels,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=False),
                ConvModule(
                    self.reduced_channels,
                    self.reduced_channels,
                    kernel_size=3,
                    padding=2,
                    dilation=2,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=False)),
            # Branch 3: 1x1 conv + 3x3 dilated conv (d=4)
            Sequential(
                ConvModule(
                    in_channels,
                    self.reduced_channels,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=False),
                ConvModule(
                    self.reduced_channels,
                    self.reduced_channels,
                    kernel_size=3,
                    padding=4,
                    dilation=4,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=False)),
            # Branch 4: Identity mapping with 1x1 conv
            ConvModule(
                in_channels,
                self.reduced_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=None,
                bias=False)
        ])

        # Fusion convolution
        self.fusion = ConvModule(
            self.reduced_channels * 4,
            in_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
            bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_pool = F.adaptive_avg_pool2d(x, 1)
        branch_outs = [branch(x_pool) for branch in self.branches]
        out = torch.cat(branch_outs, dim=1)
        out = self.fusion(out)
        return self.sigmoid(out)



@MODELS.register_module()
class STDCNet3(BaseModule):
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
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        if self.with_final_conv:
            outs[-1] = self.final_conv(outs[-1])
        outs = outs[self.num_shallow_features:]
        return tuple(outs)


class FeatureFusionModule2(BaseModule):
    """Enhanced Feature Fusion Module with Dual Attention and Multi-scale Context.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        scale_factor (int): The number of channel scale factor. Default: 4.
        norm_cfg (dict): Config dict for normalization layer. Default: dict(type='BN').
        act_cfg (dict): The activation config for conv layers. Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        # Channel Attention Branch
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                out_channels,
                out_channels // scale_factor,
                1,
                norm_cfg=None,
                bias=False,
                act_cfg=act_cfg),
            ConvModule(
                out_channels // scale_factor,
                out_channels,
                1,
                norm_cfg=None,
                bias=False,
                act_cfg=None),
            nn.Sigmoid()
        )

        # Spatial Attention Branch
        self.spatial_attention = nn.Sequential(
            ConvModule(
                2, 1, 7, padding=3,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='Sigmoid')),
        )

        # Multi-scale Context
        self.dilation_convs = ModuleList([
            ConvModule(
                out_channels, out_channels // 4, 3,
                dilation=d, padding=d,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            for d in [1, 2, 4]
        ])

        # Fusion Conv
        self.fusion_conv = ConvModule(
            out_channels + out_channels // 4 * 3,  # 多尺度特征拼接
            out_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        # Projection
        self.conv0 = ConvModule(
            in_channels, out_channels, 1,
            norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, spatial_inputs, context_inputs):
        # Feature Concatenation
        inputs = torch.cat([spatial_inputs, context_inputs], dim=1)
        x = self.conv0(inputs)

        # Multi-scale Context Extraction
        ms_features = [x]
        for conv in self.dilation_convs:
            ms_features.append(conv(x))
        x_ms = torch.cat(ms_features, dim=1)
        x_ms = self.fusion_conv(x_ms)

        # Dual Attention
        ## Channel Attention
        channel_attn = self.channel_attention(x_ms)
        x_ch = x_ms * channel_attn

        ## Spatial Attention
        spatial_avg = torch.mean(x_ch, dim=1, keepdim=True)
        spatial_max, _ = torch.max(x_ch, dim=1, keepdim=True)
        spatial_attn = self.spatial_attention(
            torch.cat([spatial_avg, spatial_max], dim=1))
        x_final = x_ch * spatial_attn

        return x_final + x  # Residual Connection


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


class FeatureFusionModule4(BaseModule):
    """Improved Feature Fusion Module with ASPP-enhanced Channel Attention.

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

        # Initial fusion convolution
        self.conv0 = ConvModule(
            in_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # ASPP-enhanced attention module
        self.aspp_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(
                out_channels,
                channels,
                1,
                norm_cfg=None,
                bias=False,
                act_cfg=act_cfg),

            # ASPP parallel branches
            nn.ModuleList([
                nn.Sequential(
                    ConvModule(
                        channels,
                        channels,
                        3,
                        padding=d,
                        dilation=d,
                        groups=channels,
                        norm_cfg=None,
                        bias=False,
                        act_cfg=act_cfg)
                ) for d in [1, 3, 6]
            ]),

            # Feature aggregation
            ConvModule(
                channels * 3,  # 3 branches
                out_channels,
                1,
                norm_cfg=None,
                bias=False,
                act_cfg=None),
            nn.Sigmoid()
        )

    def forward(self, spatial_inputs, context_inputs):
        # Feature concatenation
        fused = torch.cat([spatial_inputs, context_inputs], dim=1)
        x = self.conv0(fused)

        # ASPP attention mechanism
        attn = self.aspp_attention[0](x)  # Global pooling
        attn = self.aspp_attention[1](attn)  # Channel reduction

        # Process through ASPP branches
        aspp_outs = []
        for branch in self.aspp_attention[2]:
            aspp_outs.append(branch(attn))
        attn = torch.cat(aspp_outs, dim=1)

        # Final aggregation
        attn = self.aspp_attention[3](attn)
        attn = self.aspp_attention[4](attn)

        return x * attn + x


# 新增模块代码

# @MODELS.register_module()
# class LightEnhanceModule(BaseModule):
#     """Lightweight Image Enhancement Module with depthwise separable conv and
#     Sobel filtering for edge enhancement.
#
#     Args:
#         in_channels (int): Number of input channels. Default: 3
#         out_channels (int): Number of output channels. Default: 64
#         norm_cfg (dict): Config dict for normalization layer. Default: None
#         act_cfg (dict): Activation config for conv layers. Default: ReLU
#         init_cfg (dict, optional): Initialization config dict. Default: None
#     """
#
#     def __init__(self,
#                  in_channels=3,
#                  out_channels=3,
#                  norm_cfg=None,
#                  act_cfg=dict(type='ReLU'),
#                  init_cfg=None):
#         super().__init__(init_cfg=init_cfg)
#
#         # Sobel filter kernels (non-trainable)
#         self.sobel_conv = nn.Conv2d(in_channels, 2, 3, padding=1, bias=False)
#         sobel_kernel = torch.tensor([
#             [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
#              [[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
#               ]]], dtype = torch.float32)
#         self.sobel_conv.weight.data = sobel_kernel.repeat(in_channels, 1, 1, 1)
#         self.sobel_conv.weight.requires_grad = False
#
#         # Depthwise separable convolution branch
#         self.dw_conv = ConvModule(
#             in_channels,
#             out_channels,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             groups=in_channels,  # Depthwise
#             norm_cfg=norm_cfg,
#             act_cfg=act_cfg)
#         self.pw_conv = ConvModule(
#             out_channels,
#             out_channels,
#             kernel_size=1,
#             norm_cfg=norm_cfg,
#             act_cfg=act_cfg)
#
#         # Edge processing branch
#         self.edge_conv = ConvModule(
#             2,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             norm_cfg=norm_cfg,
#             act_cfg=act_cfg)
#
#         # Fusion and residual
#         self.fusion_conv = ConvModule(
#             out_channels + out_channels,
#             out_channels,
#             kernel_size=1,
#             norm_cfg=norm_cfg,
#             act_cfg=None)
#
#         self.res_conv = ConvModule(
#             in_channels,
#             out_channels,
#             kernel_size=1,
#             norm_cfg=norm_cfg,
#             act_cfg=None)
#
#     def forward(self, x):
#         # Original input
#         identity = self.res_conv(x)
#
#         # Sobel edge detection
#         edge = self.sobel_conv(x)
#         edge_feat = self.edge_conv(edge)
#
#         # Depthwise separable feature extraction
#         dw_feat = self.dw_conv(x)
#         pw_feat = self.pw_conv(dw_feat)
#
#         # Feature fusion
#         fused = torch.cat([pw_feat, edge_feat], dim=1)
#         fused = self.fusion_conv(fused)
#
#         # Residual connection
#         return F.relu(fused + identity)

@MODELS.register_module()
class STDCContextPathNet3(BaseModule):
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
                 ffm_cfg=dict(
                     in_channels=512, out_channels=256, scale_factor=4),
                 upsample_mode='nearest',
                 align_corners=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        # self.enhance = FogEnhanceModule()

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
        # 59.44
        self.ffm = FeatureFusionModule3(**ffm_cfg)

        self.upsample_mode = upsample_mode
        self.align_corners = align_corners

    def forward(self, x):
        # 亮度均衡
        # x = self.enhance(x)

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
