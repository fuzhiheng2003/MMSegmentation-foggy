_base_ = './bisenetv1_r50-d32_4xb4-80k_cityscapes-512x1024.py'
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        backbone_cfg=dict(
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://resnet50_v1c'))))
