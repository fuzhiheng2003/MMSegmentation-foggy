_base_ = './stdc1_b8x80k_acdc_fog-512x1024.py'
model = dict(backbone=dict(backbone_cfg=dict(stdc_type='STDCNet2')))
