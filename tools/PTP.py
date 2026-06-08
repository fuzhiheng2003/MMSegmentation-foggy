import torch
import torchvision.models as models
import torch
import numpy as np

# 添加 numpy 支持
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

# 重写 torch.load，自动关闭 weights_only
_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = patched_torch_load
#
# checkpoint = torch.load('SeaFormer_B_cls_76.4.pth.tar')
import torch

# 加载 .pth.tar 文件
checkpoint = torch.load("SeaFormer_B_cls_76.4.pth.tar", map_location='cpu')  # 确保加载到CPU

# 提取模型权重 (假设 key 是 'state_dict')
model_state_dict = checkpoint['state_dict']

# 直接保存为 .pth 文件
torch.save(model_state_dict, "SeaFormer_B.pth.pth")
