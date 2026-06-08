import os
import torch
import cv2
import argparse
import numpy as np
from pprint import pprint
from tqdm import tqdm
from mmseg.apis import init_model, inference_model

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

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# 测试图像所在文件夹
# IMAGE_FILE_PATH = r"data/acdc_fog/rgb_anon/fog/val"
# IMAGE_FILE_PATH = r"data/foggy_driving/leftImg8bit/test"

IMAGE_FILE_PATH = r"data/cityscapes/leftImg8bit/val"
# 模型训练结果的config配置文件路径
# stdc
CONFIG = r'configs/stdc/stdc2_in1k-pre_b8x80k_acdc_fog-512x1024.py'
CHECKPOINT = r'work_dirs/stdc2_in1k-pre_b8x80k_acdc_fog-512x1024_06298/best_mIoU_iter_76000.pth'










# 模型推理测试结果的保存路径，每个模型的推理结果都保存在`{save_dir}/{模型config同名文件夹}`下，如文末图片所示。
SAVE_DIR = r"work_dir\infer_results"


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize CAM')
    parser.add_argument('--img', default=IMAGE_FILE_PATH, help='Image file')
    parser.add_argument('--config', default=CONFIG, help='Config file')
    parser.add_argument('--checkpoint', default=CHECKPOINT, help='Checkpoint file')
    parser.add_argument('--device', default=DEVICE, help='device')
    parser.add_argument('--save_dir', default=SAVE_DIR, help='save_dir')

    args = parser.parse_args()
    return args

def get_filelist(dir):
    Filelist = []
    for home, dirs, files in os.walk(dir):
        for filename in files:
            # 文件名列表，包含完整路径
            file = os.path.join(home, filename)
            # # 文件名列表，只包含文件名
            print(file)
            Filelist.append(file)

    return Filelist

palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170,
                                                    30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]

def main():
    args = parse_args()

    model_mmseg = init_model(args.config, args.checkpoint, device=args.device)

    for imgs in tqdm(get_filelist(args.img)):
        result = inference_model(model_mmseg, imgs)
        pred_mask = result.pred_sem_seg.data.squeeze(0).detach().cpu().numpy().astype(np.uint8)
        # pred_mask[pred_mask == 1] = 255
        # print(type(pred_mask))
        seg = np.array(pred_mask)
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color

        color_seg = color_seg[..., ::-1]  # convert to BGR

        save_path = os.path.join(args.save_dir, f"{os.path.basename(args.config).split('.')[0]}")

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        cv2.imwrite(os.path.join(save_path, f"{os.path.basename(result.img_path).split('.')[0]}.png"), color_seg,
                    [cv2.IMWRITE_PNG_COMPRESSION, 0])

if __name__ == '__main__':
    main()
