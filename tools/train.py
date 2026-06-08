# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

import torch
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS

# import torch
# import numpy as np
#
# # 添加 numpy 支持
# torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
#
# # 重写 torch.load，自动关闭 weights_only
# _original_torch_load = torch.load
#
# def patched_torch_load(*args, **kwargs):
#     kwargs['weights_only'] = False
#     return _original_torch_load(*args, **kwargs)
#
# torch.load = patched_torch_load

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    cfg.resume = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
    # runner.randomness=dict(seed=42)
    # runner.randomness=dict(seed=296958598)
    # pretrained_dict = torch.load(r"D:\test\mmsegmentation-main\work_dirs\1_stdc2_in1k-pre_b8x80k_acdc_fog-512x1024_l1_06627\best_mIoU_iter_70000.pth")
    # # pre = pre['state_dict']
    # # pretrained_dict = torch.load('model.pkl')
    # model_dict = pretrained_dict['state_dict']
    # # for k, v in model_dict.items():
    # #     print(k)
    # # print(len(pretrained_dict))
    # # 关键在于下面这句，从model_dict中读取key、value时，用if筛选掉不需要的网络层
    #
    # old_model_dict = runner.model.state_dict()
    #
    # model_dict = {key: value for key, value in model_dict.items() if 'sobel' in key}
    #
    # # print("-----------------------------------------------")
    # # print(type(model_dict))
    # # 替换权重
    # for k, v in model_dict.items():
    #     old_model_dict[k] = v
    #     # print(v)
    # # print("-----------------------------------------------")
    # # print(len(pretrained_dict))
    # # model_dict.update(pretrained_dict)
    # runner.model.load_state_dict(old_model_dict)

    # start training
    # runner.test()
    runner.train()

if __name__ == '__main__':
    main()
