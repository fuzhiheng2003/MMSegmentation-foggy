# 环境配置
首先需要准备好conda以及torch环境,安装conda这一步建议查找相关博客解决

关于MMSegmentation框架本身的信息可以查询它在github上的原版readme：https://github.com/open-mmlab/mmsegmentation

50系显卡推荐使用该博主的博客配置mmcv环境 https://blog.csdn.net/qq_43356449/article/details/147192685#comments_38658181

需要基于自己实验环境下的torch版本安装相关mmcv对应版本的库，以下教程仅在英伟达50系显卡上进行过验证，非50系可以参考部分流程

首先win+r输入cmd打开终端，开始创建环境
``
conda create -n openmmlab python=3.9 -y
conda activate openmmlab
``

然后安装torch
``
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
``

验证安装是否成功
``python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
``

配置mmcv环境
`pip install -U openmim
mim install mmcv==2.1.0  #这是我唯一试出来可行的版本
`

然后将使用cd指令将终端的工作目录跳转到下载的项目文件的解压文件夹目录下，运行指令，将项目文件中的mmseg包导入到conda环境中
``pip install -r requirements/build.txt
pip install -v -e .``

由于mmseg包的特性，该环境下的mmseg是绑定该项目本身的，如果还需要运行另外一个mmseg项目，则需要重新创建一个新环境
# 数据集配置
将下载好的数据集放到对应的data文件夹内，注意不要嵌套，即data文件夹内的数据集名称文件夹为一级文件夹，二级文件夹为train，val等子文件夹

# 训练，推理和测试
在打开的终端内，激活该项目对应的conda环境`conda activate openmmlab`，然后使用cd指令跳转到该项目的目录下，输入
`python tools/train.py configs/stdc1/1_stdc2_in1k-pre_b8x80k_acdc_fog-512x1024.py`
即可开始训练，相关对比算法训练指令以及推理和测试的操作指令均存放于quick_train文件夹内

推理的话记得使用work_dirs内的权重进行推理，本项目给出了本文方法在cityscapes数据集上训练的权重以及在acdc雾天数据集部分训练的权重，可直接使用或者测试

