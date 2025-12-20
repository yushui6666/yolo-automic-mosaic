#-----------------------------------------------------------------------#
#   utils.py：通用工具函数集合
#   功能概述：
#   1. 图像格式转换和预处理
#   2. 图像尺寸调整（支持 letterbox）
#   3. 类别名称读取
#   4. 学习率获取
#   5. 随机种子设置
#   6. 配置信息显示
#   7. 模型权重下载
#-----------------------------------------------------------------------#
import random

import numpy as np
import torch
from PIL import Image


#---------------------------------------------------------#
#   将图像转换成 RGB 图像，防止灰度图在预测时报错
#   代码仅支持 RGB 图像的预测，所有其它类型的图像都会转化成 RGB
#---------------------------------------------------------#
def cvtColor(image):
    """
    将图像转换为 RGB 格式
    
    参数:
        image: PIL.Image, 输入图像（可能是灰度图或其他格式）
    
    返回:
        PIL.Image: RGB 格式的图像
    
    注意:
        - 如果图像已经是 RGB 格式（3 通道），直接返回
        - 否则转换为 RGB 格式（如灰度图会变成 3 通道灰度图）
    """
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image  # 已经是 RGB 格式
    else:
        image = image.convert('RGB')  # 转换为 RGB
        return image 

#---------------------------------------------------#
#   对输入图像进行 resize
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    """
    调整图像尺寸
    
    参数:
        image: PIL.Image, 输入图像
        size: tuple, 目标尺寸 (width, height)
        letterbox_image: bool, 是否使用 letterbox 方式
                         - True: 保持宽高比，用灰色填充
                         - False: 直接拉伸（可能变形）
    
    返回:
        PIL.Image: 调整后的图像
    
    注意:
        - letterbox 方式可以避免图像变形，但会添加灰色边框
        - 直接 resize 会改变图像宽高比，可能导致目标变形
    """
    iw, ih  = image.size  # 原始图像宽度和高度
    w, h    = size        # 目标图像宽度和高度
    
    if letterbox_image:
        # Letterbox 方式：保持宽高比，用灰色填充
        # 计算缩放比例（取较小的比例，确保图像完全包含在目标尺寸内）
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)  # 缩放后的宽度
        nh      = int(ih*scale)  # 缩放后的高度

        # 缩放图像（使用双三次插值，质量较好）
        image   = image.resize((nw, nh), Image.BICUBIC)
        
        # 创建灰色背景图像
        new_image = Image.new('RGB', size, (128, 128, 128))
        
        # 将缩放后的图像居中粘贴到灰色背景上
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        # 直接拉伸到目标尺寸（可能变形）
        new_image = image.resize((w, h), Image.BICUBIC)
    
    return new_image

#---------------------------------------------------#
#   读取类别名称列表
#---------------------------------------------------#
def get_classes(classes_path):
    """
    从文件中读取类别名称列表
    
    参数:
        classes_path: str, 类别文件路径（每行一个类别名）
    
    返回:
        tuple: (class_names, num_classes)
            - class_names: list, 类别名称列表
            - num_classes: int, 类别数量
    
    文件格式示例:
        person
        car
        dog
        ...
    """
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    # 去除每行的换行符和空格
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

#---------------------------------------------------#
#   获取当前学习率
#---------------------------------------------------#
def get_lr(optimizer):
    """
    从优化器中获取当前学习率
    
    参数:
        optimizer: torch.optim.Optimizer, 优化器对象
    
    返回:
        float: 当前学习率
    
    注意:
        - 如果优化器有多个参数组，返回第一个参数组的学习率
        - 通常所有参数组使用相同的学习率
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   设置随机种子（确保结果可复现）
#---------------------------------------------------#
def seed_everything(seed=11):
    """
    设置所有随机数生成器的种子，确保结果可复现
    
    参数:
        seed: int, 随机种子值，默认 11
    
    功能:
        1. 设置 Python 内置 random 模块的种子
        2. 设置 NumPy 的随机种子
        3. 设置 PyTorch CPU 的随机种子
        4. 设置 PyTorch GPU 的随机种子（所有 GPU）
        5. 设置 cuDNN 为确定性模式（可能降低性能但保证可复现）
    
    注意:
        - 设置确定性模式可能会降低训练速度
        - 在分布式训练中，每个进程应该使用不同的种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 确定性模式
    torch.backends.cudnn.benchmark = False     # 关闭 benchmark（保证可复现）

#---------------------------------------------------#
#   设置 DataLoader 工作进程的随机种子
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    """
    为 DataLoader 的每个工作进程设置不同的随机种子
    
    参数:
        worker_id: int, 工作进程 ID
        rank: int, 分布式训练的进程排名
        seed: int, 基础随机种子
    
    用途:
        在 DataLoader 初始化时使用：
        DataLoader(..., worker_init_fn=lambda id: worker_init_fn(id, rank, seed))
    
    注意:
        - 每个工作进程使用不同的种子，确保数据打乱的一致性
        - 在分布式训练中，不同进程使用不同的基础种子（rank）
    """
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def preprocess_input(image):
    """
    图像预处理：归一化到 [0, 1] 范围
    
    参数:
        image: numpy.ndarray, 输入图像（像素值范围 0-255）
    
    返回:
        numpy.ndarray: 归一化后的图像（像素值范围 0-1）
    
    注意:
        - 将像素值从 [0, 255] 缩放到 [0, 1]
        - 这是深度学习模型常用的预处理步骤
    """
    image /= 255.0
    return image

def show_config(**kwargs):
    """
    以表格形式打印配置信息
    
    参数:
        **kwargs: 配置参数字典
    
    功能:
        以格式化的表格形式显示所有配置参数，便于查看和调试
    
    输出示例:
        Configurations:
        ----------------------------------------------------------------------
        |                    keys |                                 values|
        ----------------------------------------------------------------------
        |              model_path |         model/best_epoch_weights.pth|
        |            input_shape |                              [640, 640]|
        ...
    """
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)
        
def download_weights(phi, model_dir="./model_data"):
    """
    从 GitHub 下载预训练模型权重
    
    参数:
        phi: str, 模型变体标识符
            - "n": nano（最小模型）
            - "s": small（小模型）
            - "m": medium（中等模型）
            - "l": large（大模型）
            - "x": xlarge（超大模型）
        model_dir: str, 模型保存目录，默认 "./model_data"
    
    功能:
        1. 根据模型变体选择对应的下载链接
        2. 创建保存目录（如果不存在）
        3. 从 GitHub 下载预训练权重文件
    
    注意:
        - 需要网络连接
        - 下载的是 backbone 权重，需要进一步训练才能用于检测任务
    """
    import os

    from torch.hub import load_state_dict_from_url
    
    # 不同模型变体的下载链接
    download_urls = {
        "n": 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_n_backbone_weights.pth',
        "s": 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s_backbone_weights.pth',
        "m": 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_m_backbone_weights.pth',
        "l": 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_l_backbone_weights.pth',
        "x": 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_x_backbone_weights.pth',
    }
    url = download_urls[phi]
    
    # 创建保存目录
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 下载权重文件
    load_state_dict_from_url(url, model_dir)