#-----------------------------------------------------------------------#
#   yolo2.py：YOLO 人脸检测器 + 马赛克脱敏功能
#   功能概述：
#   1. 基于 YOLO 模型进行人脸检测
#   2. 对检测到的人脸区域进行马赛克脱敏处理
#   3. 支持单张图片检测和视频流检测（通过 detect_boxes 接口）
#-----------------------------------------------------------------------#

# 颜色空间转换工具，用于生成不同类别的显示颜色
import colorsys
# 文件系统操作
import os
# 时间计算，用于 FPS 测试
import time
# OpenCV：图像处理和计算机视觉库
import cv2
# NumPy：数值计算库
import numpy as np
# PyTorch：深度学习框架
import torch
import torch.nn as nn
# PIL：Python 图像处理库
from PIL import Image, ImageFont

# 导入自定义模块
from nets.yolo import YoloBody  # YOLO 网络结构定义
from utils.utils import (cvtColor, get_classes, preprocess_input,
                         resize_image, show_config)  # 图像预处理工具函数
from utils.utils_bbox import DecodeBox  # 边界框解码和非极大值抑制工具


def mosaic_area(img_bgr, left, top, right, bottom, block=14):
    """
    对 OpenCV BGR 图像的指定矩形区域打马赛克（像素化处理）
    
    马赛克原理：
    1. 将目标区域缩小到很小的尺寸（如原尺寸的 1/block）
    2. 再将缩小后的图像放大回原尺寸
    3. 由于使用最近邻插值，会产生明显的像素块效果
    
    参数:
        img_bgr: numpy.ndarray, OpenCV 格式的 BGR 图像，形状为 (H, W, 3)
        left: int/float, 矩形区域左边界（x 坐标）
        top: int/float, 矩形区域上边界（y 坐标）
        right: int/float, 矩形区域右边界（x 坐标）
        bottom: int/float, 矩形区域下边界（y 坐标）
        block: int, 马赛克块大小，控制马赛克强度
               - 值越大，马赛克块越大，图像越模糊（脱敏效果更强）
               - 值越小，马赛克块越小，图像相对清晰
               - 默认值 14 表示将区域分成约 14x14 个块
    
    返回:
        img_bgr: numpy.ndarray, 打码后的 BGR 图像（原地修改）
    
    注意:
        - 函数会直接修改输入的 img_bgr，不创建副本（节省内存）
        - 坐标会被自动裁剪到图像边界内，防止越界
    """
    # 获取图像的高度和宽度
    h, w = img_bgr.shape[:2]
    
    # 边界检查：确保坐标在图像范围内，防止数组越界
    # left 和 top 不能小于 0，也不能大于等于图像尺寸
    left = max(0, min(int(left), w - 1))
    right = max(0, min(int(right), w))
    top = max(0, min(int(top), h - 1))
    bottom = max(0, min(int(bottom), h))

    # 有效性检查：如果矩形区域无效（宽度或高度 <= 0），直接返回原图
    if right <= left or bottom <= top:
        return img_bgr

    # 提取感兴趣区域（Region of Interest, ROI）
    roi = img_bgr[top:bottom, left:right]
    
    # 如果 ROI 为空，返回原图
    if roi.size == 0:
        return img_bgr

    # 获取 ROI 的高度和宽度
    rh, rw = roi.shape[:2]
    
    # 计算缩小后的尺寸：将原尺寸除以 block，得到马赛克块的数量
    # max(1, ...) 确保至少为 1，避免除零错误
    sw = max(1, rw // block)  # 缩小后的宽度
    sh = max(1, rh // block)  # 缩小后的高度

    # 步骤 1：将 ROI 缩小到很小的尺寸（使用线性插值）
    # 这一步会丢失大量细节，产生模糊效果
    small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_LINEAR)
    
    # 步骤 2：将缩小后的图像放大回原始尺寸（使用最近邻插值）
    # 最近邻插值会产生明显的像素块，形成马赛克效果
    mosaic = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    
    # 步骤 3：将马赛克区域替换回原图的对应位置
    img_bgr[top:bottom, left:right] = mosaic
    
    return img_bgr


class YOLO(object):
    """
    YOLO 人脸检测器类
    
    主要功能：
    1. 加载训练好的 YOLO 模型权重
    2. 对输入图像进行人脸检测
    3. 对检测到的人脸进行马赛克脱敏处理
    4. 提供多种检测接口（单图检测、框检测、FPS 测试等）
    
    使用示例：
        yolo = YOLO()
        image = Image.open("test.jpg")
        result = yolo.detect_image(image)  # 返回打码后的图像
    """
    
    # 类属性：默认配置参数字典
    _defaults = {
        # 模型相关配置
        "model_path": 'model/best_epoch_weights.pth',  # 训练好的模型权重文件路径
        "classes_path": 'model/voc_classes.txt',       # 类别名称文件路径（包含所有检测类别）
        "input_shape": [640, 640],                     # 模型输入图像尺寸 [高度, 宽度]
        "phi": 's',                                   # YOLO 模型变体，'s' 表示 small（小模型，速度快）
        
        # 检测相关配置
        "confidence": 0.3,    # 置信度阈值：只有置信度 >= 0.3 的检测框才会被保留
        "nms_iou": 0.3,       # NMS（非极大值抑制）的 IoU 阈值：用于去除重叠的检测框
        "letterbox_image": True,  # 是否使用 letterbox 方式缩放图像（保持宽高比，避免变形）
        "cuda": True,         # 是否使用 GPU 加速（需要 CUDA 支持）
        
        # 脱敏相关配置（可按需调整）
        "mosaic_block": 14,      # 马赛克块大小：值越大，马赛克效果越强（图像越模糊）
        "expand_ratio": 0.2,    # 扩框比例：检测框向外扩展的比例，防止人脸边缘漏遮
                                 # 例如：原框宽度为 100，扩展后为 100 * (1 + 0.2) = 120
        "min_face_size": 10,     # 最小人脸尺寸：小于此尺寸的检测框会被忽略（可能是误检）
    }

    @classmethod
    def get_defaults(cls, n):
        """
        类方法：获取默认配置参数的值
        
        参数:
            n: str, 参数名称
        
        返回:
            参数值，如果参数不存在则返回错误信息字符串
        """
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        """
        初始化 YOLO 检测器
        
        参数:
            **kwargs: 可选的关键字参数，用于覆盖默认配置
                     例如：YOLO(confidence=0.5, cuda=False)
        
        初始化流程：
        1. 加载默认配置
        2. 用 kwargs 覆盖默认配置
        3. 读取类别名称文件
        4. 初始化边界框解码器
        5. 生成类别颜色（用于可视化）
        6. 加载模型权重
        """
        # 步骤 1：将默认配置更新到实例属性
        self.__dict__.update(self._defaults)
        
        # 步骤 2：用用户传入的参数覆盖默认配置
        for name, value in kwargs.items():
            setattr(self, name, value)  # 设置实例属性
            self._defaults[name] = value  # 同时更新类默认配置（影响后续实例）

        # 步骤 3：读取类别名称文件，获取所有检测类别
        # 返回：类别名称列表和类别数量
        self.class_names, self.num_classes = get_classes(self.classes_path)

        # 步骤 4：初始化边界框解码器
        # DecodeBox 负责将模型输出的原始特征图解码成边界框坐标
        self.bbox_util = DecodeBox(
            self.num_classes,  # 类别数量
            (self.input_shape[0], self.input_shape[1])  # 输入图像尺寸
        )

        # 步骤 5：为每个类别生成不同的显示颜色（HSV 色彩空间均匀分布）
        # HSV 色彩空间：H（色相）、S（饱和度）、V（明度）
        # 在 HSV 空间中均匀分布颜色，确保不同类别颜色区分明显
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        
        # 将 HSV 转换为 RGB（浮点数 0-1 范围）
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        
        # 将 RGB 从 0-1 范围转换为 0-255 范围（整数）
        self.colors = list(map(
            lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), 
            self.colors
        ))

        # 步骤 6：加载模型权重（调用 generate 方法）
        self.generate()
        
        # 显示配置信息（用于调试和确认参数）
        show_config(**self._defaults)

    def generate(self, onnx=False):
        """
        加载并初始化 YOLO 模型
        
        参数:
            onnx: bool, 是否导出为 ONNX 格式（当前未使用）
        
        流程：
        1. 创建 YOLO 网络结构
        2. 加载预训练权重
        3. 模型融合优化（fuse）
        4. 设置为评估模式（eval）
        5. 如果启用 CUDA，将模型移到 GPU 并支持多 GPU
        """
        # 步骤 1：创建 YOLO 网络结构
        # YoloBody: YOLO 的主干网络，根据 input_shape、num_classes、phi 参数构建
        self.net = YoloBody(
            self.input_shape,    # 输入图像尺寸
            self.num_classes,    # 类别数量
            self.phi            # 模型变体（'s', 'm', 'l', 'x' 等）
        )
        
        # 步骤 2：选择计算设备（优先使用 GPU）
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 步骤 3：加载预训练权重
        # map_location=device 确保权重加载到正确的设备上（CPU 或 GPU）
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        
        # 步骤 4：模型优化
        # fuse(): 融合 Conv 和 BN 层，减少计算量，提升推理速度
        # eval(): 设置为评估模式，关闭 dropout 和 batch normalization 的训练行为
        self.net = self.net.fuse().eval()
        
        print('{} model, and classes loaded.'.format(self.model_path))

        # 步骤 5：如果启用 CUDA 且不是 ONNX 模式，将模型移到 GPU
        if not onnx and self.cuda:
            # DataParallel: 多 GPU 并行推理（如果有多块 GPU）
            self.net = nn.DataParallel(self.net)
            # 将模型移到 GPU（默认使用第一块 GPU，索引 0）
            self.net = self.net.cuda()

    def _infer(self, image):
        """
        内部推理方法：执行 YOLO 前向传播，返回检测结果
        
        参数:
            image: PIL.Image, 输入的 RGB 图像
        
        返回:
            results[0]: numpy.ndarray 或 None
                - 如果检测到目标：形状为 (N, 6) 的数组
                  - 每行格式：[top, left, bottom, right, conf, cls]
                  - top, left, bottom, right: 边界框坐标（像素）
                  - conf: 置信度（0-1）
                  - cls: 类别索引（整数）
                - 如果未检测到目标：返回 None
        
        注意:
            这是内部方法，不直接对外暴露，供其他方法调用
        """
        # 步骤 1：获取原始图像尺寸（高度、宽度）
        image_shape = np.array(np.shape(image)[0:2])
        
        # 步骤 2：图像格式转换（确保是 RGB 格式）
        image = cvtColor(image)
        
        # 步骤 3：图像尺寸调整和预处理
        # resize_image: 将图像缩放到模型输入尺寸（如 640x640）
        # letterbox_image: 如果为 True，保持宽高比并填充黑边
        image_data = resize_image(
            image, 
            (self.input_shape[1], self.input_shape[0]),  # (宽度, 高度)
            self.letterbox_image
        )
        
        # 步骤 4：图像数据预处理
        # preprocess_input: 归一化（通常是将像素值从 0-255 缩放到 0-1 或标准化）
        # transpose(2, 0, 1): 将 (H, W, C) 转换为 (C, H, W)，符合 PyTorch 输入格式
        # expand_dims(..., 0): 添加 batch 维度，从 (C, H, W) 变为 (1, C, H, W)
        image_data = np.expand_dims(
            np.transpose(
                preprocess_input(np.array(image_data, dtype='float32')), 
                (2, 0, 1)
            ),
            0
        )

        # 步骤 5：模型推理（禁用梯度计算，节省内存和加速）
        with torch.no_grad():
            # 将 numpy 数组转换为 PyTorch 张量
            images = torch.from_numpy(image_data)
            
            # 如果启用 CUDA，将张量移到 GPU
            if self.cuda:
                images = images.cuda()

            # 前向传播：模型推理
            outputs = self.net(images)
            
            # 解码边界框：将模型输出的特征图解码成边界框坐标和类别
            outputs = self.bbox_util.decode_box(outputs)
            
            # 非极大值抑制（NMS）：去除重叠的检测框，只保留最佳检测结果
            results = self.bbox_util.non_max_suppression(
                outputs,                    # 解码后的检测结果
                self.num_classes,           # 类别数量
                self.input_shape,          # 输入图像尺寸
                image_shape,               # 原始图像尺寸（用于坐标还原）
                self.letterbox_image,      # 是否使用 letterbox
                conf_thres=self.confidence, # 置信度阈值
                nms_thres=self.nms_iou     # NMS IoU 阈值
            )

        # 步骤 6：返回检测结果（results 是一个列表，results[0] 是第一个 batch 的结果）
        if results[0] is None:
            return None
        return results[0]

    def detect_boxes(self, image):
        """
        检测接口：返回检测框坐标、置信度和类别标签
        主要用于视频跟踪场景，只返回检测框信息，不进行脱敏处理
        
        参数:
            image: PIL.Image, 输入的 RGB 图像
        
        返回:
            det_xyxy: numpy.ndarray, 形状为 (N, 4)，数据类型 float32
                     每行格式：[x1, y1, x2, y2]（左上角和右下角坐标）
            det_scores: numpy.ndarray, 形状为 (N,)，数据类型 float32
                        每个检测框的置信度
            det_labels: numpy.ndarray, 形状为 (N,)，数据类型 int32
                        每个检测框的类别索引
        
        使用场景:
            - 视频流处理：需要将检测框传递给跟踪器（如 SimpleTracker）
            - 实时检测：只需要框信息，不需要可视化或脱敏
        """
        # 调用内部推理方法获取检测结果
        r = self._infer(image)
        
        # 如果未检测到目标，返回空数组（保持输出格式一致）
        if r is None:
            return (
                np.zeros((0, 4), dtype=np.float32),  # 空检测框数组
                np.zeros((0,), dtype=np.float32),    # 空置信度数组
                np.zeros((0,), dtype=np.int32)       # 空类别数组
            )

        # 提取检测结果中的类别、置信度和边界框
        # r 的格式：[top, left, bottom, right, conf, cls]
        top_label = np.array(r[:, 5], dtype='int32')      # 类别索引（第 6 列）
        top_conf = r[:, 4].astype(np.float32)             # 置信度（第 5 列）
        top_boxes = r[:, :4].astype(np.float32)           # 边界框 [top, left, bottom, right]（前 4 列）

        # 坐标格式转换：从 [top, left, bottom, right] 转换为 [x1, y1, x2, y2]
        # 这是为了与跟踪器和其他工具的标准格式保持一致
        det_xyxy = np.stack([
            top_boxes[:, 1],  # left -> x1
            top_boxes[:, 0],  # top -> y1
            top_boxes[:, 3],  # right -> x2
            top_boxes[:, 2]   # bottom -> y2
        ], axis=1).astype(np.float32)
        
        return det_xyxy, top_conf, top_label

    def detect_image(self, image, crop=False, count=False):
        """
        单张图片检测接口：检测人脸并对检测区域进行马赛克脱敏处理
        
        参数:
            image: PIL.Image, 输入的 RGB 图像
            crop: bool, 是否保存裁剪的人脸图片到 "img_crop" 目录
                  如果为 True，会将每个检测到的人脸区域单独保存
            count: bool, 是否打印每个类别的检测数量统计信息
        
        返回:
            image_out: PIL.Image, 脱敏处理后的 RGB 图像
                      如果未检测到人脸，返回原图
        
        使用场景:
            - 批量图片处理：对文件夹中的图片进行脱敏
            - 单张图片测试：快速查看检测和脱敏效果
        """
        # 步骤 1：执行模型推理，获取检测结果
        r = self._infer(image)
        
        # 步骤 2：确保图像是 RGB 格式
        image = cvtColor(image)

        # 如果未检测到目标，直接返回原图
        if r is None:
            return image

        # 步骤 3：提取检测结果
        top_label = np.array(r[:, 5], dtype='int32')      # 类别索引
        top_conf = r[:, 4].astype(np.float32)             # 置信度
        top_boxes = r[:, :4].astype(np.float32)           # 边界框 [top, left, bottom, right]

        # 步骤 4：如果启用计数功能，统计每个类别的检测数量
        if count:
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)  # 统计类别 i 的检测数量
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        # 步骤 5：如果启用裁剪功能，保存每个检测到的人脸区域
        if crop:
            for i, _ in enumerate(top_boxes):
                top, left, bottom, right = top_boxes[i]
                # 边界检查，确保坐标在图像范围内
                top = max(0, int(top))
                left = max(0, int(left))
                bottom = min(image.size[1], int(bottom))  # image.size[1] 是高度
                right = min(image.size[0], int(right))    # image.size[0] 是宽度

                # 创建保存目录
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                
                # 裁剪人脸区域并保存
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(
                    os.path.join(dir_save_path, "crop_" + str(i) + ".png"), 
                    quality=95, 
                    subsampling=0
                )

        # 步骤 6：将 PIL 图像转换为 OpenCV 格式（BGR），用于马赛克处理
        # OpenCV 使用 BGR 格式，PIL 使用 RGB 格式，需要转换
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        w_img, h_img = image.size  # 图像宽度和高度

        # 步骤 7：遍历每个检测到的人脸，进行马赛克脱敏
        for i, c in list(enumerate(top_label)):
            score = float(top_conf[i])  # 当前检测框的置信度
            box = top_boxes[i]          # 当前检测框坐标 [top, left, bottom, right]
            
            # 提取并验证边界框坐标
            top, left, bottom, right = box
            top = max(0, int(top))
            left = max(0, int(left))
            bottom = min(h_img, int(bottom))
            right = min(w_img, int(right))

            # 计算边界框的宽度和高度
            bw = right - left
            bh = bottom - top
            
            # 过滤过小的检测框（可能是误检或噪声）
            if bw < self.min_face_size or bh < self.min_face_size:
                continue

            # 步骤 8：扩展边界框（防止人脸边缘漏遮）
            # 计算原框的中心点
            cx = (left + right) // 2
            cy = (top + bottom) // 2
            
            # 计算扩展后的宽度和高度
            nw = int(bw * (1 + self.expand_ratio))  # 例如：100 * 1.2 = 120
            nh = int(bh * (1 + self.expand_ratio))
            
            # 计算扩展后的边界框坐标（以中心点为中心，向外扩展）
            left_e = max(0, cx - nw // 2)      # 扩展后的左边界
            right_e = min(w_img, cx + nw // 2)  # 扩展后的右边界
            top_e = max(0, cy - nh // 2)        # 扩展后的上边界
            bottom_e = min(h_img, cy + nh // 2) # 扩展后的下边界

            # 步骤 9：对扩展后的区域进行马赛克处理
            image_cv = mosaic_area(
                image_cv, 
                left_e, top_e, right_e, bottom_e, 
                block=self.mosaic_block
            )

        # 步骤 10：将处理后的 OpenCV BGR 图像转换回 PIL RGB 格式
        image_out = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        
        return image_out

    def get_FPS(self, image, test_interval):
        """
        计算模型推理的 FPS（每秒帧数）性能
        
        参数:
            image: PIL.Image, 测试用的输入图像
            test_interval: int, 测试循环次数
                          例如：test_interval=100 表示推理 100 次，然后计算平均时间
        
        返回:
            float, 平均每次推理的时间（秒）
                   FPS = 1 / 返回值
        
        使用场景:
            - 性能测试：评估模型在不同硬件上的推理速度
            - 优化对比：比较不同配置（如是否使用 GPU）的性能差异
        
        注意:
            - 第一次推理通常较慢（模型初始化、内存分配等），所以先执行一次预热
            - 后续多次推理的平均时间更能反映真实性能
        """
        # 步骤 1：图像预处理（与 _infer 方法相同）
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(
            image, 
            (self.input_shape[1], self.input_shape[0]), 
            self.letterbox_image
        )
        image_data = np.expand_dims(
            np.transpose(
                preprocess_input(np.array(image_data, dtype='float32')), 
                (2, 0, 1)
            ),
            0
        )

        # 步骤 2：预热推理（第一次推理，不计入时间统计）
        # 预热可以触发 GPU 初始化、CUDA 内核编译等，使后续推理更稳定
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            _ = self.bbox_util.non_max_suppression(
                outputs, 
                self.num_classes, 
                self.input_shape,
                image_shape, 
                self.letterbox_image,
                conf_thres=self.confidence, 
                nms_thres=self.nms_iou
            )

        # 步骤 3：记录开始时间
        t1 = time.time()
        
        # 步骤 4：执行多次推理，计算平均时间
        for _ in range(test_interval):
            with torch.no_grad():
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                _ = self.bbox_util.non_max_suppression(
                    outputs, 
                    self.num_classes, 
                    self.input_shape,
                    image_shape, 
                    self.letterbox_image,
                    conf_thres=self.confidence, 
                    nms_thres=self.nms_iou
                )
        
        # 步骤 5：记录结束时间，计算平均推理时间
        t2 = time.time()
        
        # 返回平均每次推理的时间（秒）
        # 例如：100 次推理耗时 2 秒，则返回 0.02 秒，FPS = 1/0.02 = 50
        return (t2 - t1) / test_interval