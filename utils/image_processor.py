#-----------------------------------------------------------------------#
#   utils/image_processor.py：图像处理工具类
#   功能概述：
#   1. 对图像区域进行马赛克脱敏处理
#   2. 可扩展添加更多图像处理操作
#-----------------------------------------------------------------------#

import cv2
import numpy as np

class ImageProcessor:
    """
    图像处理工具类
    
    主要功能：
    1. 对图像的指定区域进行马赛克处理
    2. 可扩展添加更多图像处理操作
    
    使用示例：
        processor = ImageProcessor()
        image = cv2.imread("test.jpg")
        processor.apply_mosaic(image, x1, y1, x2, y2, block_size=14)
    """
    
    def apply_mosaic(self, img_bgr, left, top, right, bottom, block=14):
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
