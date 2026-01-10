#-----------------------------------------------------------------------#
#   utils/image_processor.py：图像处理工具类
#   功能概述：
#   1. 对图像区域进行多种马赛克脱敏处理
#   2. 支持像素化、高斯模糊、均值模糊、中值模糊、颜色填充等多种方式
#-----------------------------------------------------------------------#

import cv2
import numpy as np

class ImageProcessor:
    """
    图像处理工具类
    
    主要功能：
    1. 对图像的指定区域进行多种马赛克处理
    2. 支持像素化、高斯模糊、均值模糊、中值模糊、颜色填充等多种方式
    
    使用示例：
        processor = ImageProcessor()
        image = cv2.imread("test.jpg")
        # 使用像素化马赛克
        processor.apply_mosaic(image, x1, y1, x2, y2, mosaic_type='pixelate', block_size=14)
        # 使用高斯模糊马赛克
        processor.apply_mosaic(image, x1, y1, x2, y2, mosaic_type='gaussian', blur_size=15)
    """
    
    def apply_mosaic(self, img_bgr, left, top, right, bottom, mosaic_type='pixelate', **kwargs):
        """
        对 OpenCV BGR 图像的指定矩形区域应用马赛克脱敏处理
        
        支持的马赛克类型：
        1. pixelate: 像素化马赛克（默认）
        2. gaussian: 高斯模糊马赛克
        3. mean: 均值模糊马赛克
        4. median: 中值模糊马赛克
        5. color: 颜色填充马赛克
        
        参数:
            img_bgr: numpy.ndarray, OpenCV 格式的 BGR 图像，形状为 (H, W, 3)
            left: int/float, 矩形区域左边界（x 坐标）
            top: int/float, 矩形区域上边界（y 坐标）
            right: int/float, 矩形区域右边界（x 坐标）
            bottom: int/float, 矩形区域下边界（y 坐标）
            mosaic_type: str, 马赛克类型，可选值: 'pixelate', 'gaussian', 'mean', 'median', 'color'
            **kwargs: 额外参数，根据马赛克类型不同：
                - pixelate: block_size (默认14)
                - gaussian: blur_size (默认15), sigma_x (默认0)
                - mean: blur_size (默认15)
                - median: blur_size (默认15)
                - color: color (默认灰色[128, 128, 128])
        
        返回:
            img_bgr: numpy.ndarray, 打码后的 BGR 图像（原地修改）
        
        注意:
            - 函数会直接修改输入的 img_bgr，不创建副本（节省内存）
            - 坐标会被自动裁剪到图像边界内，防止越界
        """
        # 获取图像的高度和宽度
        h, w = img_bgr.shape[:2]
        
        # 边界检查：确保坐标在图像范围内，防止数组越界
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

        # 根据马赛克类型处理 ROI
        if mosaic_type == 'pixelate':
            # 像素化马赛克
            block_size = kwargs.get('block_size', 14)
            rh, rw = roi.shape[:2]
            sw = max(1, rw // block_size)
            sh = max(1, rh // block_size)
            small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_LINEAR)
            mosaic = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
            
        elif mosaic_type == 'gaussian':
            # 高斯模糊马赛克
            blur_size = kwargs.get('blur_size', 15)
            sigma_x = kwargs.get('sigma_x', 0)
            # 确保模糊核大小为奇数
            blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
            mosaic = cv2.GaussianBlur(roi, (blur_size, blur_size), sigma_x)
            
        elif mosaic_type == 'mean':
            # 均值模糊马赛克
            blur_size = kwargs.get('blur_size', 15)
            mosaic = cv2.blur(roi, (blur_size, blur_size))
            
        elif mosaic_type == 'median':
            # 中值模糊马赛克
            blur_size = kwargs.get('blur_size', 15)
            # 确保模糊核大小为奇数
            blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
            mosaic = cv2.medianBlur(roi, blur_size)
            
        elif mosaic_type == 'color':
            # 颜色填充马赛克
            color = kwargs.get('color', [128, 128, 128])  # 默认灰色
            mosaic = np.full_like(roi, color)
            
        else:
            # 未知类型，使用默认像素化马赛克
            block_size = kwargs.get('block_size', 14)
            rh, rw = roi.shape[:2]
            sw = max(1, rw // block_size)
            sh = max(1, rh // block_size)
            small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_LINEAR)
            mosaic = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)

        # 将处理后的区域替换回原图的对应位置
        img_bgr[top:bottom, left:right] = mosaic
        
        return img_bgr
