#-----------------------------------------------------------------------#
#   face_pi.py：人脸关键点定位、对齐和特征提取工具类
#   功能概述：
#   1. 使用 InsightFace 检测器提取人脸 5 个关键点
#   2. 将人脸对齐到标准 112x112 尺寸
#   3. 使用 ArcFace 模型提取人脸特征向量（embedding）
#   4. 支持人脸关键点可视化
#-----------------------------------------------------------------------#
import cv2
import numpy as np

#-----------------------------------------------------------------------#
#   ArcFace 标准模板：112x112 图像中的 5 个关键点标准位置
#   这 5 个关键点分别是：
#   - 左眼中心
#   - 右眼中心
#   - 鼻尖
#   - 左嘴角
#   - 右嘴角
#   这些坐标是 ArcFace 模型训练时使用的标准对齐模板
#-----------------------------------------------------------------------#
TEMPLATE_112 = np.array([
    [38.2946, 51.6963],  # 左眼中心
    [73.5318, 51.5014],  # 右眼中心
    [56.0252, 71.7366],  # 鼻尖
    [41.5493, 92.3655],  # 左嘴角
    [70.7299, 92.2041],  # 右嘴角
], dtype=np.float32)

class FaceKpsAlignRec:
    """
    人脸关键点定位、对齐和特征提取类
    
    主要功能：
    1. 使用 InsightFace 检测器在给定的人脸框内检测 5 个关键点
    2. 通过仿射变换将人脸对齐到标准的 112x112 尺寸
    3. 使用 ArcFace 模型提取归一化的人脸特征向量（512 维）
    
    使用流程：
        face_id = FaceKpsAlignRec(rec_onnx_path="path/to/model.onnx")
        kps5 = face_id.kps5_from_bbox(img_bgr, bbox_xyxy)  # 提取关键点
        aligned = face_id.align_112(img_bgr, kps5)          # 对齐人脸
        embedding = face_id.embedding_from_aligned(aligned) # 提取特征
    """
    def __init__(self,
                 det_size=(640, 640),
                 ctx_id=0,
                 providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
                 rec_onnx_path=None):
        """
        初始化人脸关键点检测和特征提取模型
        
        参数:
            det_size: tuple, InsightFace 检测器的输入图像尺寸，默认 (640, 640)
                     较大的尺寸可以提高检测精度，但会增加计算量
            ctx_id: int, GPU 设备索引，0 表示第一块 GPU，-1 表示使用 CPU
            providers: tuple, ONNX Runtime 的执行提供者列表
                       - CUDAExecutionProvider: 使用 GPU 加速
                       - CPUExecutionProvider: 使用 CPU 执行（备用）
            rec_onnx_path: str, ArcFace 特征提取模型的 ONNX 文件路径
                          例如：r"C:\Users\xxx\.insightface\models\buffalo_l\w600k_r50.onnx"
        
        注意:
            - 需要先安装 insightface 库：pip install insightface
            - 需要下载对应的模型文件
        """
        from insightface.app import FaceAnalysis
        from insightface.model_zoo import get_model

        # A) 初始化人脸检测器：只使用检测模块（会输出边界框和 5 个关键点）
        # FaceAnalysis 是 InsightFace 提供的高级接口
        self.det_app = FaceAnalysis(
            allowed_modules=["detection"],  # 只启用检测模块，不需要识别模块
            providers=list(providers)       # 指定执行提供者
        )
        # 准备检测器：设置设备、输入尺寸等
        self.det_app.prepare(ctx_id=ctx_id, det_size=det_size)

        # B) 初始化人脸识别模型：ArcFace 特征提取模型
        # 该模型用于将对齐后的人脸图像转换为特征向量
        if rec_onnx_path is None:
            raise ValueError("rec_onnx_path 不能为空，例如 ...\\.insightface\\models\\buffalo_l\\w600k_r50.onnx")

        # 加载 ONNX 模型
        self.rec_model = get_model(rec_onnx_path, providers=list(providers))
        # 准备识别模型
        self.rec_model.prepare(ctx_id=ctx_id)

    @staticmethod
    def _clip_roi(img_bgr, bbox_xyxy, margin=0.2):
        """
        从图像中裁剪出感兴趣区域（ROI），并扩展边界
        
        参数:
            img_bgr: numpy.ndarray, BGR 格式的输入图像，形状为 (H, W, 3)
            bbox_xyxy: array-like, 边界框坐标 [x1, y1, x2, y2]
            margin: float, 边界扩展比例，默认 0.2（即扩展 20%）
                   扩展边界有助于包含完整的人脸区域，提高关键点检测准确率
        
        返回:
            tuple: (roi, offset_x, offset_y) 或 None
                - roi: numpy.ndarray, 裁剪后的 ROI 图像
                - offset_x: int, ROI 在原图中的 x 偏移量（用于坐标转换）
                - offset_y: int, ROI 在原图中的 y 偏移量（用于坐标转换）
                - 如果 ROI 无效（太小或为空），返回 None
        
        注意:
            这是一个静态方法，不依赖实例状态
        """
        # 提取边界框坐标并转换为整数
        x1, y1, x2, y2 = map(int, bbox_xyxy)
        h, w = img_bgr.shape[:2]  # 获取图像高度和宽度
        bw, bh = x2 - x1, y2 - y1  # 计算边界框的宽度和高度
        
        # 过滤过小的边界框（可能是误检）
        if bw < 10 or bh < 10:
            return None

        # 计算扩展量：基于边界框的最大边长
        m = int(margin * max(bw, bh))
        
        # 计算扩展后的边界框坐标（确保不超出图像范围）
        xx1 = max(0, x1 - m)      # 左边界（向左扩展）
        yy1 = max(0, y1 - m)      # 上边界（向上扩展）
        xx2 = min(w, x2 + m)      # 右边界（向右扩展）
        yy2 = min(h, y2 + m)      # 下边界（向下扩展）
        
        # 裁剪 ROI
        roi = img_bgr[yy1:yy2, xx1:xx2]
        
        # 检查 ROI 是否为空
        if roi.size == 0:
            return None
        
        # 返回 ROI 和偏移量（用于将 ROI 内的坐标转换回原图坐标系）
        return (roi, xx1, yy1)

    def kps5_from_bbox(self, img_bgr, bbox_xyxy, margin=0.2):
        """
        从给定的人脸边界框中提取 5 个关键点
        
        参数:
            img_bgr: numpy.ndarray, BGR 格式的输入图像
            bbox_xyxy: array-like, 人脸边界框坐标 [x1, y1, x2, y2]
            margin: float, 边界扩展比例，默认 0.2
        
        返回:
            numpy.ndarray 或 None: 形状为 (5, 2) 的关键点坐标数组
                - 每行表示一个关键点：[x, y]
                - 5 个关键点顺序：左眼、右眼、鼻尖、左嘴角、右嘴角
                - 坐标是相对于原图的绝对坐标
                - 如果检测失败，返回 None
        
        流程:
            1. 裁剪并扩展 ROI 区域
            2. 在 ROI 中检测人脸关键点
            3. 将 ROI 坐标系的关键点转换回原图坐标系
        """
        # 步骤 1：裁剪 ROI 区域（带边界扩展）
        pack = self._clip_roi(img_bgr, bbox_xyxy, margin=margin)
        if pack is None:
            return None
        roi, offx, offy = pack  # offx, offy 是 ROI 在原图中的偏移量

        # 步骤 2：在 ROI 中检测人脸（InsightFace 检测器）
        faces = self.det_app.get(roi)
        if len(faces) == 0:
            return None  # 未检测到人脸
        
        # 步骤 3：提取第一个检测到的人脸的关键点
        kps = getattr(faces[0], "kps", None)
        if kps is None:
            return None  # 该人脸没有关键点信息

        # 步骤 4：将关键点坐标从 ROI 坐标系转换回原图坐标系
        kps5 = kps.astype(np.float32)
        kps5[:, 0] += offx  # x 坐标加上 x 偏移量
        kps5[:, 1] += offy  # y 坐标加上 y 偏移量
        
        return kps5  # 返回 (5, 2) 数组，坐标是原图坐标系

    @staticmethod
    def align_112(img_bgr, kps5):
        """
        将人脸对齐到标准的 112x112 尺寸
        
        对齐原理：
            使用仿射变换（Affine Transform）将检测到的 5 个关键点
            映射到标准的 5 个关键点位置，从而矫正人脸的姿态和尺度
        
        参数:
            img_bgr: numpy.ndarray, BGR 格式的输入图像
            kps5: numpy.ndarray, 形状为 (5, 2) 的关键点坐标数组
                  5 个关键点顺序：左眼、右眼、鼻尖、左嘴角、右嘴角
        
        返回:
            numpy.ndarray 或 None: 对齐后的 112x112 BGR 图像
                - 如果变换矩阵计算失败，返回 None
        
        注意:
            - 使用 LMEDS（Least-Median）方法估计变换矩阵，对异常值更鲁棒
            - 对齐后的人脸图像可以直接输入 ArcFace 模型提取特征
        """
        # 步骤 1：估计仿射变换矩阵
        # estimateAffinePartial2D: 估计相似变换（旋转、缩放、平移，不含剪切）
        # LMEDS: Least-Median 方法，对异常值更鲁棒
        M, _ = cv2.estimateAffinePartial2D(
            kps5.astype(np.float32),  # 源关键点（检测到的）
            TEMPLATE_112,            # 目标关键点（标准模板）
            method=cv2.LMEDS         # 估计方法
        )
        
        # 检查变换矩阵是否计算成功
        if M is None:
            return None
        
        # 步骤 2：应用仿射变换，将图像对齐到 112x112
        aligned = cv2.warpAffine(
            img_bgr,           # 输入图像
            M,                 # 仿射变换矩阵
            (112, 112),        # 输出图像尺寸
            flags=cv2.INTER_LINEAR,  # 线性插值
            borderValue=0     # 边界填充值（黑色）
        )
        
        return aligned

    def embedding_from_aligned(self, aligned_bgr):
        """
        从对齐后的人脸图像中提取特征向量（embedding）
        
        参数:
            aligned_bgr: numpy.ndarray, 对齐后的 112x112 BGR 图像
        
        返回:
            numpy.ndarray: 归一化后的特征向量，形状为 (512,)
                - 特征向量已经 L2 归一化（模长为 1）
                - 可以直接用于计算余弦相似度进行人脸识别
        
        流程:
            1. 将 BGR 图像转换为 RGB（ArcFace 模型需要 RGB 输入）
            2. 使用 ArcFace 模型提取特征向量
            3. L2 归一化特征向量（便于后续相似度计算）
        
        注意:
            - 特征向量是 512 维的浮点数组
            - 归一化后的特征向量可以直接用于余弦相似度计算
            - 余弦相似度 = dot(feat1, feat2) / (norm(feat1) * norm(feat2))
              由于已归一化，norm(feat) = 1，所以余弦相似度 = dot(feat1, feat2)
        """
        # 步骤 1：BGR 转 RGB（ArcFace 模型通常需要 RGB 输入）
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        
        # 步骤 2：使用 ArcFace 模型提取特征向量
        feat = self.rec_model.get_feat(aligned_rgb).astype(np.float32)
        
        # 步骤 3：L2 归一化特征向量
        # 归一化公式：feat = feat / ||feat||
        # 1e-9 是为了防止除零错误
        feat /= (np.linalg.norm(feat) + 1e-9)
        
        return feat  # 返回归一化后的 (512,) 特征向量

    @staticmethod
    def draw_kps(img_bgr, kps5, radius=3):
        """
        在图像上绘制人脸关键点（用于可视化）
        
        参数:
            img_bgr: numpy.ndarray, BGR 格式的输入图像（会被原地修改）
            kps5: numpy.ndarray 或 None, 形状为 (5, 2) 的关键点坐标数组
            radius: int, 关键点圆的半径，默认 3 像素
        
        返回:
            numpy.ndarray: 绘制了关键点的图像（与输入是同一个对象）
        
        注意:
            - 关键点用红色圆点标记（BGR 格式：(0, 0, 255) = 红色）
            - 这是一个静态方法，不依赖实例状态
            - 函数会直接修改输入的 img_bgr，不创建副本
        """
        # 如果关键点为空，直接返回原图
        if kps5 is None:
            return img_bgr
        
        # 遍历 5 个关键点，在图像上绘制红色圆点
        for (x, y) in kps5.astype(int):
            cv2.circle(
                img_bgr,                    # 目标图像
                (int(x), int(y)),           # 圆心坐标
                radius,                     # 半径
                (0, 0, 255),               # 颜色（BGR 格式：红色）
                -1                          # 填充圆（-1 表示填充）
            )
        
        return img_bgr
