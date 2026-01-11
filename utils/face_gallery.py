#-----------------------------------------------------------------------#
#   face_gallery.py：人脸库模块（独立）
#   功能概述：
#   1. 加载人脸底库照片，提取特征存入内存。
#   2. 提供查询接口，将输入人脸特征与底库比对。
#   3. 支持按需启用或禁用人脸识别功能。
#-----------------------------------------------------------------------#
import os
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict
import logging

from utils.face_pi import FaceKpsAlignRec  # 人脸关键点定位、对齐、特征提取
from src.yolo2 import YOLO                   # YOLO 检测

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FaceGallery:
    """
    人脸底库管理类
    
    主要功能：
    1. 构建人脸底库（加载底库照片，提取并存储特征）
    2. 人脸识别（将输入特征与底库比对，返回最相似的身份）
    
    使用示例：
        gallery = FaceGallery(yolo_model, rec_model_path)
        gallery.build_gallery(gallery_dir="known_faces")  # 构建底库
        name, sim = gallery.recognize(face_feature)        # 识别身份
    """
    
    def __init__(self, yolo_model, rec_model_path):
        """
        初始化人脸底库
        
        参数:
            yolo_model: YOLO instance, 已初始化的 YOLO 检测器
            rec_model_path: str, ArcFace 模型的 ONNX 文件路径
        """
        self.yolo = yolo_model
        self.face_id = FaceKpsAlignRec(
            det_size=(640, 640),
            ctx_id=0,
            providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
            rec_onnx_path=rec_model_path,
        )
        self.gallery = {}  # 存储底库特征：{name: mean_embedding}
        
    def build_gallery(self, gallery_dir="gallery"):
        """
        构建人脸特征底库
        
        参数:
            gallery_dir: str, 底库照片目录路径
        
        返回:
            dict: 构建好的底库字典 {name: mean_embedding}
        
        流程:
            1. 遍历目录下的图片
            2. 使用 YOLO 检测人脸
            3. 提取关键点并对齐
            4. 提取特征向量
            5. 对同一人多张图片的特征取均值
        """
        # 临时字典：id_name -> [emb1, emb2, ...]
        tmp = defaultdict(list)
        
        if not os.path.isdir(gallery_dir):
            logger.warning(f"gallery 目录不存在: {gallery_dir}")
            return {}
        
        for fn in os.listdir(gallery_dir):
            path = os.path.join(gallery_dir, fn)
            if not os.path.isfile(path):
                continue
            
            name, ext = os.path.splitext(fn)
            # 解析 ID，例如文件名 "shui_2.jpg" -> id_name="shui"
            id_name = name.split('_')[0]
            
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                logger.warning(f"无法读取图片: {path}")
                continue
            
            # 1) YOLO 检测这张底库图片中的人脸，确保提取的是人脸区域
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            det_xyxy, det_scores, det_labels = self.yolo.detect_boxes(pil_img)
            
            if det_xyxy is None or len(det_xyxy) == 0:
                logger.warning(f"{path} 未检测到人脸（YOLO），跳过")
                continue
            
            # 取置信度最高的一个人脸
            best_idx = int(det_scores.argmax())
            bbox = det_xyxy[best_idx]
            
            # 2) 计算人脸关键点
            kps5 = self.face_id.kps5_from_bbox(img_bgr, bbox, margin=0.35)
            if kps5 is None:
                logger.warning(f"{path} 未得到关键点")
                continue
            
            # 3) 矫正对齐
            aligned = self.face_id.align_112(img_bgr, kps5)
            if aligned is None:
                logger.warning(f"{path} 对齐失败")
                continue
            
            # 4) 提取特征向量
            emb = self.face_id.embedding_from_aligned(aligned)
            emb = np.asarray(emb, dtype=np.float32).reshape(-1)
            tmp[id_name].append(emb)
            logger.info(f"[gallery] 加载 {id_name}, emb shape={emb.shape}, from {fn}")
        
        # 5) 对同一个人多张图片的 embedding 求平均
        self.gallery = {}
        for id_name, embs in tmp.items():
            embs = np.stack(embs, axis=0)  # [N, 512]
            mean_emb = embs.mean(axis=0)   # [512]
            self.gallery[id_name] = mean_emb
            logger.info(f"[gallery] {id_name} 最终使用 {len(embs)} 张图片, mean_emb shape={mean_emb.shape}")
        
        return self.gallery
    
    def recognize(self, emb, sim_th=0.45):
        """
        在底库中寻找与输入特征最相似的人
        
        参数:
            emb: numpy.ndarray, 输入人脸特征向量 (512,)
            sim_th: float, 相似度阈值，低于此值返回 "unknown"
        
        返回:
            tuple: (best_name, best_sim)
                - best_name: str, 最佳匹配的名字，若低于阈值则为 "unknown"
                - best_sim: float, 最佳相似度 (0-1)
        """
        if not self.gallery:
            return "unknown", 0.0
        
        best_name = "unknown"
        best_sim = 0.0
        
        # 遍历底库中每个人
        for name, gemb in self.gallery.items():
            sim = cosine_sim(emb, gemb)
            if sim > best_sim:
                best_sim = sim
                best_name = name
        
        # 阈值判断
        if best_sim < sim_th:
            return "unknown", best_sim
        return best_name, best_sim


def cosine_sim(a, b):
    """计算余弦相似度，范围 [-1, 1]，通常人脸越相似值越高"""
    # 转成 float32 一维向量，避免形状不对齐
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na = np.linalg.norm(a) + 1e-6
    nb = np.linalg.norm(b) + 1e-6
    return float(np.dot(a, b) / (na * nb))
