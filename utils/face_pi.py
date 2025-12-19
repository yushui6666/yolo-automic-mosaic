import cv2
import numpy as np

# ArcFace 常用 112x112 模板 5 点
TEMPLATE_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

class FaceKpsAlignRec:
    """
    用 InsightFace 的 detector 给 ROI 输出 5 点，然后对齐到 112x112，
    再用 ArcFace(ONNX) 出 embedding。
    """
    def __init__(self,
                 det_size=(640, 640),
                 ctx_id=0,
                 providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
                 rec_onnx_path=None):
        from insightface.app import FaceAnalysis
        from insightface.model_zoo import get_model

        # A) detector：只用 detection（一般会给 bbox + 5点kps）
        self.det_app = FaceAnalysis(
            allowed_modules=["detection"],
            providers=list(providers)
        )
        self.det_app.prepare(ctx_id=ctx_id, det_size=det_size)

        # B) recognition：ArcFace embedding 模型（比如 buffalo_l 里的 w600k_r50.onnx）
        if rec_onnx_path is None:
            raise ValueError("rec_onnx_path 不能为空，例如 ...\\.insightface\\models\\buffalo_l\\w600k_r50.onnx")

        self.rec_model = get_model(rec_onnx_path, providers=list(providers))
        self.rec_model.prepare(ctx_id=ctx_id)

    @staticmethod
    def _clip_roi(img_bgr, bbox_xyxy, margin=0.2):
        x1, y1, x2, y2 = map(int, bbox_xyxy)
        h, w = img_bgr.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 10:
            return None

        m = int(margin * max(bw, bh))
        xx1 = max(0, x1 - m); yy1 = max(0, y1 - m)
        xx2 = min(w, x2 + m); yy2 = min(h, y2 + m)
        roi = img_bgr[yy1:yy2, xx1:xx2]
        if roi.size == 0:
            return None
        return (roi, xx1, yy1)

    def kps5_from_bbox(self, img_bgr, bbox_xyxy, margin=0.2):
        pack = self._clip_roi(img_bgr, bbox_xyxy, margin=margin)
        if pack is None:
            return None
        roi, offx, offy = pack

        faces = self.det_app.get(roi)
        if len(faces) == 0:
            return None
        kps = getattr(faces[0], "kps", None)
        if kps is None:
            return None

        kps5 = kps.astype(np.float32)
        kps5[:, 0] += offx
        kps5[:, 1] += offy
        return kps5  # (5,2) in 原图坐标系

    @staticmethod
    def align_112(img_bgr, kps5):
        M, _ = cv2.estimateAffinePartial2D(
            kps5.astype(np.float32),
            TEMPLATE_112,
            method=cv2.LMEDS
        )
        if M is None:
            return None
        aligned = cv2.warpAffine(img_bgr, M, (112, 112),
                                 flags=cv2.INTER_LINEAR, borderValue=0)
        return aligned

    def embedding_from_aligned(self, aligned_bgr):
        # rec_model 一般吃 RGB
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        feat = self.rec_model.get_feat(aligned_rgb).astype(np.float32)
        feat /= (np.linalg.norm(feat) + 1e-9)
        return feat  # (512,) normalize

    @staticmethod
    def draw_kps(img_bgr, kps5, radius=3):
        if kps5 is None:
            return img_bgr
        for (x, y) in kps5.astype(int):
            cv2.circle(img_bgr, (int(x), int(y)), radius, (0, 0, 255), -1)
        return img_bgr
