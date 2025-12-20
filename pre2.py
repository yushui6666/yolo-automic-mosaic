#-----------------------------------------------------------------------#
#   pre2.py：视频中用 YOLO 检测人脸 + 跟踪 + 人脸识别（底库）+ 未知人打码
#   功能概述：
#   1. 加载人脸底库照片，提取特征存入内存。
#   2. 读取视频，对每一帧进行 YOLO 人脸检测。
#   3. 使用 SimpleTracker 对检测框进行 ID 跟踪。
#   4. 对跟踪到的人脸提取特征，与底库比对（计算余弦相似度）。
#   5. 如果相似度低于阈值，判定为 unknown，并对该区域打马赛克。
#-----------------------------------------------------------------------#
import os
import time

# 设置 ONNX Runtime 的日志级别，减少不必要的控制台输出
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"  # 0-4: VERBOSE, INFO, WARNING, ERROR, FATAL
os.environ["ORT_LOG_VERBOSITY_LEVEL"] = "0"  # 0 最安静

import cv2
import numpy as np
from PIL import Image
from collections import defaultdict

# 导入自定义工具类
from utils.face_pi import FaceKpsAlignRec  # 负责人脸关键点定位、对齐、特征提取
from yolo2 import YOLO, mosaic_area        # 负责 YOLO 检测和马赛克绘制


def iou_xyxy(a, b):
    """
    计算两个矩形框 (x1, y1, x2, y2) 的交并比 (IoU)。
    用于跟踪算法中判断当前帧的检测框与上一帧的轨迹框是否重合。
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    # 计算交集区域坐标
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    # 计算各自面积
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, ay2 - ay1)
    # 返回 IoU
    return inter / (area_a + area_b - inter + 1e-9)


class Track:
    """
    单个轨迹对象，存储 ID、当前框位置以及丢失计数。
    """
    def __init__(self, tid, bbox_xyxy):
        self.id = tid
        self.bbox = bbox_xyxy.astype(float)
        self.miss = 0  # 记录该目标连续多少帧未被检测到


class SimpleTracker:
    """
    轻量级跟踪器：
    策略：基于 IoU 进行匹配，使用 EMA (指数移动平均) 平滑框的抖动。
    """
    def __init__(self, iou_th=0.15, max_miss=2, ema_alpha=0.6):
        self.iou_th = iou_th       # 匹配阈值，IoU大于此值才认为是同一个目标
        self.max_miss = max_miss   # 最大容忍丢失帧数，超过则删除轨迹
        self.ema_alpha = ema_alpha # 平滑系数，越大越接近当前检测框，越小越平滑但滞后
        self.tracks = []           # 存储当前的轨迹列表
        self.next_id = 1           # 下一个分配的 ID

    def update(self, dets_xyxy):
        """
        输入当前帧的检测框，更新内部轨迹状态，返回更新后的 (ID, bbox)。
        """
        dets = dets_xyxy.astype(float)
        assigned = set()  # 记录已经被匹配的检测框索引

        # --- 步骤 1: 给现有的 track 匹配最佳的检测框 (贪心匹配) ---
        for tr in self.tracks:
            best_j, best_iou = -1, 0.0
            for j, d in enumerate(dets):
                if j in assigned:
                    continue
                v = iou_xyxy(tr.bbox, d)
                if v > best_iou:
                    best_iou, best_j = v, j

            # 如果找到了匹配且 IoU 大于阈值
            if best_j >= 0 and best_iou >= self.iou_th:
                d = dets[best_j]
                a = self.ema_alpha
                # EMA 平滑公式: current = alpha * new + (1-alpha) * old
                tr.bbox = a * d + (1 - a) * tr.bbox
                tr.miss = 0
                assigned.add(best_j)
            else:
                # 没匹配上，miss 计数加 1
                tr.miss += 1

        # --- 步骤 2: 没有匹配到的检测框，创建新轨迹 ---
        for j, d in enumerate(dets):
            if j not in assigned:
                self.tracks.append(Track(self.next_id, d))
                self.next_id += 1

        # --- 步骤 3: 清理丢失太久 (miss > max_miss) 的轨迹 ---
        self.tracks = [t for t in self.tracks if t.miss <= self.max_miss]

        # 只返回当前这一帧有成功匹配到检测框的轨迹 (miss == 0)，避免显示预测的虚影
        return [(t.id, t.bbox.copy()) for t in self.tracks if t.miss == 0]


# ---------------- 人脸识别相关工具函数（底库 + 相似度） ----------------

def cosine_sim(a, b):
    """计算余弦相似度，范围 [-1, 1]，通常人脸越相似值越高"""
    # 转成 float32 一维向量，避免形状不对齐 (如 (1,512) vs (512,))
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na = np.linalg.norm(a) + 1e-6
    nb = np.linalg.norm(b) + 1e-6
    return float(np.dot(a, b) / (na * nb))


def recognize_from_gallery(emb, gallery, sim_th=0.45):
    """
    在底库 gallery 中寻找与当前 embedding 最相似的人。
    返回: (最佳匹配名字, 相似度数值)
    如果最高相似度低于 sim_th，则返回 "unknown"。
    """
    if not gallery:
        return "unknown", 0.0
    best_name = "unknown"
    best_sim = 0.0
    
    # 遍历底库中每个人
    for name, gemb in gallery.items():
        sim = cosine_sim(emb, gemb)
        if sim > best_sim:
            best_sim = sim
            best_name = name
            
    # 阈值判断
    if best_sim < sim_th:
        return "unknown", best_sim
    return best_name, best_sim


def build_gallery(face_id, yolo, gallery_dir="gallery"):
    """
    构建人脸特征底库。
    逻辑：
    1. 遍历 gallery_dir 目录下的图片。
    2. 使用 YOLO 再次检测图片中的人脸（防止底库图包含背景）。
    3. 提取特征向量 (Embedding)。
    4. 支持同一人多张图片（通过文件名下划线区分，如 user_1.jpg, user_2.jpg），计算特征均值。
    """
    # 临时字典：id_name -> [emb1, emb2, ...]
    tmp = defaultdict(list)

    if not os.path.isdir(gallery_dir):
        print(f"[warn] gallery 目录不存在: {gallery_dir}")
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
            print(f"[warn] 无法读取图片: {path}")
            continue

        # 1) YOLO 检测这张底库图片中的人脸，确保提取的是人脸区域
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        det_xyxy, det_scores, det_labels = yolo.detect_boxes(pil_img)

        if det_xyxy is None or len(det_xyxy) == 0:
            print(f"[warn] {path} 未检测到人脸（YOLO），跳过")
            continue

        # 取置信度最高的一个人脸
        best_idx = int(det_scores.argmax())
        bbox = det_xyxy[best_idx]

        # 2) 计算人脸关键点 (margin 0.35 表示扩充一点框，有助于关键点定位准确)
        kps5 = face_id.kps5_from_bbox(img_bgr, bbox, margin=0.35)
        if kps5 is None:
            print(f"[warn] {path} 未得到关键点")
            continue

        # 3) 矫正对齐 (Affine Transform) 到 112x112 标准尺寸
        aligned = face_id.align_112(img_bgr, kps5)
        if aligned is None:
            print(f"[warn] {path} 对齐失败")
            continue

        # 4) 提取特征向量 (通常是 512 维)
        emb = face_id.embedding_from_aligned(aligned)
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        tmp[id_name].append(emb)
        print(f"[gallery] 加载 {id_name}, emb shape={emb.shape}, from {fn}")

    # 5) 对同一个人多张图片的 embedding 求平均，增强底库鲁棒性
    gallery = {}
    for id_name, embs in tmp.items():
        embs = np.stack(embs, axis=0)  # [N, 512]
        mean_emb = embs.mean(axis=0)   # [512]
        gallery[id_name] = mean_emb
        print(f"[gallery] {id_name} 最终使用 {len(embs)} 张图片, mean_emb shape={mean_emb.shape}")

    return gallery


# ------------------------------- 主程序入口 -------------------------------

if __name__ == "__main__":
    # 初始化 YOLO 检测器
    yolo = YOLO()

    mode = "video" # 当前模式：处理视频文件

    # 输入视频路径和输出保存路径
    video_path = 'vedio/man.mp4'
    video_save_path = "vedio/man_out.mp4"
    video_fps = 60  # 保存视频时的帧率

    if mode == "video":
        capture = cv2.VideoCapture(video_path)

        # 初始化视频写入器
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取视频，请检查视频路径。")

        # 初始化跟踪器
        tracker = SimpleTracker(iou_th=0.3, max_miss=2, ema_alpha=0.6)

        # 初始化人脸特征提取模型 (需确保 rec_onnx_path 路径下的模型文件存在)
        # 注意：这里的路径是绝对路径，换电脑运行需要修改！
        face_id = FaceKpsAlignRec(
            det_size=(640, 640),
            ctx_id=0,  # GPU索引，0表示第一块显卡；若无 GPU 需检查 ONNXRuntime 是否为 CPU 版
            providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
            rec_onnx_path=r"C:\Users\34239\.insightface\models\buffalo_l\w600k_r50.onnx",
        )

        # 构建底库：读取 ./gallery 文件夹
        gallery = build_gallery(face_id, yolo, gallery_dir="gallery")

        fps = 0.0
        frame_idx = 0
        rec_interval = 1           # 优化参数：每隔几帧做一次特征提取？1表示每帧都做，设大可提速
        track_embs = {}            # 缓存字典：track_id -> 特征向量
        track_names = {}           # 缓存字典：track_id -> 显示的名字(含相似度)
        num = 2                    # 新增：同一ID连续正确识别超过 num 帧后，短暂失败也不变 unknown
        track_stable_cnt = {}      # 新增：track_id -> 连续正确识别帧数
        unknown_warmup = 5         # 新增：同一 ID 连续 unknown 达到这个帧数才真正当陌生人
        track_unknown_cnt = {}     # 新增：track_id -> 连续 unknown 帧数
        print("开始视频处理循环...")
        while True:
            t1 = time.time()

            ref, frame_bgr = capture.read()
            if not ref:
                break

            frame_idx += 1
            h, w = frame_bgr.shape[:2]

            # 格式转换：BGR (OpenCV) -> RGB -> PIL (YOLO 需要 PIL 格式)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(np.uint8(frame_rgb))

            # --- 1) YOLO 检测人脸框 ---
            det_xyxy, det_scores, det_labels = yolo.detect_boxes(pil_img)

            # --- 2) 过滤低置信度框 (score < 0.4) ---
            keep = det_scores >= 0.4
            det_xyxy = det_xyxy[keep]

            # --- 3) 跟踪模块：获取当前帧稳定的 Track ---
            tracks = tracker.update(det_xyxy)

            # 遍历每一个跟踪到的人脸
            for tid, bb in tracks:
                x1, y1, x2, y2 = bb
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                bw, bh = x2 - x1, y2 - y1
                
                # 过滤掉过小的误检框 (小于 10x10)
                if bw < 10 or bh < 10:
                    continue

                # 绘制人脸框 (绿色)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 判断是否需要进行人脸识别：
                # 1. 该 ID 还没识别过 (不在 track_embs 里)
                # 2. 或者到了定时的帧数 (rec_interval)
                need_rec = (tid not in track_embs) or (frame_idx % rec_interval == 0)

                # 默认用上一帧的结果
                name_text = track_names.get(tid, "...")

                if need_rec:
                    # --- 4) 人脸识别流程 ---
                    kps5 = face_id.kps5_from_bbox(frame_bgr, bb, margin=0.35)
                    aligned = face_id.align_112(frame_bgr, kps5) if kps5 is not None else None

                    # 本帧的候选识别结果
                    if aligned is not None:
                        emb = face_id.embedding_from_aligned(aligned)
                        track_embs[tid] = emb
                        name, sim = recognize_from_gallery(
                            emb, gallery, sim_th=0.3  # 相似度阈值
                        )
                        candidate_text = f"{name}({sim:.2f})"
                    else:
                        name, sim = "unknown", 0.0
                        candidate_text = "unknown"

                    prev_text = track_names.get(tid, "unknown")
                    prev_id = prev_text.split("(")[0] if not prev_text.startswith("unknown") else None
                    cur_id = name if name != "unknown" else None

                    stable_cnt = track_stable_cnt.get(tid, 0)
                    unknown_cnt = track_unknown_cnt.get(tid, 0)

                    if cur_id is not None:
                        # 本帧识别到了已知人
                        if cur_id == prev_id:
                            stable_cnt += 1
                        else:
                            stable_cnt = 1  # 换了一个新的人
                        track_stable_cnt[tid] = stable_cnt

                        unknown_cnt = 0   # 既然是已知人，就把 unknown 计数清零
                        track_unknown_cnt[tid] = unknown_cnt

                        name_text = candidate_text
                    else:
                        # 本帧结果是 unknown
                        if stable_cnt >= num and prev_id is not None:
                            # 已连续正确识别过 num 帧，短暂识别失败时继续用之前结果
                            name_text = prev_text
                            unknown_cnt = 0  # 虽然这帧算失败，但仍当作已知人的短暂抖动
                        else:
                            # 还没稳定识别成功过，确实是 unknown，累计 unknown 帧数
                            name_text = candidate_text
                            stable_cnt = 0
                            unknown_cnt += 1

                        track_stable_cnt[tid] = stable_cnt
                        track_unknown_cnt[tid] = unknown_cnt

                    track_names[tid] = name_text

                # 在框上方绘制 ID 和 识别结果
                cv2.putText(
                    frame_bgr,
                    f"ID {tid}: {name_text}",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

                # --- 5) 隐私打码逻辑 ---
                # 只有当识别结果是 unknown 且连续 unknown 帧数 >= unknown_warmup 才打码
                unk_cnt = track_unknown_cnt.get(tid, 0)
                if name_text.startswith("unknown") and unk_cnt >= unknown_warmup:
                    expand_ratio = 0.18
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    nw, nh = int(bw * (1 + expand_ratio)), int(bh * (1 + expand_ratio))

                    ex1 = max(0, cx - nw // 2)
                    ey1 = max(0, cy - nh // 2)
                    ex2 = min(w, cx + nw // 2)
                    ey2 = min(h, cy + nh // 2)

                    frame_bgr = mosaic_area(
                        frame_bgr,
                        ex1,
                        ey1,
                        ex2,
                        ey2,
                        block=yolo.mosaic_block,
                    )

            # 计算并显示 FPS
            fps = (fps + (1.0 / (time.time() - t1))) / 2
            frame_bgr = cv2.putText(
                frame_bgr,
                f"fps= {fps:.2f}",
                (0, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # 显示当前帧
            cv2.imshow("video", frame_bgr)

            # 写入视频文件
            if video_save_path != "":
                out.write(frame_bgr)

            # 按 ESC 退出
            c = cv2.waitKey(1) & 0xFF
            if c == 27:
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps', 'dir_predict'.")