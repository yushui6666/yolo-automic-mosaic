import os
import sys
import cv2
import numpy as np
import torch
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from io import BytesIO

# 设置日志级别以抑制不必要的输出
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)
logging.getLogger('insightface').setLevel(logging.WARNING)

# 抑制PyTorch警告
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# 配置日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 导入您的Python模块
# 添加当前目录到Python路径，以便导入nets模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from yolo2 import YOLO
from utils.face_gallery import FaceGallery
from utils.image_processor import ImageProcessor

# 初始化Flask应用
app = Flask(__name__)

# 配置
UPLOAD_FOLDER = 'uploads'
VIDEO_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vedio')
GALLERY_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'gallery')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png', 'gif'}

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(GALLERY_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 限制上传文件大小为500MB

# 初始化模型
yolo = None
face_gallery = None
gallery = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_models():
    """初始化YOLO和人脸识别模型"""
    global yolo, face_gallery, gallery
    
    # 抑制输出
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    # 初始化YOLO检测器（抑制输出）
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
    
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            # 从环境变量获取检测阈值参数
            detection_confidence = float(os.getenv("DETECTION_CONFIDENCE", "0.3"))
            nms_iou_threshold = float(os.getenv("NMS_IOU_THRESHOLD", "0.3"))
            
            yolo = YOLO(
                model_path=os.path.join(model_dir, 'best_epoch_weights.pth'),
                classes_path=os.path.join(model_dir, 'voc_classes.txt'),
                confidence=detection_confidence,
                nms_iou=nms_iou_threshold,
                mosaic_type="pixelate"  # 默认马赛克类型
            )
    
    # 初始化人脸识别（抑制输出）
    rec_onnx_path = os.path.join(os.path.expanduser('~'), '.insightface', 'models', 'buffalo_l', 'w600k_r50.onnx')
    if not os.path.exists(rec_onnx_path):
        rec_onnx_path = None
    
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            face_gallery = FaceGallery(yolo, rec_onnx_path)
            gallery = face_gallery.build_gallery(gallery_dir=GALLERY_FOLDER)
    
    logger.info("模型初始化完成")

@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')

@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    """上传照片到人脸库，支持命名"""
    # 检查是否有文件部分
    if 'file' not in request.files:
        return jsonify({'error': '没有文件部分'}), 400
    
    file = request.files['file']
    
    # 如果用户没有选择文件
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    # 检查文件类型
    original_filename = file.filename
    file_ext = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else ''
    
    if file_ext not in {'jpg', 'jpeg', 'png', 'gif'}:
        return jsonify({'error': '只支持图片格式 (jpg, jpeg, png, gif)'}), 400
    
    try:
        # 获取用户自定义名称
        custom_name = request.form.get('name', '').strip()
        
        # 确定最终文件名
        if custom_name:
            # 使用自定义名称，保留原始扩展名
            # 移除自定义名称中的扩展名（如果用户输入了）
            name_without_ext = custom_name.rsplit('.', 1)[0] if '.' in custom_name else custom_name
            final_filename = secure_filename(name_without_ext) + '.' + file_ext
            logger.info(f"使用自定义名称: {custom_name} -> {final_filename}")
        else:
            # 使用原始文件名
            final_filename = secure_filename(original_filename)
            logger.info(f"使用原始文件名: {final_filename}")
        
        # 保存到 gallery 文件夹
        file_path = os.path.join(GALLERY_FOLDER, final_filename)
        file.save(file_path)
        
        logger.info(f"照片已保存到: {file_path}")
        
        return jsonify({
            'success': True,
            'message': f'照片已保存到人脸库: {final_filename}',
            'filename': final_filename
        })
        
    except Exception as e:
        logger.error(f"上传照片时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'上传照片时出错: {str(e)}'}), 500

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """上传视频并处理"""
    # 检查是否有文件部分
    if 'file' not in request.files:
        return jsonify({'error': '没有文件部分'}), 400
    
    file = request.files['file']
    
    # 如果用户没有选择文件
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    # 检查文件类型
    filename = secure_filename(file.filename)
    file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    if file_ext not in {'mp4', 'avi', 'mov'}:
        return jsonify({'error': '只支持视频格式 (mp4, avi, mov)'}), 400
    
    try:
        # 保存到 vedio 文件夹
        file_path = os.path.join(VIDEO_FOLDER, filename)
        file.save(file_path)
        
        # 获取处理选项
        apply_mosaic = request.form.get('applyMosaic', 'false').lower() == 'true'
        enable_face_detection = request.form.get('faceDetection', 'false').lower() == 'true'
        mosaic_type = request.form.get('mosaicType', 'pixelate')  # 获取马赛克类型，默认为pixelate
        
        # 处理视频
        result_path = process_video(file_path, apply_mosaic, enable_face_detection, mosaic_type)
        
        # 返回处理后的视频
        result_filename = os.path.basename(result_path)
        return jsonify({
            'success': True,
            'message': '视频处理成功',
            'download_url': f'/download/{result_filename}',
            'filename': result_filename
        })
        
    except Exception as e:
        return jsonify({'error': f'处理视频时出错: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """提供处理后的视频文件下载"""
    return send_from_directory(VIDEO_FOLDER, filename)


def process_video(video_path, apply_mosaic, enable_face_detection, mosaic_type="pixelate"):
    """处理视频
    参数:
        video_path: 视频文件路径
        apply_mosaic: 是否应用马赛克
        enable_face_detection: 是否启用人脸识别
        mosaic_type: 马赛克类型，可选值: pixelate, gaussian, mean, median, color
    """
    # 初始化视频写入器
    video_capture = cv2.VideoCapture(video_path)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    
    # 创建输出视频文件名
    video_filename = f"processed_{os.path.basename(video_path)}"
    output_path = os.path.join(VIDEO_FOLDER, video_filename)
    
    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # 初始化跟踪器 (简化版)
    track_embs = {}
    track_names = {}
    track_unknown_cnt = {}
    unknown_warmup = 5
    
    frame_idx = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # 检测人脸
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        det_xyxy, det_scores, det_labels = yolo.detect_boxes(pil_img)
        
        # 过滤低置信度框
        confidence_threshold = yolo.confidence
        keep = det_scores >= confidence_threshold
        det_xyxy = det_xyxy[keep]
    
        # 为每个检测到的人脸进行处理
        for i, bbox in enumerate(det_xyxy):
            x1, y1, x2, y2 = map(int, bbox)
            bw, bh = x2 - x1, y2 - y1
            
            # 过滤过小的检测框
            if bw < 10 or bh < 10:
                continue
            
            # 跟踪ID (简化版，实际应用应使用更复杂的跟踪算法)
            track_id = f"track_{i}_{frame_idx}"
            
            # 默认用上一帧的结果
            name_text = track_names.get(track_id, "unknown")
            name = "unknown"  # 默认值
            
            if enable_face_detection and face_gallery:
                # 人脸识别
                if gallery:
                    # 每隔几帧做一次识别
                    if frame_idx % 5 == 0 or track_id not in track_embs:
                        kps5 = face_gallery.face_id.kps5_from_bbox(frame, [x1, y1, x2, y2], margin=0.35)
                        if kps5 is not None:
                            aligned = face_gallery.face_id.align_112(frame, kps5)
                            if aligned is not None:
                                emb = face_gallery.face_id.embedding_from_aligned(aligned)
                                track_embs[track_id] = emb
                                name, sim = face_gallery.recognize(emb, sim_th=0.45)
                                name_text = f"{name}({sim:.2f})"
                                name = name  # 保存名字用于马赛克判断
                                
                                # 更新unknown计数
                                if name == "unknown":
                                    track_unknown_cnt[track_id] = track_unknown_cnt.get(track_id, 0) + 1
                                else:
                                    track_unknown_cnt[track_id] = 0
                            else:
                                track_unknown_cnt[track_id] = track_unknown_cnt.get(track_id, 0) + 1
                        else:
                            track_unknown_cnt[track_id] = track_unknown_cnt.get(track_id, 0) + 1
                    else:
                        # 没有识别，使用上一帧结果
                        name, sim = name_text.split("(")
                        sim = sim.replace(")", "")
                        name_text = f"{name}({sim})"
            
            track_names[track_id] = name_text
            
            # 应用马赛克
            if apply_mosaic:
                if not enable_face_detection or name == "unknown" or track_unknown_cnt.get(track_id, 0) >= unknown_warmup:
                    # 扩展框
                    expand_ratio = 0.18
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    nw, nh = int(bw * (1 + expand_ratio)), int(bh * (1 + expand_ratio))
                    
                    ex1 = max(0, cx - nw // 2)
                    ey1 = max(0, cy - nh // 2)
                    ex2 = min(frame_width, cx + nw // 2)
                    ey2 = min(frame_height, cy + nh // 2)
                    
                    image_processor = ImageProcessor()
                    frame = image_processor.apply_mosaic(
                        frame, ex1, ey1, ex2, ey2, 
                        mosaic_type=mosaic_type,
                        block_size=yolo.mosaic_block
                    )
            
            # 在框上方绘制 ID 和识别结果
            cv2.putText(frame, f"ID {track_id.split('_')[1]}: {name_text}", 
                       (x1, max(0, y1 - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 保存处理后的帧
        out.write(frame)
    
    # 释放资源
    video_capture.release()
    out.release()
    
    return output_path

if __name__ == '__main__':
    # 初始化模型
    initialize_models()
    
    # 启动Flask应用
    app.run(debug=True, host='0.0.0.0', port=5000)
