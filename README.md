# YOLO 人脸检测与识别系统

一个基于 YOLO 和 InsightFace 的人脸检测、识别和隐私保护系统。支持视频流实时处理、人脸底库管理和自动马赛克脱敏。

## ✨ 主要功能

- 🎯 **YOLO 人脸检测**：基于 YOLO 网络的高精度人脸检测
- 🧠 **人脸识别**：使用 InsightFace/ArcFace 进行人脸特征提取和识别
- 👥 **人脸底库**：支持构建和管理人脸特征库
- 🎭 **马赛克脱敏**：对未知人脸自动打码，保护隐私
- 🎥 **视频处理**：支持实时视频流和视频文件处理
- 🌐 **Web 应用**：提供 Flask Web 界面，支持照片和视频上传
- 📊 **目标跟踪**：基于 IoU 的轻量级跟踪算法

## 📁 项目结构

```
project/
├── nets/                   # 神经网络定义
│   ├── backbone.py        # 主干网络
│   └── yolo.py           # YOLO 网络结构
├── utils/                 # 工具模块
│   ├── face_gallery.py    # 人脸底库管理
│   ├── face_pi.py         # 人脸关键点、对齐、特征提取
│   ├── image_processor.py # 图像处理（马赛克）
│   ├── utils.py          # 通用工具函数
│   └── utils_bbox.py     # 边界框解码和 NMS
├── src/                  # 源代码
│   ├── app.py            # Flask Web 应用
│   ├── pre2.py           # 视频处理脚本
│   └── yolo2.py          # YOLO 检测器封装
├── model/                # 模型文件
│   └── voc_classes.txt   # 类别名称
├── gallery/              # 人脸底库照片目录
├── vedio/               # 视频文件目录
├── uploads/             # 上传文件临时目录
└── requirements.txt      # Python 依赖
```

## 🚀 快速开始

### 环境要求

- Python 3.7+
- CUDA 11.0+（可选，用于 GPU 加速）
- PyTorch 1.10+

### 安装依赖

```bash
pip install -r requirements.txt
```

### 模型准备

1. **下载 YOLO 模型权重**
   - 将训练好的 `best_epoch_weights.pth` 放入 `model/` 目录
   - 确保 `model/voc_classes.txt` 包含正确的类别名称

2. **下载 InsightFace 模型**
   ```bash
   # 首次运行会自动下载到 ~/.insightface/models/
   # 或手动下载 w600k_r50.onnx 到该目录
   ```

3. **准备人脸底库**
   - 将人脸照片放入 `gallery/` 目录
   - 文件命名格式：`姓名_序号.jpg`（如：`zhangsan_1.jpg`、`zhangsan_2.jpg`）

## 💡 使用方法

### 1. Web 应用

启动 Flask Web 服务：

```bash
python src/app.py
```

访问 `http://localhost:5000` 使用 Web 界面。

**功能：**
- 上传照片到人脸底库
- 上传视频进行处理
- 配置是否应用马赛克和人脸识别

### 2. 视频处理脚本

直接运行视频处理脚本：

```bash
python src/pre2.py
```

**配置参数：**
```python
enable_mosaic = False  # 是否启用马赛克（True/False）
video_path = 'vedio/man.mp4'  # 输入视频路径
video_save_path = "vedio/man_out.mp4"  # 输出视频路径
rec_onnx_path = r"C:\Users\用户名\.insightface\models\buffalo_l\w600k_r50.onnx"  # 模型路径
```

### 3. 单张图片检测

```python
from yolo2 import YOLO

# 初始化检测器
yolo = YOLO(
    model_path='model/best_epoch_weights.pth',
    classes_path='model/voc_classes.txt',
    confidence=0.3,
    nms_iou=0.3
)

# 检测图片
from PIL import Image
image = Image.open("test.jpg")
result = yolo.detect_image(image)  # 返回打码后的图片
result.save("output.jpg")
```

### 4. 获取检测框（用于自定义处理）

```python
from yolo2 import YOLO
from PIL import Image

yolo = YOLO()
image = Image.open("test.jpg")

# 获取检测框坐标、置信度和类别
det_xyxy, det_scores, det_labels = yolo.detect_boxes(image)

# det_xyxy: [[x1, y1, x2, y2], ...]  边界框坐标
# det_scores: [0.95, 0.87, ...]       置信度
# det_labels: [0, 0, ...]             类别索引
```

## 🔧 核心模块说明

### YOLO 检测器 (`src/yolo2.py`)

封装了 YOLO 模型的推理功能，提供多个接口：

- `detect_image(image)` - 检测并打码单张图片
- `detect_boxes(image)` - 获取检测框信息
- `get_FPS(image, test_interval)` - 测试推理速度

**主要参数：**
- `confidence`: 置信度阈值（默认 0.3）
- `nms_iou`: NMS 阈值（默认 0.3）
- `input_shape`: 输入图像尺寸（默认 640x640）
- `mosaic_block`: 马赛克块大小（默认 14）
- `expand_ratio`: 扩框比例（默认 0.2）

### 人脸识别模块

#### `utils/face_pi.py` - 人脸处理工具类

```python
from utils.face_pi import FaceKpsAlignRec

# 初始化
face_id = FaceKpsAlignRec(
    det_size=(640, 640),
    rec_onnx_path="path/to/w600k_r50.onnx"
)

# 提取关键点
kps5 = face_id.kps5_from_bbox(img_bgr, bbox, margin=0.35)

# 人脸对齐
aligned = face_id.align_112(img_bgr, kps5)

# 提取特征
embedding = face_id.embedding_from_aligned(aligned)
```

#### `utils/face_gallery.py` - 人脸底库管理

```python
from utils.face_gallery import FaceGallery
from yolo2 import YOLO

yolo = YOLO()
gallery = FaceGallery(yolo, rec_onnx_path)

# 构建底库
gallery.build_gallery(gallery_dir="gallery")

# 识别
name, sim = gallery.recognize(embedding, sim_th=0.45)
```

### 图像处理模块 (`utils/image_processor.py`)

```python
from utils.image_processor import ImageProcessor
import cv2

processor = ImageProcessor()
img = cv2.imread("test.jpg")

# 应用马赛克
img = processor.apply_mosaic(
    img, 
    left=100, top=100, right=200, bottom=200, 
    block=14
)
```

## 📊 性能优化

### GPU 加速

确保安装 CUDA 版本的 PyTorch：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 推理速度

- 使用 `get_FPS()` 方法测试性能
- 调整 `input_shape` 平衡精度和速度
- 使用 `letterbox_image=False` 可提升速度（但可能降低精度）

### 人脸识别优化

- 调整 `rec_interval` 参数控制识别频率
- 使用特征缓存减少重复计算
- 调整 `sim_th` 阈值平衡识别准确率和误识别率

## 🎨 配置说明

### 马赛克配置

```python
# 在 YOLO 类中配置
yolo.mosaic_block = 14      # 马赛克块大小（越大越模糊）
yolo.expand_ratio = 0.2     # 扩框比例
yolo.min_face_size = 10     # 最小人脸尺寸
```

### 识别配置

```python
# 相似度阈值
sim_th = 0.45  # 越高越严格

# 识别稳定性
unknown_warmup = 5  # 连续多少帧 unknown 才真正当陌生人
num = 2            # 连续正确识别多少帧才算稳定
```

## 🐛 常见问题

### 1. 模型加载失败

**问题：** `FileNotFoundError: model/best_epoch_weights.pth not found`

**解决：**
- 确保模型文件在正确路径
- 检查文件名是否正确

### 2. InsightFace 模型未找到

**问题：** `rec_onnx_path cannot be None`

**解决：**
- 首次运行会自动下载模型到 `~/.insightface/models/`
- 检查 `w600k_r50.onnx` 是否存在
- 手动指定正确的模型路径

### 3. 人脸底库为空

**问题：** 人脸识别全部返回 "unknown"

**解决：**
- 确认 `gallery/` 目录下有照片
- 检查照片中的人脸能否被检测到
- 调整 `sim_th` 阈值

### 4. 检测不到人脸

**问题：** 所有检测框置信度都低于阈值

**解决：**
- 降低 `confidence` 阈值（如改为 0.2）
- 检查模型是否训练好
- 确认输入图片清晰度和光照条件

## 📝 更新日志

### v2.0 (当前版本)
- ✅ 删除训练相关模块（callbacks, dataloader, utils_fit, utils_map）
- ✅ 优化项目结构，专注于推理应用
- ✅ 完善 Web 应用功能
- ✅ 添加视频处理支持
- ✅ 集成人脸识别和底库管理

## 📄 许可证

本项目采用 MIT 许可证。

## 🙏 致谢

- YOLO: Ultralytics YOLO implementation
- InsightFace: DeepInsight Face Analysis Toolkit
- PyTorch: Open source machine learning framework
