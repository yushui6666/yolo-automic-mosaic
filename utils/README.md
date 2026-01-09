# Utils 文件夹功能介绍

本文档介绍了 `utils` 文件夹中各个模块的功能和用途。该文件夹包含了项目推理阶段所需的工具类和函数，主要用于数据处理、模型推理和人脸识别等方面。

## 文件列表

| 文件名 | 功能概述 | 状态 |
|--------|----------|------|
| [face_pi.py](face_pi.py) | 人脸关键点定位、对齐和特征提取工具类 | ✅ 使用中 |
| [face_gallery.py](face_gallery.py) | 人脸底库管理模块 | ✅ 使用中 |
| [image_processor.py](image_processor.py) | 图像处理类，包含马赛克等功能 | ✅ 使用中 |
| [utils.py](utils.py) | 通用工具函数集合 | ✅ 使用中 |
| [utils_bbox.py](utils_bbox.py) | 边界框解码和非极大值抑制（NMS） | ✅ 使用中 |
| [__init__.py](__init__.py) | Python 包初始化文件 | ✅ 使用中 |

> **注意：** 训练相关的模块（callbacks.py, dataloader.py, utils_fit.py, utils_map.py）已在本项目中删除，因为当前版本专注于推理应用。

## 详细说明

### [face_pi.py](face_pi.py)

**主要功能：**
1. 使用 InsightFace 检测器提取人脸 5 个关键点
2. 将人脸对齐到标准 112x112 尺寸
3. 使用 ArcFace 模型提取人脸特征向量（embedding）
4. 支持人脸关键点可视化

**核心类：**
- `FaceKpsAlignRec`：负责人脸关键点检测、对齐和特征提取的工具类。

**使用示例：**
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

### [face_gallery.py](face_gallery.py)

**主要功能：**
1. 构建人脸特征底库
2. 人脸识别（将输入特征与底库比对）
3. 支持同一个人多张照片的特征平均

**核心类：**
- `FaceGallery`：人脸底库管理类。

**使用示例：**
```python
from utils.face_gallery import FaceGallery
from src.yolo2 import YOLO

# 初始化
yolo = YOLO()
gallery = FaceGallery(yolo, rec_onnx_path)

# 构建底库
gallery = gallery.build_gallery(gallery_dir="gallery")

# 识别
name, sim = gallery.recognize(embedding, sim_th=0.45)
```

### [image_processor.py](image_processor.py)

**主要功能：**
1. 提供人脸马赛克脱敏功能
2. 未来可扩展更多图像处理操作

**核心类：**
- `ImageProcessor`：图像处理工具类，目前提供马赛克功能。

**使用示例：**
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

### [utils.py](utils.py)

**主要功能：**
1. 图像格式转换和预处理
2. 图像尺寸调整（支持 letterbox）
3. 类别名称读取
4. 学习率获取
5. 随机种子设置
6. 配置信息显示
7. 模型权重下载

**核心函数：**
- `cvtColor`：图像格式转换。
- `resize_image`：调整图像尺寸。
- `get_classes`：读取类别名称列表。
- `get_lr`：获取当前学习率。
- `seed_everything`：设置随机种子。
- `preprocess_input`：图像预处理。
- `show_config`：显示配置信息。
- `download_weights`：下载预训练模型权重。

### [utils_bbox.py](utils_bbox.py)

**主要功能：**
1. **DecodeBox**：将模型输出的特征图解码为边界框坐标
2. 生成锚点（anchor points）和步长（strides）
3. 距离到边界框的转换（dist2bbox）
4. 非极大值抑制（NMS）：去除重叠的检测框
5. 坐标格式转换：支持 xyxy 和 xywh 格式

**核心类：**
- `DecodeBox`：用于解码模型输出的边界框坐标，并应用非极大值抑制。

**使用示例：**
```python
from utils.utils_bbox import DecodeBox

# 初始化解码器
bbox_util = DecodeBox(num_classes, input_shape)

# 解码边界框
outputs = bbox_util.decode_box(inputs)

# 应用 NMS
results = bbox_util.non_max_suppression(
    outputs, 
    num_classes, 
    input_shape, 
    image_shape, 
    letterbox_image,
    conf_thres=0.5, 
    nms_thres=0.4
)
```

## 使用说明

1. **YOLO 检测**：使用 `utils.py` 和 `utils_bbox.py` 进行图像预处理和边界框解码。
2. **人脸处理**：使用 `face_pi.py` 进行人脸关键点检测、对齐和特征提取。
3. **图像处理**：使用 `image_processor.py` 进行马赛克等图像处理操作。
4. **人脸识别**：使用 `face_gallery.py` 进行人脸底库管理和识别。

## 模块依赖关系

```
src/yolo2.py
    ├── utils/utils.py          # 图像预处理
    └── utils/utils_bbox.py     # 边界框解码

src/app.py, src/pre2.py
    ├── src/yolo2.py
    ├── utils/face_gallery.py   # 人脸底库管理
    │   └── utils/face_pi.py   # 人脸特征提取
    └── utils/image_processor.py # 图像处理（马赛克）
```

## 注意事项

1. 确保所有依赖库（如 OpenCV、PyTorch、NumPy 等）已正确安装。
2. 在使用人脸相关功能时，需要确保 InsightFace 模型已下载并正确配置路径。
3. 首次使用 InsightFace 时，模型会自动下载到 `~/.insightface/models/` 目录。
4. 确保人脸底库图片格式正确，且图片中的人脸能够被 YOLO 检测到。

## 版本历史

### v2.0 (当前版本)
- ✅ 删除训练相关模块（callbacks.py, dataloader.py, utils_fit.py, utils_map.py）
- ✅ 专注于推理应用的核心功能
- ✅ 优化文档结构，清晰标注模块状态

## 贡献

如果您有任何建议或发现 bug，请提交 issue 或 pull request。

## 许可证

请参考项目根目录下的 LICENSE 文件。
