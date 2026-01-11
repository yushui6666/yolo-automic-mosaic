# 项目参数文档

本文档详细说明了整个项目中使用的所有参数，包括它们的作用、调用位置以及使用效果。

---

## 目录

1. [Flask应用参数 (src/app.py)](#flask应用参数)
2. [YOLO检测器参数 (src/yolo2.py)](#yolo检测器参数)
3. [视频处理参数 (src/pre2.py)](#视频处理参数)
4. [人脸库参数 (utils/face_gallery.py)](#人脸库参数)
5. [图像处理参数 (utils/image_processor.py)](#图像处理参数)
6. [YOLO网络参数 (nets/yolo.py)](#yolo网络参数)

---

## Flask应用参数

### 文件路径配置
| 参数名 | 类型 | 默认值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `UPLOAD_FOLDER` | str | 'uploads' | app.py 第26行 | 存储临时上传文件的目录路径 |
| `VIDEO_FOLDER` | str | 项目根目录/vedio | app.py 第27行 | 存储输入和处理后视频的目录路径 |
| `GALLERY_FOLDER` | str | 项目根目录/gallery | app.py 第28行 | 存储人脸底库照片的目录路径 |

### 文件类型限制
| 参数名 | 类型 | 默认值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `ALLOWED_EXTENSIONS` | set | {'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png', 'gif'} | app.py 第30行 | 允许上传的文件扩展名集合，用于文件类型验证 |

### 服务器配置
| 参数名 | 类型 | 默认值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `MAX_CONTENT_LENGTH` | int | 500 * 1024 * 1024 (500MB) | app.py 第36行 | 限制上传文件的最大大小，超过此大小的文件将被拒绝 |

### 服务器运行参数
| 参数名 | 类型 | 默认值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `debug` | bool | True | app.py 最后一行 | Flask调试模式，True时显示详细错误信息并自动重启 |
| `host` | str | '0.0.0.0' | app.py 最后一行 | 服务器监听地址，'0.0.0.0'表示监听所有网络接口 |
| `port` | int | 5000 | app.py 最后一行 | 服务器监听端口 |

### YOLO初始化参数（在Flask中调用）
| 参数名 | 类型 | 默认值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `model_path` | str | model/best_epoch_weights.pth | app.py 第55行 | YOLO模型权重文件路径 |
| `classes_path` | str | model/voc_classes.txt | app.py 第56行 | 类别名称文件路径 |
| `confidence` | float | 0.3 | app.py 第57行 | 置信度阈值，低于此值的检测框将被过滤 |
| `nms_iou` | float | 0.3 | app.py 第58行 | NMS（非极大值抑制）的IoU阈值，用于去除重叠框 |
| `mosaic_type` | str | "pixelate" | app.py 第59行 | 马赛克类型，可选：pixelate, gaussian, mean, median, color |

### 视频处理参数（process_video函数）
| 参数名 | 类型 | 默认值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `apply_mosaic` | bool | 从前端获取 | app.py 第90行 | 是否对视频应用马赛克脱敏处理 |
| `enable_face_detection` | bool | 从前端获取 | app.py 第91行 | 是否启用人脸识别功能 |
| `mosaic_type` | str | 从前端获取，默认'pixelate' | app.py 第92行 | 马赛克类型 |
| `expand_ratio` | float | 0.18 | app.py 第182行 | 检测框扩展比例，防止人脸边缘漏遮 |
| `unknown_warmup` | int | 5 | app.py 第172行 | 连续多少帧识别为unknown后才真正标记为陌生人 |

---

## YOLO检测器参数

### 模型配置参数
| 参数名 | 类型 | 默认值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `model_path` | str | 'model/best_epoch_weights.pth' | yolo2.py 第53行 | 训练好的YOLO模型权重文件路径 |
| `classes_path` | str | 'model/voc_classes.txt' | yolo2.py 第54行 | 类别名称文件，包含所有检测类别 |
| `input_shape` | list | [640, 640] | yolo2.py 第55行 | 模型输入图像尺寸 [高度, 宽度] |
| `phi` | str | 's' | yolo2.py 第56行 | **模型变体参数，决定模型大小和性能**<br>可选值：<br>- 'n': nano（最小，最快）<br>- 's': small（小）<br>- 'm': medium（中等）<br>- 'l': large（大）<br>- 'x': xlarge（超大，最准确）<br><br>**注意**：切换phi时，权重文件必须对应训练时的模型大小，否则会报错 |

### 检测阈值参数
| 参数名 | 类型 | 默认值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `confidence` | float | 0.3 | yolo2.py 第56行 | 置信度阈值，只保留置信度 >= 此值的检测框 |
| `nms_iou` | float | 0.3 | yolo2.py 第57行 | NMS的IoU阈值，用于去除重叠的检测框 |

### 图像预处理参数
| 参数名 | 类型 | 默认值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `letterbox_image` | bool | True | yolo2.py 第58行 | 是否使用letterbox方式缩放图像（保持宽高比，填充黑边） |

### 计算设备参数
| 参数名 | 类型 | 默认值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `cuda` | bool | True | yolo2.py 第59行 | 是否使用GPU加速（需要CUDA支持） |

### 脱敏处理参数
| 参数名 | 类型 | 默认值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `mosaic_type` | str | "pixelate" | yolo2.py 第62行 | 马赛克类型：<br>- pixelate: 像素化<br>- gaussian: 高斯模糊<br>- mean: 均值模糊<br>- median: 中值模糊<br>- color: 颜色填充 |
| `mosaic_block` | int | 14 | yolo2.py 第63行 | 像素化马赛克块大小，值越大马赛克效果越强 |
| `expand_ratio` | float | 0.2 | yolo2.py 第64行 | 扩框比例，检测框向外扩展的比例（防止人脸边缘漏遮）<br>例如：原框宽度100，扩展后为120 |
| `min_face_size` | int | 10 | yolo2.py 第65行 | 最小人脸尺寸，小于此尺寸的检测框会被忽略（可能是误检） |

---

## 视频处理参数

### 视频输入输出配置
| 参数名 | 类型 | 默认值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `video_path` | str | 'vedio/man.mp4' | pre2.py 第163行 | 输入视频文件路径 |
| `video_save_path` | str | "vedio/man_out.mp4" | pre2.py 第164行 | 处理后视频的保存路径 |
| `video_fps` | int | 60 | pre2.py 第165行 | 输出视频的帧率 |

### 跟踪器参数（SimpleTracker）
| 参数名 | 类型 | 默认值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `iou_th` | float | 0.15 | pre2.py 第173行 | 跟踪匹配的IoU阈值，两个框IoU大于此值才认为是同一目标 |
| `max_miss` | int | 2 | pre2.py 第173行 | 最大容忍丢失帧数，超过此值则删除轨迹 |
| `ema_alpha` | float | 0.6 | pre2.py 第173行 | EMA平滑系数，用于平滑跟踪框的抖动<br>越大越接近当前检测框，越小越平滑但滞后 |

### 人脸识别相关参数
| 参数名 | 类型 | 默认值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `rec_onnx_path` | str | 绝对路径 | pre2.py 第178行 | ArcFace人脸识别模型的ONNX文件路径 |
| `gallery_dir` | str | "gallery" | pre2.py 第180行 | 人脸底库照片目录路径 |

### 识别优化参数
| 参数名 | 类型 | 默认值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `rec_interval` | int | 1 | pre2.py 第186行 | 识别间隔帧数，每隔几帧做一次特征提取<br>1表示每帧都做，设大可提速 |
| `num` | int | 3 | pre2.py 第189行 | 连续正确识别超过此帧数后，短暂失败也不变unknown |
| `unknown_warmup` | int | 8 | pre2.py 第190行 | 同一ID连续unknown达到此帧数才真正当陌生人 |
| `min_track_length` | int | 5 | pre2.py 第191行 | 目标至少被跟踪多少帧后才开始考虑打码，减少新目标抖动 |

### 人脸关键点参数
| 参数名 | 类型 | 默认值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `margin` | float | 0.35 | pre2.py 第231行 | 人脸关键点扩展比例，扩充框有助于关键点定位准确 |

### 马赛克控制参数
| 参数名 | 类型 | 默认值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `enable_mosaic` | bool | False | pre2.py 第151行 | 是否对unknown人脸进行马赛克处理 |
| `enable_recognition` | bool | True | pre2.py 第229行 | 是否启用人脸识别功能 |

---

## 人脸库参数

### 构建底库参数
| 参数名 | 类型 | 默认值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `gallery_dir` | str | "gallery" | face_gallery.py 第52行 | 底库照片目录路径 |

### 识别参数
| 参数名 | 类型 | 默认值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `sim_th` | float | 0.45 | face_gallery.py 第88行 | 相似度阈值，低于此值返回"unknown"<br>值越高识别越严格，值越低识别越宽松 |

### FaceKpsAlignRec参数
| 参数名 | 类型 | 默认值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `det_size` | tuple | (640, 640) | face_gallery.py 第34行 | 人脸检测模型的输入尺寸 |
| `ctx_id` | int | 0 | face_gallery.py 第35行 | GPU设备ID（0表示第一个GPU） |
| `providers` | tuple | ("CUDAExecutionProvider", "CPUExecutionProvider") | face_gallery.py 第36行 | ONNX Runtime执行提供者，优先使用GPU |

---

## 图像处理参数

### apply_mosaic函数参数
| 参数名 | 类型 | 默认值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `mosaic_type` | str | 'pixelate' | image_processor.py 第34行 | 马赛克类型，可选值：<br>- pixelate: 像素化<br>- gaussian: 高斯模糊<br>- mean: 均值模糊<br>- median: 中值模糊<br>- color: 颜色填充 |

#### pixelate（像素化）参数
| 参数名 | 类型 | 默认值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `block_size` | int | 14 | image_processor.py 第60行 | 像素化块大小，值越大马赛克效果越强 |

#### gaussian（高斯模糊）参数
| 参数名 | 类型 | 默认值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `blur_size` | int | 15 | image_processor.py 第66行 | 高斯核大小（会自动调整为奇数）<br>值越大模糊效果越强 |
| `sigma_x` | int | 0 | image_processor.py 第67行 | 高斯核在X方向的标准差<br>0表示自动计算 |

#### mean（均值模糊）参数
| 参数名 | 类型 | 默认值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `blur_size` | int | 15 | image_processor.py 第72行 | 均值核大小，值越大模糊效果越强 |

#### median（中值模糊）参数
| 参数名 | 类型 | 默认值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `blur_size` | int | 15 | image_processor.py 第77行 | 中值核大小（会自动调整为奇数）<br>值越大平滑效果越强 |

#### color（颜色填充）参数
| 参数名 | 类型 | 默认值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `color` | list | [128, 128, 128] | image_processor.py 第82行 | 填充颜色 [B, G, R]<br>默认为灰色 |

---

## YOLO网络参数

### 模型变体参数
| 参数名 | 类型 | 可选值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `phi` | str | 'n', 's', 'm', 'l', 'x' | yolo.py 第127行 | 模型变体<br>- n: nano（最小，最快）<br>- s: small（小）<br>- m: medium（中等）<br>- l: large（大）<br>- x: xlarge（超大，最准确） |

### 深度和宽度倍数
| 参数名 | 类型 | 默认值(s模型) | 调用位置 | 效果说明 |
|--------|------|---------------|----------|----------|
| `dep_mul` | float | 0.33 | yolo.py 第128行 | 深度倍数，控制网络层数<br>值越大网络越深，计算量越大 |
| `wid_mul` | float | 0.50 | yolo.py 第129行 | 宽度倍数，控制每层通道数<br>值越大通道数越多，计算量越大 |
| `deep_mul` | float | 1.00 | yolo.py 第130行 | 深层宽度倍数，控制深层通道数<br>值越大深层特征越丰富 |

### 网络结构参数
| 参数名 | 类型 | 默认值 | 调用位置 | 效果说明 |
|--------|------|--------|----------|----------|
| `input_shape` | list/tuple | - | yolo.py 第127行 | 输入图像尺寸 [高度, 宽度] |
| `num_classes` | int | - | yolo.py 第127行 | 检测类别数量 |
| `pretrained` | bool | False | yolo.py 第127行 | 是否使用预训练权重 |
| `reg_max` | int | 16 | yolo.py 第169行 | DFL（分布焦点损失）的分布通道数<br>表示坐标值被离散化为0到15的整数 |

### 网络输出参数
| 参数名 | 类型 | 计算方式 | 调用位置 | 效果说明 |
|--------|------|----------|----------|----------|
| `base_channels` | int | wid_mul * 64 | yolo.py 第135行 | 基础通道数，所有层的通道数基于此值 |
| `base_depth` | int | max(round(dep_mul * 3), 1) | yolo.py 第136行 | 基础深度，所有C2f模块的重复次数基于此值 |

---

## 参数调优建议

### 性能优化
1. **降低推理速度**: 
   - 减小 `input_shape` (如从640x640降到480x480)
   - 使用更小的模型变体 `phi='n'`
   - 增加 `confidence` 阈值以过滤低置信度检测
   - 增加 `rec_interval` 减少人脸识别频率

2. **提高检测精度**:
   - 增大 `input_shape` (如从640x640增到1280x1280)
   - 使用更大的模型变体 `phi='x'`
   - 降低 `confidence` 阈值以保留更多检测框

3. **减少误检**:
   - 提高 `confidence` 阈值
   - 提高 `nms_iou` 阈值以保留更多框
   - 增大 `min_face_size` 忽略小目标

### 隐私保护调优
1. **增强马赛克效果**:
   - 增大 `mosaic_block` 值
   - 增大模糊核大小 `blur_size`

2. **防止边缘漏遮**:
   - 增大 `expand_ratio` 值

3. **人脸识别灵敏度**:
   - 调整 `sim_th` 阈值（值越高识别越严格）
   - 调整 `unknown_warmup`（值越大越不容易打码）

---

## 参数使用示例

### 示例1：初始化YOLO检测器
```python
from yolo2 import YOLO

# 使用默认参数（phi='s'）
yolo = YOLO()

# 使用nano模型（最小，最快）
yolo = YOLO(phi='n')

# 使用大模型（更准确，但速度慢）
yolo = YOLO(phi='l')

# 自定义多个参数
yolo = YOLO(
    phi='m',                # 使用medium模型
    confidence=0.5,         # 提高置信度阈值
    nms_iou=0.4,            # 提高NMS阈值
    mosaic_type="gaussian", # 使用高斯模糊马赛克
    mosaic_block=20,        # 增强马赛克效果
    expand_ratio=0.3        # 防止边缘漏遮
)
```

### 示例2：切换模型变体
```python
from yolo2 import YOLO

# 快速模式 - 适合实时视频处理
yolo_fast = YOLO(phi='n', input_shape=[480, 480])

# 平衡模式 - 默认配置
yolo_balanced = YOLO(phi='s', input_shape=[640, 640])

# 高精度模式 - 适合离线处理
yolo_accurate = YOLO(phi='x', input_shape=[1280, 1280])
```

### 示例2：处理视频
```python
from yolo2 import YOLO
from utils.face_gallery import FaceGallery

yolo = YOLO()
face_gallery = FaceGallery(yolo, rec_onnx_path="path/to/model.onnx")
gallery = face_gallery.build_gallery(gallery_dir="gallery")

# 调整识别参数
name, sim = face_gallery.recognize(emb, sim_th=0.5)  # 提高阈值使识别更严格
```

### 示例3：应用马赛克
```python
from utils.image_processor import ImageProcessor

processor = ImageProcessor()

# 像素化马赛克
image = processor.apply_mosaic(image, x1, y1, x2, y2, 
                               mosaic_type='pixelate', block_size=20)

# 高斯模糊马赛克
image = processor.apply_mosaic(image, x1, y1, x2, y2, 
                               mosaic_type='gaussian', blur_size=25, sigma_x=0)

# 颜色填充
image = processor.apply_mosaic(image, x1, y1, x2, y2, 
                               mosaic_type='color', color=[0, 0, 0])  # 黑色
```

---

## 注意事项

1. **路径参数**: 所有路径参数建议使用绝对路径或相对于项目根目录的路径
2. **GPU加速**: 确保安装CUDA和cuDNN才能使用 `cuda=True`
3. **内存限制**: 大模型（phi='x'）需要更多显存，小显存设备建议使用phi='n'或phi='s'
4. **帧率平衡**: 提高精度会降低帧率，需要根据实际需求平衡
5. **阈值调优**: 不同场景下最佳阈值可能不同，建议根据实际数据测试调整
6. **phi参数重要提示**: 
   - 切换phi参数时，模型权重文件必须与训练时的模型大小匹配
   - 如果权重文件是用phi='s'训练的，不能在phi='n'或phi='x'上加载，会报错
   - 每个phi值都有对应的权重文件，确保使用正确的权重文件
   - 如果没有对应phi的权重文件，需要重新训练模型或下载预训练权重

---

## 环境变量配置

以下是项目中使用的主要环境变量及其默认值：

### 识别优化参数
| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `RECOGNITION_INTERVAL` | 1 | 识别间隔帧数，每隔几帧做一次特征提取 |
| `RECOGNITION_NUM` | 3 | 连续正确识别超过此帧数后，短暂失败也不变unknown |
| `UNKNOWN_WARMUP` | 8 | 同一ID连续unknown达到此帧数才真正当陌生人 |
| `MIN_TRACK_LENGTH` | 5 | 目标至少被跟踪多少帧后才开始考虑打码 |

### 跟踪器参数
| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `TRACKER_IOU_THRESHOLD` | 0.15 | 跟踪匹配的IoU阈值，两个框IoU大于此值才认为是同一目标 |
| `TRACKER_MAX_MISS` | 2 | 最大容忍丢失帧数，超过此值则删除轨迹 |
| `TRACKER_EMA_ALPHA` | 0.6 | EMA平滑系数，用于平滑跟踪框的抖动 |

### 检测阈值参数
| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `DETECTION_CONFIDENCE` | 0.3 | 置信度阈值，只有置信度大于等于此值的检测框才会被保留 |
| `NMS_IOU_THRESHOLD` | 0.3 | NMS（非极大值抑制）的IoU阈值，用于去除重叠的检测框 |

### 如何设置环境变量

1. **Linux/Mac系统**:
```bash
export RECOGNITION_INTERVAL=2
export DETECTION_CONFIDENCE=0.5
```

2. **Windows系统**:
```cmd
set RECOGNITION_INTERVAL=2
set DETECTION_CONFIDENCE=0.5
```

3. **Python代码中设置**:
```python
import os
os.environ["RECOGNITION_INTERVAL"] = "2"
os.environ["DETECTION_CONFIDENCE"] = "0.5"
```

4. **Docker中使用**:
```dockerfile
ENV RECOGNITION_INTERVAL=2
ENV DETECTION_CONFIDENCE=0.5
```

---

**文档版本**: 1.1  
**最后更新**: 2026-01-11  
**项目**: yolo-automic-mosaic
