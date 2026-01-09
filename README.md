# YOLO人脸检测与识别系统

基于YOLO目标检测和InsightFace人脸识别的智能视频/图像处理系统，支持人脸检测、识别和马赛克脱敏处理。

## 主要功能

- **人脸检测**：使用YOLOv8模型实时检测图片和视频中的人脸
- **人脸识别**：基于InsightFace的高精度人脸识别，支持自定义人脸底库
- **马赛克脱敏**：可选择对识别的人脸区域应用马赛克保护隐私
- **智能跟踪**：简化版人脸跟踪算法，提高视频处理效率
- **Web界面**：简洁易用的Flask Web界面，支持文件上传和在线处理
- **多格式支持**：支持图片（JPG, JPEG, PNG, GIF）和视频（MP4, AVI, MOV）

## 项目结构

```
project/
├── src/                        # Flask应用源代码
│   ├── app.py                 # Flask主应用（Web服务器）
│   ├── pre2.py                # 原始视频处理脚本
│   ├── yolo2.py               # YOLO检测器封装类
│   └── templates/
│       └── index.html         # Web前端界面
├── nets/                       # 神经网络模块
│   ├── yolo.py                # YOLO主体网络结构
│   ├── yolo_training.py       # YOLO训练相关代码
│   ├── backbone.py            # Backbone主干网络
│   └── nets_explanation.md    # 网络结构说明文档
├── utils/                      # 工具模块
│   ├── face_gallery.py        # 人脸识别与底库管理
│   ├── image_processor.py     # 图像处理工具
│   ├── face_pi.py             # 人脸检测辅助工具
│   ├── utils_bbox.py          # 边界框工具
│   ├── utils_map.py           # MAP评估工具
│   ├── utils_fit.py           # 训练拟合工具
│   ├── callbacks.py           # 训练回调函数
│   └── dataloader.py          # 数据加载器
├── model/                      # 模型文件目录
│   ├── best_epoch_weights.pth # YOLO训练权重
│   └── voc_classes.txt        # 类别定义文件
├── gallery/                    # 人脸底库目录
├── video/                      # 处理后的视频存储目录
├── uploads/                    # 临时上传文件目录
├── requirements.txt           # Python依赖列表
├── index.html                  # 前端页面（备用）
└── README.md                   # 项目说明文档
```

## 环境要求

- Python 3.8+
- CUDA 11.0+（可选，用于GPU加速）
- 至少4GB可用内存（视频处理建议8GB以上）
- 2GB可用磁盘空间（用于模型文件）

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/yushui6666/yolo-automic-mosaic.git
cd yolo-automic-mosaic
```

### 2. 创建虚拟环境（推荐）

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

依赖包列表：
- flask==2.3.0 - Web框架
- werkzeug==2.3.0 - WSGI工具库
- opencv-python==4.7.0.72 - 图像/视频处理
- numpy==1.24.3 - 数值计算
- torch>=2.0.0 - 深度学习框架
- pillow>=9.0.0 - 图像处理
- onnxruntime>=1.15.0 - ONNX推理引擎

### 4. 安装InsightFace模型

```bash
pip install insightface
python -c "import insightface; insightface.app.download_model('buffalo_l')"
```

模型将自动下载到 `~/.insightface/models/buffalo_l/` 目录

### 5. 确认模型文件

确保以下文件存在：
- `model/best_epoch_weights.pth` - YOLO模型权重
- `model/voc_classes.txt` - 类别定义
- `~/.insightface/models/buffalo_l/w600k_r50.onnx` - 人脸识别模型

## 使用说明

### 启动Web服务

```bash
cd src
python app.py
```

服务将在 `http://localhost:5000` 启动

### Web界面操作

#### 1. 上传照片到人脸底库
- 点击"上传照片"按钮
- 选择要添加的照片
- 可选：输入自定义名称（例如："张三"）
- 照片将保存到 `gallery/` 目录，用于后续识别

#### 2. 处理视频
- 点击"选择视频"按钮
- 选择要处理的视频文件（MP4, AVI, MOV）
- 配置处理选项：
  - **启用人脸识别**：勾选后将识别人脸并显示姓名
  - **应用马赛克**：勾选后对人脸区域应用马赛克
- 点击"处理文件"按钮
- 等待处理完成后，点击"下载结果"获取处理后的视频

### 处理逻辑

1. **人脸检测**：YOLOv8模型检测每一帧中的人脸
2. **人脸识别**（可选）：
   - 使用InsightFace提取人脸特征
   - 与gallery底库中的特征进行比对
   - 返回识别结果和相似度
3. **智能跟踪**：简化跟踪算法保持人脸ID一致性
4. **马赛克处理**（可选）：
   - 仅对未识别的人脸应用马赛克
   - 或对所有人脸应用马赛克
   - 可配置扩展比例和马赛克块大小

## 技术架构

### 网络结构

#### YOLOv8检测网络
- **Backbone**：特征提取主干网络（CSPDarknet）
- **Neck**：特征融合网络（FPN + PAN）
- **Head**：检测头（分类 + 回归）
- **DFL**：分布焦点损失模块

#### 模型变体
支持多种模型规模：
- `n`: nano - 最小模型，推理最快
- `s`: small - 小型模型，平衡速度和精度
- `m`: medium - 中型模型，较高精度
- `l`: large - 大型模型，高精度
- `x`: xlarge - 超大模型，最高精度

### 人脸识别流程

1. **关键点检测**：使用5点关键点检测
2. **人脸对齐**：基于关键点进行人脸对齐到112x112
3. **特征提取**：使用ResNet50提取512维特征向量
4. **特征比对**：余弦相似度计算，阈值0.45

## 命令行参数

在 `src/app.py` 中可配置：

```python
# Flask配置
app.run(debug=True, host='0.0.0.0', port=5000)

# YOLO配置
yolo = YOLO(
    model_path='model/best_epoch_weights.pth',
    classes_path='model/voc_classes.txt',
    confidence=0.3,  # 检测置信度阈值
    nms_iou=0.3      # NMS IoU阈值
)

# 人脸识别配置
sim_th=0.45  # 识别相似度阈值
```

## 性能优化

### GPU加速
确保CUDA环境正确配置：
```bash
# 检查CUDA是否可用
python -c "import torch; print(torch.cuda.is_available())"
```

### 模型融合
使用模型融合加速推理：
```python
yolo.model.fuse()  # 融合Conv和BN层
```

### 批量处理
对于大量文件，建议编写批量处理脚本。

## 常见问题

### 1. 模型文件不存在错误
```
解决方案：
- 确保model目录下有best_epoch_weights.pth和voc_classes.txt
- 运行：python -c "import insightface; insightface.app.download_model('buffalo_l')"
```

### 2. 内存不足错误
```
解决方案：
- 处理较小的视频文件
- 降低batch_size（如适用）
- 使用较小的模型变体（yolov8n）
```

### 3. GPU不可用
```
解决方案：
- 检查CUDA版本是否与PyTorch兼容
- 重新安装GPU版本的PyTorch：pip install torch --index-url https://download.pytorch.org/whl/cu118
- 系统会自动回退到CPU模式，但速度较慢
```

### 4. 人脸识别准确率低
```
解决方案：
- 确保gallery中的照片清晰、正面
- 增加底库中的样本数量
- 调整相似度阈值（sim_th）
```

## 开发说明

### 训练自定义YOLO模型

```bash
# 准备数据集
# 按照VOC格式组织数据

# 训练
python train.py --data_path /path/to/data --epochs 100 --batch_size 16
```

### 扩展功能

项目采用模块化设计，易于扩展：
- 添加新的检测类别：修改 `model/voc_classes.txt`
- 自定义马赛克效果：修改 `utils/image_processor.py`
- 实现更复杂的跟踪算法：修改 `src/app.py` 中的跟踪逻辑

## 依赖项版本兼容性

| 包名 | 版本 | 说明 |
|------|------|------|
| Python | 3.8+ | 推荐使用3.9或3.10 |
| PyTorch | 2.0.0+ | 建议使用GPU版本 |
| CUDA | 11.0+ | 与PyTorch版本匹配 |
| OpenCV | 4.7.0 | 用于图像/视频处理 |
| Flask | 2.3.0 | Web框架 |

## 许可证

本项目遵循MIT许可证。详见LICENSE文件。

## 贡献指南

欢迎提交Issue和Pull Request！

## 联系方式

- 项目地址：https://github.com/yushui6666/yolo-automic-mosaic
- Issues：https://github.com/yushui6666/yolo-automic-mosaic/issues

## 更新日志

### v1.0.0
- 初始版本发布
- 支持YOLOv8人脸检测
- 集成InsightFace人脸识别
- 实现马赛克脱敏功能
- 提供Web界面

## 致谢

- [YOLOv8](https://github.com/ultralytics/ultralytics) - 目标检测算法
- [InsightFace](https://github.com/deepinsight/insightface) - 人脸识别算法
- [Flask](https://flask.palletsprojects.com/) - Web框架
