# Utils 文件夹功能介绍

本文档介绍了 `utils` 文件夹中各个模块的功能和用途。该文件夹包含了项目所需的工具类和函数，主要用于数据处理、模型训练、评估和图像处理等方面。

## 文件列表

| 文件名 | 功能概述 |
|--------|----------|
| [callbacks.py](callbacks.py) | 训练过程中的回调和可视化工具 |
| [dataloader.py](dataloader.py) | YOLO 数据集加载和数据增强 |
| [face_pi.py](face_pi.py) | 人脸关键点定位、对齐和特征提取工具类 |
| [image_processor.py](image_processor.py) | 图像处理类，包含各种图像滤镜和操作 |
| [utils.py](utils.py) | 通用工具函数集合 |
| [utils_bbox.py](utils_bbox.py) | 边界框解码和非极大值抑制（NMS） |
| [utils_fit.py](utils_fit.py) | 训练一个 epoch 的核心函数 |
| [utils_map.py](utils_map.py) | mAP（mean Average Precision）计算和评估 |
| [__init__.py](__init__.py) | Python 包初始化文件 |

## 详细说明

### [callbacks.py](callbacks.py)

**主要功能：**
1. **LossHistory**：记录和可视化训练损失、验证损失
2. **EvalCallback**：定期评估模型性能，计算 mAP 并可视化
3. 支持 TensorBoard 日志记录和图像保存

**核心类：**
- `LossHistory`：用于记录训练过程中的损失值，并生成损失曲线图。
- `EvalCallback`：在训练的特定阶段评估模型性能，计算 mAP（平均精度均值），并保存评估结果。

### [dataloader.py](dataloader.py)

**主要功能：**
1. **YoloDataset**：自定义数据集类，用于加载训练和验证数据
2. 支持多种数据增强：Mosaic、MixUp、随机翻转、颜色变换等
3. 标签格式转换：将标注框转换为模型训练所需的格式
4. yolo_dataset_collate：DataLoader 的批处理函数

**核心类：**
- `YoloDataset`：继承自 `Dataset`，用于加载和预处理 YOLO 训练数据。
- `yolo_dataset_collate`：自定义的批处理函数，用于将数据打包成 batch。

### [face_pi.py](face_pi.py)

**主要功能：**
1. 使用 InsightFace 检测器提取人脸 5 个关键点
2. 将人脸对齐到标准 112x112 尺寸
3. 使用 ArcFace 模型提取人脸特征向量（embedding）
4. 支持人脸关键点可视化

**核心类：**
- `FaceKpsAlignRec`：负责人脸关键点检测、对齐和特征提取的工具类。

### [image_processor.py](image_processor.py)

**主要功能：**
1. 提供人脸马赛克脱敏功能
2. 未来可扩展更多图像处理操作

**核心类：**
- `ImageProcessor`：图像处理工具类，目前提供马赛克功能。

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

### [utils_fit.py](utils_fit.py)

**主要功能：**
1. 执行一个完整的训练 epoch（训练 + 验证）
2. 支持混合精度训练（FP16）
3. 支持 EMA（指数移动平均）模型更新
4. 自动保存最佳模型和定期检查点

**核心函数：**
- `fit_one_epoch`：执行一个训练 epoch 的核心函数。

### [utils_map.py](utils_map.py)

**主要功能：**
1. 计算 VOC 格式的 mAP（Pascal VOC 评估标准）
2. 计算 COCO 格式的 mAP（COCO 评估标准，包含多个 IoU 阈值）
3. 绘制评估曲线：AP、Precision、Recall、F1 曲线
4. 生成评估报告和可视化结果
5. 支持 TP/FP/FN 统计和可视化

**核心函数：**
- `get_map`：计算 VOC 格式的 mAP。
- `get_coco_map`：计算 COCO 格式的 mAP。

## 使用说明

1. **数据加载**：使用 `dataloader.py` 中的 `YoloDataset` 类加载数据。
2. **模型训练**：使用 `utils_fit.py` 中的 `fit_one_epoch` 函数进行训练。
3. **模型评估**：使用 `utils_map.py` 中的 `get_map` 或 `get_coco_map` 函数评估模型性能。
4. **人脸处理**：使用 `face_pi.py` 中的 `FaceKpsAlignRec` 类进行人脸相关操作。
5. **图像处理**：使用 `image_processor.py` 中的 `ImageProcessor` 类进行图像处理。
6. **通用工具**：使用 `utils.py` 中的各种辅助函数进行数据处理和模型操作。

## 注意事项

1. 确保所有依赖库（如 OpenCV、PyTorch、NumPy 等）已正确安装。
2. 在使用人脸相关功能时，需要确保 InsightFace 模型已下载并正确配置路径。
3. 在运行 mAP 计算时，确保检测结果和真实标签的格式正确。

## 贡献

如果您有任何建议或发现 bug，请提交 issue 或 pull request。

## 许可证

请参考项目根目录下的 LICENSE 文件。
