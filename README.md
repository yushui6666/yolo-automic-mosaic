# 人脸识别与马赛克脱敏系统

这个系统集成了YOLO人脸检测和InsightFace人脸识别技术，可以对人脸进行检测、识别，并可选择性地对人脸应用马赛克脱敏处理。

## 功能特点

- **人脸检测**：使用YOLO模型检测图片或视频中的人脸
- **人脸识别**：使用InsightFace模型对检测到的人脸进行识别
- **马赛克脱敏**：可选择对人脸区域应用马赛克处理
- **支持媒体类型**：支持图片(JPG, PNG, GIF)和视频(MP4, AVI, MOV)文件
- **Web界面**：提供简洁易用的Web界面进行文件上传和处理

## 系统要求

- Python 3.8+
- CUDA支持的GPU（推荐，用于加速模型推理）
- 足够的存储空间用于存放模型文件和处理结果

## 安装说明

1. 克隆此仓库到本地：
   ```bash
   git clone https://github.com/yourusername/yolo-automic-mosaic.git
   cd yolo-automic-mosaic
   ```

2. 创建并激活虚拟环境：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或者
   venv\Scripts\activate     # Windows
   ```

3. 安装依赖项：
   ```bash
   pip install -r requirements.txt
   ```

4. 安装InsightFace模型：
   ```bash
   python -m pip install insightface
   python -c "import insightface; insightface.app.download_model('buffalo_l')"
   ```

5. 确保模型文件存在：
   - 检查`model`目录下是否有`best_epoch_weights.pth`和`voc_classes.txt`文件
   - 检查`~/.insightface/models/buffalo_l/`目录下是否有`w600k_r50.onnx`文件

## 使用说明

1. 启动Flask应用：
   ```bash
   cd src
   python app.py
   ```

2. 在浏览器中访问：`http://localhost:5000`

3. 上传图片或视频文件：
   - 点击"选择文件"按钮，从本地选择要处理的文件
   - 上传成功后，可以选择是否启用人脸识别和/或马赛克脱敏
   - 点击"处理文件"按钮开始处理

4. 下载处理结果：
   - 处理完成后，点击"下载处理结果"按钮下载处理后的文件

## 目录结构

```
yolo-automic-mosaic/
├── src/                 # Flask应用源代码
│   ├── app.py          # Flask主应用
│   ├── pre2.py         # 原始视频处理脚本
│   └── yolo2.py        # YOLO检测器类
├── utils/               # 工具模块
│   ├── face_gallery.py # 人脸识别功能
│   ├── image_processor.py # 图像处理工具
│   └── ...             # 其他工具模块
├── model/               # 模型文件目录
│   ├── best_epoch_weights.pth # YOLO模型权重
│   └── voc_classes.txt       # 类别定义文件
├── gallery/            # 人脸底库目录
├── vedio/              # 处理后的视频存储目录
├── uploads/            # 上传文件和处理结果目录
├── templates/          # HTML模板
│   └── index.html      # 上传界面
├── requirements.txt    # Python依赖列表
└── README.md           # 说明文档
```

## 注意事项

1. 首次运行时，系统会自动下载所需的InsightFace模型，这可能需要一些时间。
2. 视频处理可能会消耗较多计算资源，请确保系统有足够的内存和GPU资源。
3. 人脸识别的准确性取决于训练数据质量和图片/视频的清晰度。
4. 处理大型文件可能需要较长时间，请耐心等待。

## 常见问题

1. **模型文件不存在错误**
   - 确保`model`目录中有正确的模型文件
   - 检查InsightFace模型是否正确安装：`python -c "import insightface; print(insightface.__version__)"`

2. **内存不足错误**
   - 尝试处理较小的文件
   - 确保系统有足够的可用内存

3. **GPU不可用**
   - 系统将自动回退到CPU模式，但处理速度会显著降低

## 技术支持

如有问题或建议，请提交Issue或联系开发者。

## 许可证

本项目遵循MIT许可证。
