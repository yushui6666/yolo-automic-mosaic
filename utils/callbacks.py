#-----------------------------------------------------------------------#
#   callbacks.py：训练过程中的回调和可视化工具
#   功能概述：
#   1. LossHistory：记录和可视化训练损失、验证损失
#   2. EvalCallback：定期评估模型性能，计算 mAP 并可视化
#   3. 支持 TensorBoard 日志记录和图像保存
#-----------------------------------------------------------------------#
import datetime
import os

import torch
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 日志记录

import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm  # 进度条显示
from .utils import cvtColor, preprocess_input, resize_image
from .utils_bbox import DecodeBox
from .utils_map import get_coco_map, get_map
class LossHistory():
    """
    训练损失历史记录和可视化类
    
    主要功能：
    1. 记录每个 epoch 的训练损失和验证损失
    2. 将损失写入文本文件（便于后续分析）
    3. 使用 TensorBoard 记录损失（可通过 tensorboard --logdir=log_dir 查看）
    4. 绘制损失曲线图并保存为 PNG 文件
    
    使用示例：
        loss_history = LossHistory(log_dir="./logs", model=model, input_shape=[640, 640])
        for epoch in range(epochs):
            train_loss = ...
            val_loss = ...
            loss_history.append_loss(epoch, train_loss, val_loss)
    """
    def __init__(self, log_dir, model, input_shape):
        """
        初始化损失历史记录器
        
        参数:
            log_dir: str, 日志保存目录路径
            model: torch.nn.Module, 模型对象（用于 TensorBoard 图可视化，可选）
            input_shape: list/tuple, 模型输入图像尺寸 [height, width]
        """
        self.log_dir    = log_dir      # 日志目录
        self.losses     = []           # 训练损失列表
        self.val_loss   = []           # 验证损失列表
        
        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化 TensorBoard 写入器
        # 可以通过 tensorboard --logdir=log_dir 命令查看可视化结果
        self.writer     = SummaryWriter(self.log_dir)
        
        # 可选：记录模型计算图（用于 TensorBoard 可视化模型结构）
        # 注释掉是因为某些模型可能不支持图可视化
        # try:
        #     dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
        #     self.writer.add_graph(model, dummy_input)
        # except:
        #     pass

    def append_loss(self, epoch, loss, val_loss):
        """
        添加一个 epoch 的损失值并更新可视化
        
        参数:
            epoch: int, 当前 epoch 编号
            loss: float, 训练损失值
            val_loss: float, 验证损失值
        
        功能:
            1. 将损失值添加到历史记录
            2. 将损失值追加到文本文件
            3. 将损失值写入 TensorBoard
            4. 更新损失曲线图
        """
        # 确保日志目录存在
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # 记录损失值
        self.losses.append(loss)
        self.val_loss.append(val_loss)

        # 将损失值追加到文本文件（便于后续分析和脚本处理）
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        # 将损失值写入 TensorBoard（用于实时可视化）
        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        
        # 更新损失曲线图
        self.loss_plot()

    def loss_plot(self):
        """
        绘制并保存损失曲线图
        
        功能:
            1. 绘制训练损失和验证损失曲线
            2. 可选：绘制平滑后的损失曲线（Savitzky-Golay 滤波）
            3. 保存为 PNG 图片
        
        输出:
            epoch_loss.png: 损失曲线图，保存在 log_dir 目录下
        """
        # 生成 epoch 索引
        iters = range(len(self.losses))

        # 创建新图形
        plt.figure()
        
        # 绘制原始损失曲线
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        
        # 可选：绘制平滑后的损失曲线（使用 Savitzky-Golay 滤波器）
        # 平滑可以去除噪声，更清晰地观察损失趋势
        # 注释掉是因为需要 scipy.signal，且可能在某些情况下失败
        # try:
        #     if len(self.losses) < 25:
        #         num = 5   # 数据点少时，使用较小的窗口
        #     else:
        #         num = 15  # 数据点多时，使用较大的窗口
            
        #     # Savitzky-Golay 滤波：在保持数据形状的同时平滑曲线
        #     plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 
        #              'green', linestyle = '--', linewidth = 2, label='smooth train loss')
        #     plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), 
        #              '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        # except:
        #     pass

        # 设置图表属性
        plt.grid(True)                    # 显示网格
        plt.xlabel('Epoch')               # x 轴标签
        plt.ylabel('Loss')                # y 轴标签
        plt.legend(loc="upper right")      # 显示图例

        # 保存图片
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        # 清理图形（释放内存）
        plt.cla()      # 清除当前轴
        plt.close("all")  # 关闭所有图形

class EvalCallback():
    """
    模型评估回调类
    
    主要功能：
    1. 定期在验证集上评估模型性能
    2. 计算 mAP（mean Average Precision）指标
    3. 保存检测结果和真实标签到临时文件
    4. 绘制 mAP 曲线图
    5. 支持 COCO 格式和 VOC 格式的 mAP 计算
    
    使用示例：
        eval_callback = EvalCallback(
            net=model, 
            input_shape=[640, 640],
            class_names=class_names,
            num_classes=num_classes,
            val_lines=val_lines,
            log_dir="./logs",
            cuda=True
        )
        eval_callback.on_epoch_end(epoch, model)
    """
    def __init__(self, net, input_shape, class_names, num_classes, val_lines, log_dir, cuda, \
            map_out_path=".temp_map_out", max_boxes=100, confidence=0.05, nms_iou=0.5, letterbox_image=True, MINOVERLAP=0.5, eval_flag=True, period=1):
        """
        初始化评估回调
        
        参数:
            net: torch.nn.Module, 模型对象
            input_shape: list/tuple, 模型输入图像尺寸 [height, width]
            class_names: list, 类别名称列表
            num_classes: int, 类别数量
            val_lines: list, 验证集标注行列表，每行格式：图片路径 框1 框2 ...
            log_dir: str, 日志保存目录
            cuda: bool, 是否使用 GPU
            map_out_path: str, 临时文件保存路径（用于 mAP 计算）
            max_boxes: int, 每张图片最多保留的检测框数量
            confidence: float, 置信度阈值
            nms_iou: float, NMS 的 IoU 阈值
            letterbox_image: bool, 是否使用 letterbox 方式缩放图像
            MINOVERLAP: float, mAP 计算时的最小 IoU 阈值（通常为 0.5）
            eval_flag: bool, 是否启用评估（False 时跳过评估）
            period: int, 评估周期（每 period 个 epoch 评估一次）
        """
        super(EvalCallback, self).__init__()
        
        # 保存配置参数
        self.net                = net
        self.input_shape        = input_shape
        self.class_names        = class_names
        self.num_classes        = num_classes
        self.val_lines          = val_lines      # 验证集数据
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.map_out_path       = map_out_path  # 临时文件路径
        self.max_boxes          = max_boxes      # 每张图最多检测框数
        self.confidence         = confidence     # 置信度阈值
        self.nms_iou            = nms_iou        # NMS IoU 阈值
        self.letterbox_image    = letterbox_image
        self.MINOVERLAP         = MINOVERLAP     # mAP 计算的最小 IoU
        self.eval_flag          = eval_flag      # 是否启用评估
        self.period             = period         # 评估周期
        
        # 初始化边界框解码器
        self.bbox_util          = DecodeBox(self.num_classes, (self.input_shape[0], self.input_shape[1]))
        
        # 初始化 mAP 和 epoch 历史记录
        self.maps       = [0]    # mAP 历史值列表
        self.epoches    = [0]    # 对应的 epoch 列表
        
        # 初始化 mAP 文本文件
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        """
        对单张图像进行检测，并将检测结果保存到文本文件
        
        参数:
            image_id: str, 图像 ID（不含扩展名）
            image: PIL.Image, 输入图像
            class_names: list, 类别名称列表
            map_out_path: str, 输出目录路径
        
        输出文件格式:
            detection-results/{image_id}.txt
            每行格式：类别名 置信度 x1 y1 x2 y2
        
        流程:
            1. 图像预处理（RGB 转换、resize、归一化）
            2. 模型推理
            3. 边界框解码和 NMS
            4. 保存检测结果到文本文件
        """
        # 打开输出文件（用于保存检测结果）
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"), "w", encoding='utf-8') 
        
        # 获取原始图像尺寸
        image_shape = np.array(np.shape(image)[0:2])
        
        #---------------------------------------------------------#
        #   将图像转换成 RGB 图像，防止灰度图在预测时报错
        #   代码仅支持 RGB 图像的预测，所有其它类型的图像都会转化成 RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的 resize
        #   也可以直接 resize 进行识别（但会变形）
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        
        #---------------------------------------------------------#
        #   图像预处理：归一化、维度转换、添加 batch 维度
        #   (H, W, C) -> (C, H, W) -> (1, C, H, W)
        #---------------------------------------------------------#
        image_data  = np.expand_dims(
            np.transpose(
                preprocess_input(np.array(image_data, dtype='float32')), 
                (2, 0, 1)
            ), 
            0
        )

        # 模型推理（禁用梯度计算，节省内存）
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            
            #---------------------------------------------------------#
            #   将图像输入网络进行预测
            #---------------------------------------------------------#
            outputs = self.net(images)
            
            # 解码边界框：将模型输出转换为边界框坐标
            outputs = self.bbox_util.decode_box(outputs)
            
            #---------------------------------------------------------#
            #   非极大值抑制（NMS）：去除重叠的检测框
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(
                outputs, 
                self.num_classes, 
                self.input_shape, 
                image_shape, 
                self.letterbox_image, 
                conf_thres = self.confidence, 
                nms_thres = self.nms_iou
            )
                                                    
            # 如果没有检测到目标，直接返回
            if results[0] is None: 
                f.close()
                return 

            # 提取检测结果：类别、置信度、边界框
            top_label   = np.array(results[0][:, 5], dtype = 'int32')  # 类别索引
            top_conf    = results[0][:, 4]                             # 置信度
            top_boxes   = results[0][:, :4]                            # 边界框 [top, left, bottom, right]

        # 按置信度排序，只保留前 max_boxes 个检测框
        top_100     = np.argsort(top_conf)[::-1][:self.max_boxes]
        top_boxes   = top_boxes[top_100]
        top_conf    = top_conf[top_100]
        top_label   = top_label[top_100]

        # 将检测结果写入文件
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]  # 类别名称
            box             = top_boxes[i]              # 边界框
            score           = str(top_conf[i])           # 置信度

            top, left, bottom, right = box
            
            # 跳过不在类别列表中的检测结果
            if predicted_class not in class_names:
                continue

            # 写入文件：类别名 置信度 x1 y1 x2 y2
            # 注意：box 格式是 [top, left, bottom, right]，需要转换为 [x1, y1, x2, y2]
            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, 
                score[:6],              # 置信度保留 6 位
                str(int(left)),         # x1
                str(int(top)),          # y1
                str(int(right)),        # x2
                str(int(bottom))        # y2
            ))

        f.close()
        return 
    
    def on_epoch_end(self, epoch, model_eval):
        """
        在每个 epoch 结束时调用，进行模型评估
        
        参数:
            epoch: int, 当前 epoch 编号
            model_eval: torch.nn.Module, 用于评估的模型（通常是 EMA 模型）
        
        功能:
            1. 检查是否到达评估周期
            2. 在验证集上进行检测，保存检测结果
            3. 保存真实标签
            4. 计算 mAP
            5. 绘制 mAP 曲线图
            6. 清理临时文件
        """
        # 检查是否到达评估周期且启用评估
        if epoch % self.period == 0 and self.eval_flag:
            # 更新模型（使用 EMA 模型进行评估）
            self.net = model_eval
            
            # 创建临时目录
            if not os.path.exists(self.map_out_path):
                os.makedirs(self.map_out_path)
            if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
                os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
            if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
                os.makedirs(os.path.join(self.map_out_path, "detection-results"))
            
            print("Get map.")
            
            # 遍历验证集，进行检测
            for annotation_line in tqdm(self.val_lines):
                line        = annotation_line.split()
                image_id    = os.path.basename(line[0]).split('.')[0]  # 提取图像 ID
                
                #------------------------------#
                #   读取图像并转换成 RGB 图像
                #------------------------------#
                image       = Image.open(line[0])
                
                #------------------------------#
                #   解析真实标签框
                #   格式：left,top,right,bottom,class_id
                #------------------------------#
                gt_boxes    = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
                
                #------------------------------#
                #   对图像进行检测，保存检测结果到文本文件
                #------------------------------#
                self.get_map_txt(image_id, image, self.class_names, self.map_out_path)
                
                #------------------------------#
                #   保存真实标签框到文本文件
                #   格式：类别名 x1 y1 x2 y2
                #------------------------------#
                with open(os.path.join(self.map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                    for box in gt_boxes:
                        left, top, right, bottom, obj = box
                        obj_name = self.class_names[obj]
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
                        
            print("Calculate Map.")
            
            # 计算 mAP：优先使用 COCO 格式，失败则使用 VOC 格式
            try:
                # COCO 格式的 mAP 计算（返回多个指标）
                temp_map = get_coco_map(class_names = self.class_names, path = self.map_out_path)[1]
            except:
                # VOC 格式的 mAP 计算（备用方案）
                temp_map = get_map(self.MINOVERLAP, False, path = self.map_out_path)
            
            # 记录 mAP 值
            self.maps.append(temp_map)
            self.epoches.append(epoch)

            # 将 mAP 值追加到文本文件
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(temp_map))
                f.write("\n")
            
            # 绘制 mAP 曲线图
            plt.figure()
            plt.plot(self.epoches, self.maps, 'red', linewidth = 2, label='train map')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Map %s'%str(self.MINOVERLAP))
            plt.title('A Map Curve')
            plt.legend(loc="upper right")

            # 保存图片
            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")

            print("Get map done.")
            
            # 清理临时文件（释放磁盘空间）
            shutil.rmtree(self.map_out_path)