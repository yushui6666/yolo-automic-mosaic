#-----------------------------------------------------------------------#
#   yolo.py：YOLO 主体网络结构
#   功能概述：
#   1. YoloBody：完整的 YOLO 检测网络（Backbone + Neck + Head）
#   2. DFL：分布焦点损失模块（Distribution Focal Loss）
#   3. fuse_conv_and_bn：融合卷积和批归一化层（推理加速）
#-----------------------------------------------------------------------#
import numpy as np
import torch
import torch.nn as nn

from nets.backbone import Backbone, C2f, Conv
from nets.yolo_training import weights_init
from utils.utils_bbox import make_anchors

def fuse_conv_and_bn(conv, bn):
    """
    融合 Conv2d 和 BatchNorm2d 层，减少推理时的计算量
    
    原理:
        将 BatchNorm 的参数合并到 Conv 的权重和偏置中：
        - 新权重 = BN.weight / sqrt(BN.running_var + eps) * Conv.weight
        - 新偏置 = BN.bias - BN.weight * BN.running_mean / sqrt(BN.running_var + eps)
    
    参考:
        https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    
    参数:
        conv: nn.Conv2d, 卷积层
        bn: nn.BatchNorm2d, 批归一化层
    
    返回:
        nn.Conv2d: 融合后的卷积层（包含偏置）
    
    注意:
        - 融合后的层不需要 BatchNorm，可以提升推理速度
        - 只影响推理，不影响训练
    """
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # 准备融合后的权重（kernel）
    # 将卷积权重展平为 2D 矩阵
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    # 计算 BatchNorm 的缩放因子（对角矩阵）
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    # 融合权重：w_fused = w_bn @ w_conv
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # 准备融合后的偏置（bias）
    # 如果原卷积层没有偏置，则初始化为 0
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    # 计算 BatchNorm 的偏置项
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    # 融合偏置：b_fused = w_bn @ b_conv + b_bn
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

class DFL(nn.Module):
    """
    DFL（Distribution Focal Loss）模块
    
    用于将边界框的分布表示转换为具体的坐标值
    
    原理:
        模型预测的不是直接的坐标值，而是坐标的分布（0 到 reg_max-1 的概率分布）
        DFL 通过加权求和将分布转换为具体的坐标值：
        coord = sum(prob[i] * i) for i in range(reg_max)
    
    参考论文:
        Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes
        https://ieeexplore.ieee.org/document/9792391
    
    参数:
        c1: int, 分布通道数（reg_max），默认 16
            - 表示坐标值被离散化为 0 到 c1-1 的整数
    
    输入:
        x: torch.Tensor, 形状为 (B, reg_max*4, num_anchors)
          - 4 个坐标（左上右下）的分布预测
    
    输出:
        torch.Tensor, 形状为 (B, 4, num_anchors)
          - 4 个坐标的具体值
    """
    def __init__(self, c1=16):
        super().__init__()
        # 1x1 卷积，用于加权求和（权重固定为 [0, 1, 2, ..., c1-1]）
        self.conv   = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        
        # 初始化权重为 [0, 1, 2, ..., c1-1]，用于加权求和
        x           = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1     = c1

    def forward(self, x):
        """
        前向传播：将分布转换为坐标值
        
        参数:
            x: torch.Tensor, 形状为 (B, reg_max*4, num_anchors)
        
        返回:
            torch.Tensor, 形状为 (B, 4, num_anchors)
        """
        # bs, self.reg_max * 4, num_anchors
        b, c, a = x.shape
        
        # 重塑为 (B, 4, reg_max, num_anchors)
        # 转置为 (B, reg_max, 4, num_anchors)
        # 对 reg_max 维度进行 softmax，得到概率分布
        # 通过卷积（加权求和）得到最终坐标值
        # 重塑为 (B, 4, num_anchors)
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        
#---------------------------------------------------#
#   YOLO 主体网络
#---------------------------------------------------#
class YoloBody(nn.Module):
    """
    完整的 YOLO 检测网络
    
    结构组成:
        1. Backbone：特征提取主干网络
        2. Neck：特征融合网络（FPN + PAN）
        3. Head：检测头（分类 + 回归）
    
    网络流程:
        Input (3, 640, 640)
        -> Backbone (提取 3 个特征层)
        -> Neck (上采样 + 下采样，融合多尺度特征)
        -> Head (输出分类和回归结果)
    
    参数:
        input_shape: list/tuple, 输入图像尺寸 [height, width]
        num_classes: int, 类别数量
        phi: str, 模型变体
            - 'n': nano（最小）
            - 's': small（小）
            - 'm': medium（中等）
            - 'l': large（大）
            - 'x': xlarge（超大）
        pretrained: bool, 是否使用预训练权重
    
    输出:
        dbox: torch.Tensor, 解码后的边界框坐标
        cls: torch.Tensor, 类别预测
        x: list, 原始特征（用于训练）
        anchors: torch.Tensor, 锚点坐标
        strides: torch.Tensor, 步长
    """
    def __init__(self, input_shape, num_classes, phi, pretrained=False):
        super(YoloBody, self).__init__()
        # 不同模型变体的配置参数
        depth_dict          = {'n' : 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.00,}  # 深度倍数
        width_dict          = {'n' : 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}  # 宽度倍数
        deep_width_dict     = {'n' : 1.00, 's' : 1.00, 'm' : 0.75, 'l' : 0.50, 'x' : 0.50,}  # 深层宽度倍数
        
        # 获取当前模型变体的参数
        dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]

        # 计算基础通道数和深度
        base_channels       = int(wid_mul * 64)  # 基础通道数（通常为 64）
        base_depth          = max(round(dep_mul * 3), 1)  # 基础深度（至少为 1）
        
        #-----------------------------------------------#
        #   输入图片是 3, 640, 640
        #-----------------------------------------------#

        #---------------------------------------------------#   
        #   生成主干模型（Backbone）
        #   获得三个有效特征层，他们的 shape 分别是：
        #   - feat1: base_channels*4, H/8, W/8  (例如: 256, 80, 80)
        #   - feat2: base_channels*8, H/16, W/16 (例如: 512, 40, 40)
        #   - feat3: base_channels*16*deep_mul, H/32, W/32 (例如: 1024*deep_mul, 20, 20)
        #---------------------------------------------------#
        self.backbone   = Backbone(base_channels, base_depth, deep_mul, phi, pretrained=pretrained)

        #------------------------特征融合网络（Neck：FPN + PAN）------------------------# 
        # 上采样层：用于将深层特征上采样到浅层
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        # FPN 路径（自顶向下）：融合深层和浅层特征
        # 1024 * deep_mul + 512, 40, 40 => 512, 40, 40
        self.conv3_for_upsample1    = C2f(
            int(base_channels * 16 * deep_mul) + base_channels * 8, 
            base_channels * 8, 
            base_depth, 
            shortcut=False
        )
        # 768, 80, 80 => 256, 80, 80
        self.conv3_for_upsample2    = C2f(
            base_channels * 8 + base_channels * 4, 
            base_channels * 4, 
            base_depth, 
            shortcut=False
        )
        
        # PAN 路径（自底向上）：再次融合特征
        # 256, 80, 80 => 256, 40, 40
        self.down_sample1           = Conv(base_channels * 4, base_channels * 4, 3, 2)
        # 512 + 256, 40, 40 => 512, 40, 40
        self.conv3_for_downsample1  = C2f(
            base_channels * 8 + base_channels * 4, 
            base_channels * 8, 
            base_depth, 
            shortcut=False
        )

        # 512, 40, 40 => 512, 20, 20
        self.down_sample2           = Conv(base_channels * 8, base_channels * 8, 3, 2)
        # 1024 * deep_mul + 512, 20, 20 => 1024 * deep_mul, 20, 20
        self.conv3_for_downsample2  = C2f(
            int(base_channels * 16 * deep_mul) + base_channels * 8, 
            int(base_channels * 16 * deep_mul), 
            base_depth, 
            shortcut=False
        )
        #------------------------特征融合网络------------------------# 
        
        # 三个特征层的通道数
        ch              = [base_channels * 4, base_channels * 8, int(base_channels * 16 * deep_mul)]
        self.shape      = None  # 用于缓存输入形状
        self.nl         = len(ch)  # 特征层数量（3）
        
        # 计算每个特征层的步长（相对于输入图像）
        # 通过前向传播一个虚拟输入来计算
        self.stride     = torch.tensor([
            256 / x.shape[-2] 
            for x in self.backbone.forward(torch.zeros(1, 3, 256, 256))
        ])
        
        self.reg_max    = 16  # DFL 的分布通道数（用于边界框回归）
        self.no         = num_classes + self.reg_max * 4  # 每个锚点的输出通道数
        self.num_classes = num_classes
        
        # 检测头的通道数
        c2, c3   = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], num_classes)
        
        # 回归头（边界框预测）：输出 4 * reg_max 通道（4 个坐标的分布）
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3), 
                Conv(c2, c2, 3), 
                nn.Conv2d(c2, 4 * self.reg_max, 1)
            ) for x in ch
        )
        
        # 分类头（类别预测）：输出 num_classes 通道
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), 
                Conv(c3, c3, 3), 
                nn.Conv2d(c3, num_classes, 1)
            ) for x in ch
        )
        
        # 初始化权重（如果未使用预训练权重）
        if not pretrained:
            weights_init(self)
        
        # DFL 模块（用于将分布转换为坐标值）
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()


    def fuse(self):
        """
        融合模型中的 Conv 和 BatchNorm 层，加速推理
        
        功能:
            遍历所有模块，将 Conv 模块中的卷积层和批归一化层融合
        
        返回:
            self: 返回自身（支持链式调用）
        
        注意:
            - 融合后模型只能用于推理，不能继续训练
            - 可以提升推理速度，减少内存占用
        """
        print('Fusing layers... ')
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                # 融合卷积和批归一化
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                # 删除批归一化层
                delattr(m, 'bn')
                # 更新前向传播方法（使用融合版本）
                m.forward = m.forward_fuse
        return self
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: torch.Tensor, 输入图像，形状为 (B, 3, H, W)
        
        返回:
            dbox: torch.Tensor, 解码后的边界框坐标，形状为 (B, 4, num_anchors)
            cls: torch.Tensor, 类别预测，形状为 (B, num_classes, num_anchors)
            x: list, 原始特征列表（用于训练），包含 3 个特征层
            anchors: torch.Tensor, 锚点坐标
            strides: torch.Tensor, 步长
        """
        # ==================== Backbone：特征提取 ====================
        feat1, feat2, feat3 = self.backbone.forward(x)
        
        # ==================== Neck：特征融合网络（FPN + PAN）====================
        # FPN 路径（自顶向下）：融合深层特征到浅层
        
        # 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 40, 40
        P5_upsample = self.upsample(feat3)
        # 1024 * deep_mul, 40, 40 cat 512, 40, 40 => 1024 * deep_mul + 512, 40, 40
        P4          = torch.cat([P5_upsample, feat2], 1)
        # 1024 * deep_mul + 512, 40, 40 => 512, 40, 40
        P4          = self.conv3_for_upsample1(P4)

        # 512, 40, 40 => 512, 80, 80
        P4_upsample = self.upsample(P4)
        # 512, 80, 80 cat 256, 80, 80 => 768, 80, 80
        P3          = torch.cat([P4_upsample, feat1], 1)
        # 768, 80, 80 => 256, 80, 80
        P3          = self.conv3_for_upsample2(P3)

        # PAN 路径（自底向上）：再次融合特征
        
        # 256, 80, 80 => 256, 40, 40
        P3_downsample = self.down_sample1(P3)
        # 512, 40, 40 cat 256, 40, 40 => 768, 40, 40
        P4 = torch.cat([P3_downsample, P4], 1)
        # 768, 40, 40 => 512, 40, 40
        P4 = self.conv3_for_downsample1(P4)

        # 512, 40, 40 => 512, 20, 20
        P4_downsample = self.down_sample2(P4)
        # 512, 20, 20 cat 1024 * deep_mul, 20, 20 => 1024 * deep_mul + 512, 20, 20
        P5 = torch.cat([P4_downsample, feat3], 1)
        # 1024 * deep_mul + 512, 20, 20 => 1024 * deep_mul, 20, 20
        P5 = self.conv3_for_downsample2(P5)
        
        # ==================== Head：检测头 ====================
        # P3: 256, 80, 80  (用于检测小目标)
        # P4: 512, 40, 40  (用于检测中等目标)
        # P5: 1024 * deep_mul, 20, 20  (用于检测大目标)
        shape = P3.shape  # BCHW
        
        # 对每个特征层进行检测预测
        # P3: 256, 80, 80 => num_classes + self.reg_max * 4, 80, 80
        # P4: 512, 40, 40 => num_classes + self.reg_max * 4, 40, 40
        # P5: 1024 * deep_mul, 20, 20 => num_classes + self.reg_max * 4, 20, 20
        x = [P3, P4, P5]
        for i in range(self.nl):
            # 拼接回归头和分类头的输出
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        # 如果输入形状改变，重新计算锚点和步长
        if self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        
        # 分离回归和分类预测
        # num_classes + self.reg_max * 4, num_anchors => 
        #   box: self.reg_max * 4, num_anchors
        #   cls: num_classes, num_anchors
        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split(
            (self.reg_max * 4, self.num_classes), 1
        )
        
        # 使用 DFL 将边界框分布转换为具体坐标值
        dbox = self.dfl(box)
        
        return dbox, cls, x, self.anchors.to(dbox.device), self.strides.to(dbox.device)