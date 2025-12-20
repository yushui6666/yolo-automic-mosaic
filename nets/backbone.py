#-----------------------------------------------------------------------#
#   backbone.py：YOLO 网络的主干结构（Backbone）
#   功能概述：
#   1. 定义基础卷积模块（Conv、Bottleneck、C2f、SPPF）
#   2. 构建特征提取主干网络（Backbone）
#   3. 实现多尺度特征提取（3 个有效特征层）
#-----------------------------------------------------------------------#
import torch
import torch.nn as nn


def autopad(k, p=None, d=1):  
    """
    自动计算卷积的 padding 值，实现 "Same" 卷积效果
    
    参数:
        k: int 或 list, 卷积核大小（kernel size）
        p: int 或 list 或 None, 手动指定的 padding 值
           - 如果为 None，则自动计算
        d: int, 膨胀率（dilation），默认 1
    
    返回:
        int 或 list: 计算得到的 padding 值
    
    原理:
        - "Same" 卷积：输出尺寸 = 输入尺寸 / stride
        - 当 dilation > 1 时，有效卷积核大小 = d * (k - 1) + 1
        - padding = (有效卷积核大小 - 1) // 2
    
    示例:
        autopad(3) -> 1  # 3x3 卷积，padding=1
        autopad(5, d=2) -> 4  # 5x5 卷积，dilation=2，有效核大小=9，padding=4
    """
    # kernel, padding, dilation
    # 对输入的特征层进行自动 padding，按照 Same 原则
    if d > 1:
        # 计算膨胀卷积的有效卷积核大小
        # 有效核大小 = dilation * (kernel_size - 1) + 1
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    
    if p is None:
        # 自动计算 padding：padding = (kernel_size - 1) // 2
        # 这样可以实现 "Same" 卷积（输出尺寸 = 输入尺寸 / stride）
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    
    return p

class SiLU(nn.Module):  
    """
    SiLU（Sigmoid Linear Unit）激活函数，也称为 Swish 激活函数
    
    公式: SiLU(x) = x * sigmoid(x)
    
    特点:
        - 平滑、非单调激活函数
        - 在 YOLO 等现代目标检测网络中广泛使用
        - 相比 ReLU，在负值区域有小的梯度，有助于梯度流动
    
    参数:
        x: torch.Tensor, 输入张量
    
    返回:
        torch.Tensor: 激活后的张量
    """
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
    
class Conv(nn.Module):
    """
    标准卷积模块：卷积 + 批归一化 + 激活函数
    
    这是 YOLO 网络中最基础的构建块，组合了：
    1. 2D 卷积（Conv2d）
    2. 批归一化（BatchNorm2d）
    3. 激活函数（默认 SiLU）
    
    参数:
        c1: int, 输入通道数
        c2: int, 输出通道数
        k: int, 卷积核大小，默认 1
        s: int, 步长（stride），默认 1
        p: int 或 None, padding 值，None 时自动计算
        g: int, 分组卷积的组数，默认 1（普通卷积）
        d: int, 膨胀率（dilation），默认 1
        act: bool 或 nn.Module, 激活函数
             - True: 使用默认的 SiLU 激活
             - False: 不使用激活（Identity）
             - nn.Module: 使用指定的激活函数
    
    注意:
        - 卷积层不使用偏置（bias=False），因为后面有 BatchNorm
        - forward_fuse: 融合模式（用于推理加速），跳过 BatchNorm
    """
    default_act = SiLU()  # 默认激活函数
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        # 2D 卷积层（不使用偏置，因为后面有 BatchNorm）
        self.conv   = nn.Conv2d(
            c1, c2, k, s, 
            autopad(k, p, d),  # 自动计算 padding
            groups=g,          # 分组卷积
            dilation=d,        # 膨胀卷积
            bias=False         # 不使用偏置
        )
        # 批归一化层
        self.bn     = nn.BatchNorm2d(
            c2, 
            eps=0.001,              # 防止除零的小值
            momentum=0.03,          # 移动平均的动量
            affine=True,            # 使用可学习的缩放和偏移
            track_running_stats=True # 跟踪运行统计
        )
        # 激活函数
        self.act    = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        标准前向传播：卷积 -> 批归一化 -> 激活
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        融合模式前向传播：卷积 -> 激活（跳过 BatchNorm）
        
        用于模型融合（fuse）后的推理加速
        """
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    """
    标准瓶颈结构（Bottleneck），带残差连接
    
    结构:
        x -> Conv(c1 -> c_) -> Conv(c_ -> c2) -> output
        |                                           |
        +-------------------------------------------+
        (如果 shortcut=True 且 c1 == c2)
    
    参数:
        c1: int, 输入通道数
        c2: int, 输出通道数
        shortcut: bool, 是否使用残差连接，默认 True
        g: int, 第二个卷积的分组数，默认 1
        k: tuple, 两个卷积的卷积核大小，默认 (3, 3)
        e: float, 扩展比例，隐藏层通道数 = c2 * e，默认 0.5
    
    特点:
        - 先降维再升维，减少计算量
        - 残差连接有助于梯度流动和特征学习
        - 只有当输入输出通道数相等时才使用残差连接
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # 隐藏层通道数（扩展比例）
        
        # 第一个卷积：降维
        self.cv1 = Conv(c1, c_, k[0], 1)
        # 第二个卷积：升维（可能使用分组卷积）
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        
        # 是否使用残差连接（需要输入输出通道数相等）
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        前向传播
        
        如果启用残差连接且通道数匹配，则输出 = x + conv2(conv1(x))
        否则输出 = conv2(conv1(x))
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))    

class C2f(nn.Module):
    """
    C2f 模块：改进的 CSP（Cross Stage Partial）结构
    
    这是 YOLOv8 中使用的特征提取模块，结合了 CSP 和密集残差连接
    
    结构:
        x -> Conv(c1 -> 2c) -> split -> [y0, y1]
        y0 -> (直接保留)
        y1 -> Bottleneck -> Bottleneck -> ... (n 个)
        concat([y0, y1, bottleneck_outputs...]) -> Conv -> output
    
    参数:
        c1: int, 输入通道数
        c2: int, 输出通道数
        n: int, Bottleneck 模块的数量，默认 1
        shortcut: bool, Bottleneck 中是否使用残差连接，默认 False
        g: int, Bottleneck 中分组卷积的组数，默认 1
        e: float, 扩展比例，隐藏层通道数 = c2 * e，默认 0.5
    
    特点:
        - 将特征分成两部分，一部分直接传递，另一部分经过多个 Bottleneck
        - 所有中间结果都保留并拼接，形成密集残差连接
        - 相比传统 CSP，信息流动更充分
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c      = int(c2 * e)  # 隐藏层通道数
        
        # 第一个卷积：将输入通道数扩展为 2c
        self.cv1    = Conv(c1, 2 * self.c, 1, 1)
        
        # 最后一个卷积：将所有特征融合回 c2 通道
        self.cv2    = Conv((2 + n) * self.c, c2, 1)
        
        # n 个 Bottleneck 模块
        self.m      = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) 
            for _ in range(n)
        )

    def forward(self, x):
        """
        前向传播
        
        流程:
        1. 将输入通过 cv1 卷积，得到 2c 通道的特征
        2. 将特征分成两部分，每部分 c 通道：[y0, y1]
        3. y0 直接保留，y1 依次通过 n 个 Bottleneck
        4. 每次 Bottleneck 的输出都保留
        5. 将所有特征拼接：[y0, y1, bottleneck1_out, bottleneck2_out, ...]
        6. 通过 cv2 卷积融合所有特征
        """
        # 进行一个卷积，然后划分成两份，每个通道都为 c
        y = list(self.cv1(x).split((self.c, self.c), 1))
        
        # 每进行一次残差结构都保留，然后堆叠在一起，密集残差
        y.extend(m(y[-1]) for m in self.m)
        
        return self.cv2(torch.cat(y, 1))
    
class SPPF(nn.Module):
    """
    SPPF（Spatial Pyramid Pooling - Fast）模块
    
    快速空间金字塔池化，通过串联多个最大池化层实现多尺度特征提取
    
    结构:
        x -> Conv(c1 -> c_) -> MaxPool(k) -> y1
        y1 -> MaxPool(k) -> y2
        y2 -> MaxPool(k) -> y3
        concat([x, y1, y2, y3]) -> Conv -> output
    
    参数:
        c1: int, 输入通道数
        c2: int, 输出通道数
        k: int, 最大池化核大小，默认 5
    
    特点:
        - 相比传统 SPP，使用串联而非并行，计算更高效
        - 通过多次池化获得不同感受野的特征（等效于 5x5, 9x9, 13x13 池化）
        - 增强模型对不同尺度目标的检测能力
    
    注意:
        - 输出特征包含 4 个不同尺度的池化结果
        - 通道数从 c_ 变为 c_ * 4，最后融合为 c2
    """
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_          = c1 // 2  # 中间通道数（降维）
        
        # 第一个卷积：降维
        self.cv1    = Conv(c1, c_, 1, 1)
        
        # 最后一个卷积：融合多尺度特征
        self.cv2    = Conv(c_ * 4, c2, 1, 1)
        
        # 最大池化层（stride=1，padding=k//2，保持尺寸不变）
        self.m      = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """
        前向传播
        
        通过串联的池化操作，获得不同感受野的特征：
        - x: 原始特征（感受野最小）
        - y1: 一次池化后的特征
        - y2: 两次池化后的特征
        - y3: 三次池化后的特征（感受野最大）
        """
        x = self.cv1(x)
        y1 = self.m(x)      # 第一次池化
        y2 = self.m(y1)     # 第二次池化
        y3 = self.m(y2)     # 第三次池化
        
        # 拼接所有尺度的特征并融合
        return self.cv2(torch.cat((x, y1, y2, y3), 1))

class Backbone(nn.Module):
    """
    YOLO 主干网络（Backbone）
    
    从输入图像中提取多尺度特征，输出 3 个有效特征层用于目标检测
    
    网络结构（以 640x640 输入为例）:
        Input: 3, 640, 640
        Stem: 3 -> base_channels (下采样 2 倍)
        Dark2: base_channels -> base_channels*2 (下采样 2 倍)
        Dark3: base_channels*2 -> base_channels*4 (下采样 2 倍) -> feat1 (80x80)
        Dark4: base_channels*4 -> base_channels*8 (下采样 2 倍) -> feat2 (40x40)
        Dark5: base_channels*8 -> base_channels*16*deep_mul (下采样 2 倍) -> feat3 (20x20)
    
    参数:
        base_channels: int, 基础通道数（通常为 64）
        base_depth: int, 基础深度（C2f 模块中 Bottleneck 的数量）
        deep_mul: float, 深层通道数倍数（用于调整模型大小）
        phi: str, 模型变体标识符（'n', 's', 'm', 'l', 'x'）
        pretrained: bool, 是否使用预训练权重（当前未实现）
    
    输出:
        feat1: torch.Tensor, 形状为 (B, base_channels*4, H/8, W/8)
               - 用于检测小目标
        feat2: torch.Tensor, 形状为 (B, base_channels*8, H/16, W/16)
               - 用于检测中等目标
        feat3: torch.Tensor, 形状为 (B, base_channels*16*deep_mul, H/32, W/32)
               - 用于检测大目标
    """
    def __init__(self, base_channels, base_depth, deep_mul, phi, pretrained=False):
        super().__init__()
        #-----------------------------------------------#
        #   输入图片是 3, 640, 640
        #-----------------------------------------------#
        # 3, 640, 640 => base_channels, 320, 320
        # Stem：初始特征提取层（下采样 2 倍）
        self.stem = Conv(3, base_channels, 3, 2)
        
        # base_channels, 320, 320 => base_channels*2, 160, 160
        # Dark2：第一个特征提取阶段
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),  # 下采样
            C2f(base_channels * 2, base_channels * 2, base_depth, True),  # 特征提取
        )
        
        # base_channels*2, 160, 160 => base_channels*4, 80, 80
        # Dark3：第二个特征提取阶段（输出有效特征层 1）
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),  # 下采样
            C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),  # 特征提取（深度加倍）
        )
        
        # base_channels*4, 80, 80 => base_channels*8, 40, 40
        # Dark4：第三个特征提取阶段（输出有效特征层 2）
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),  # 下采样
            C2f(base_channels * 8, base_channels * 8, base_depth * 2, True),  # 特征提取（深度加倍）
        )
        
        # base_channels*8, 40, 40 => base_channels*16*deep_mul, 20, 20
        # Dark5：第四个特征提取阶段（输出有效特征层 3）
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),  # 下采样
            C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, True),  # 特征提取
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5)  # 多尺度特征融合
        )
        

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: torch.Tensor, 输入图像，形状为 (B, 3, H, W)
        
        返回:
            feat1: torch.Tensor, 第一个有效特征层，用于检测小目标
            feat2: torch.Tensor, 第二个有效特征层，用于检测中等目标
            feat3: torch.Tensor, 第三个有效特征层，用于检测大目标
        """
        x = self.stem(x)    # 初始特征提取
        x = self.dark2(x)   # 第一阶段
        
        #-----------------------------------------------#
        #   dark3 的输出为 base_channels*4, H/8, W/8
        #   这是一个有效特征层，用于检测小目标
        #-----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        
        #-----------------------------------------------#
        #   dark4 的输出为 base_channels*8, H/16, W/16
        #   这是一个有效特征层，用于检测中等目标
        #-----------------------------------------------#
        x = self.dark4(x)
        feat2 = x
        
        #-----------------------------------------------#
        #   dark5 的输出为 base_channels*16*deep_mul, H/32, W/32
        #   这是一个有效特征层，用于检测大目标
        #-----------------------------------------------#
        x = self.dark5(x)
        feat3 = x
        
        return feat1, feat2, feat3
