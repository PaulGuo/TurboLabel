import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import autopad

class SimAMWithSlicing(nn.Module):
    def __init__(self,e_lambda=1e-4):
        super(SimAMWithSlicing, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()

        block_size_h = height // 2
        block_size_w = width // 2

        block1 = x[:, :, :block_size_h, :block_size_w]
        block2 = x[:, :, :block_size_h, block_size_w:]
        block3 = x[:, :, block_size_h:, :block_size_w]
        block4 = x[:, :, block_size_h:, block_size_w:]

        enhanced_blocks = []
        for block in [block1, block2, block3, block4]:
            n = block_size_h * block_size_w - 1
            block_minus_mu_square = (block - block.mean(dim=[2, 3], keepdim=True)).pow(2)
            y = block_minus_mu_square / (
                        4 * (block_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
            enhanced_blocks.append(block * self.activation(y))

        enhanced_image = torch.cat([torch.cat([enhanced_blocks[0], enhanced_blocks[1]], dim=3),
                                    torch.cat([enhanced_blocks[2], enhanced_blocks[3]], dim=3)], dim=2)

        return enhanced_image

class Conv_SWS(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv_SWS, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.att = SimAMWithSlicing(c2)

    def forward(self, x):
        return self.att(self.act(self.bn(self.conv(x))))

    def fuseforward(self, x):
        return self.att(self.act(self.conv(x)))

class SimAMWith4x4Slicing(nn.Module):
    def __init__(self,e_lambda=1e-4):
        super(SimAMWith4x4Slicing, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()

        # 检查高度和宽度是否能被 4 整除，如果不能则进行填充
        pad_h = (4 - height % 4) % 4  # 计算高度需要填充的数量
        pad_w = (4 - width % 4) % 4   # 计算宽度需要填充的数量

        # 在右边和下方进行填充，确保可以均匀分成 4x4 块
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # Pad in the order of (left, right, top, bottom)

        # 重新计算填充后的高度和宽度
        height, width = x.size(2), x.size(3)
        block_size_h = height // 4
        block_size_w = width // 4

        # 使用嵌套循环将特征图分为 4x4 的子块
        enhanced_blocks = []
        for i in range(4):  # 切分行
            row_blocks = []  # 存储当前行的所有子块
            for j in range(4):  # 切分列
                # 提取每个子块 (C, H/4, W/4)
                block = x[:, :, i * block_size_h: (i + 1) * block_size_h,
                                j * block_size_w: (j + 1) * block_size_w]

                # 计算注意力权重
                n = block_size_h * block_size_w - 1
                block_minus_mu_square = (block - block.mean(dim=[2, 3], keepdim=True)).pow(2)
                y = block_minus_mu_square / (4 * (block_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

                # 注意力加权
                enhanced_block = block * self.activation(y)
                row_blocks.append(enhanced_block)

            # 将每行的 4 个子块拼接成一个宽度方向上的整体
            enhanced_row = torch.cat(row_blocks, dim=3)
            enhanced_blocks.append(enhanced_row)

        # 将 4 行拼接成完整的增强特征图
        enhanced_image = torch.cat(enhanced_blocks, dim=2)

        # 去除填充（如果存在）
        if pad_h > 0 or pad_w > 0:
            enhanced_image = enhanced_image[:, :, :height - pad_h, :width - pad_w]

        return enhanced_image

class Conv_SWS_4x4(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv_SWS_4x4, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.att = SimAMWith4x4Slicing(c2)

    def forward(self, x):
        return self.att(self.act(self.bn(self.conv(x))))

    def fuseforward(self, x):
        return self.att(self.act(self.conv(x)))

# Squeeze-and-Excitation Block，用于通道级别的权重动态调整
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 通过全局池化提取通道的全局信息
        y = self.fc(y).view(b, c, 1, 1)  # 全局信息通过 MLP 生成动态权重
        return x * y.expand_as(x)  # 将权重与输入特征逐通道相乘

# 改进版 SimAMWith4x4Slicing 模块，引入动态权重调整策略
class DynamicSimAMWith4x4Slicing(nn.Module):
    def __init__(self, in_channels, e_lambda=1e-4, dynamic=True):
        super(DynamicSimAMWith4x4Slicing, self).__init__()
        self.e_lambda = e_lambda
        self.dynamic = dynamic  # 是否使用动态调整策略
        self.activation = nn.Sigmoid()
        self.se_block = SEBlock(in_channels)  # 使用 SE Block 进行动态通道权重调整

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()

        # 检查高度和宽度是否能被 4 整除，如果不能则进行填充
        pad_h = (4 - height % 4) % 4  # 计算高度需要填充的数量
        pad_w = (4 - width % 4) % 4   # 计算宽度需要填充的数量

        # 在右边和下方进行填充，确保可以均匀分成 4x4 块
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # Pad in the order of (left, right, top, bottom)

        # 重新计算填充后的高度和宽度
        height, width = x.size(2), x.size(3)
        block_size_h = height // 4
        block_size_w = width // 4

        # 使用嵌套循环将特征图分为 4x4 的子块
        enhanced_blocks = []
        for i in range(4):  # 切分行
            row_blocks = []  # 存储当前行的所有子块
            for j in range(4):  # 切分列
                # 提取每个子块 (C, H/4, W/4)
                block = x[:, :, i * block_size_h: (i + 1) * block_size_h,
                                j * block_size_w: (j + 1) * block_size_w]

                # 动态调整 e_lambda 参数
                if self.dynamic:
                    global_mean = x.mean()
                    dynamic_lambda = self.e_lambda * torch.log(1 + torch.abs(global_mean))  # 基于全局均值动态调整 e_lambda
                else:
                    dynamic_lambda = self.e_lambda

                # 计算注意力权重
                n = block_size_h * block_size_w - 1
                block_minus_mu_square = (block - block.mean(dim=[2, 3], keepdim=True)).pow(2)
                y = block_minus_mu_square / (4 * (block_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + dynamic_lambda)) + 0.5

                # 使用 SE Block 进行通道级别的动态加权
                enhanced_block = block * self.activation(y)
                enhanced_block = self.se_block(enhanced_block)

                row_blocks.append(enhanced_block)

            # 将每行的 4 个子块拼接成一个宽度方向上的整体
            enhanced_row = torch.cat(row_blocks, dim=3)
            enhanced_blocks.append(enhanced_row)

        # 将 4 行拼接成完整的增强特征图
        enhanced_image = torch.cat(enhanced_blocks, dim=2)

        # 去除填充（如果存在）
        if pad_h > 0 or pad_w > 0:
            enhanced_image = enhanced_image[:, :, :height - pad_h, :width - pad_w]

        # 全局特征残差连接，保留全局特征的一部分，避免过度调整
        # enhanced_image = enhanced_image + x * 0.1  # 添加全局特征残差连接

        return enhanced_image

# 定义动态权重标准卷积模块
class DSWSConv(nn.Module):
    # Standard convolution with dynamic weight adjustment
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(DSWSConv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.att = DynamicSimAMWith4x4Slicing(in_channels=c2)  # 使用动态权重调整的 SimAM 注意力模块

    def forward(self, x):
        return self.att(self.act(self.bn(self.conv(x))))

    def fuseforward(self, x):
        return self.att(self.act(self.conv(x)))

# 改进版 SimAMWith2x2Slicing 模块，引入动态权重调整策略
class DynamicSimAMWith2x2Slicing(nn.Module):
    def __init__(self, in_channels, e_lambda=1e-4, dynamic=True):
        super(DynamicSimAMWith2x2Slicing, self).__init__()
        self.e_lambda = e_lambda
        self.dynamic = dynamic  # 是否使用动态调整策略
        self.activation = nn.Sigmoid()
        self.se_block = SEBlock(in_channels)  # 使用 SE Block 进行动态通道权重调整

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()

        # 检查高度和宽度是否能被 2 整除，如果不能则进行填充
        pad_h = (2 - height % 2) % 2  # 计算高度需要填充的数量
        pad_w = (2 - width % 2) % 2   # 计算宽度需要填充的数量

        # 在右边和下方进行填充，确保可以均匀分成 2x2 块
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # Pad in the order of (left, right, top, bottom)

        # 重新计算填充后的高度和宽度
        height, width = x.size(2), x.size(3)
        block_size_h = height // 2
        block_size_w = width // 2

        # 使用嵌套循环将特征图分为 2x2 的子块
        enhanced_blocks = []
        for i in range(2):  # 切分行
            row_blocks = []  # 存储当前行的所有子块
            for j in range(2):  # 切分列
                # 提取每个子块 (C, H/2, W/2)
                block = x[:, :, i * block_size_h: (i + 1) * block_size_h,
                                j * block_size_w: (j + 1) * block_size_w]

                # 动态调整 e_lambda 参数
                if self.dynamic:
                    global_mean = x.mean()
                    dynamic_lambda = self.e_lambda * torch.log(1 + torch.abs(global_mean))  # 基于全局均值动态调整 e_lambda
                else:
                    dynamic_lambda = self.e_lambda

                # 计算注意力权重
                n = block_size_h * block_size_w - 1
                block_minus_mu_square = (block - block.mean(dim=[2, 3], keepdim=True)).pow(2)
                y = block_minus_mu_square / (4 * (block_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + dynamic_lambda)) + 0.5

                # 使用 SE Block 进行通道级别的动态加权
                enhanced_block = block * self.activation(y)
                enhanced_block = self.se_block(enhanced_block)

                row_blocks.append(enhanced_block)

            # 将每行的 2 个子块拼接成一个宽度方向上的整体
            enhanced_row = torch.cat(row_blocks, dim=3)
            enhanced_blocks.append(enhanced_row)

        # 将 2 行拼接成完整的增强特征图
        enhanced_image = torch.cat(enhanced_blocks, dim=2)

        # 去除填充（如果存在）
        if pad_h > 0 or pad_w > 0:
            enhanced_image = enhanced_image[:, :, :height - pad_h, :width - pad_w]

        return enhanced_image

class DSWSConv2x2(nn.Module):
    # Standard convolution with dynamic weight adjustment
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(DSWSConv2x2, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.att = DynamicSimAMWith2x2Slicing(in_channels=c2)  # 使用动态权重调整的 SimAM 注意力模块

    def forward(self, x):
        return self.att(self.act(self.bn(self.conv(x))))

    def fuseforward(self, x):
        return self.att(self.act(self.conv(x)))