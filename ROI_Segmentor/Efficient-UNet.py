import torch
import torch.nn as nn
from torchvision import models

class EfficientNetUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=6, model_name='efficientnet_b0'):
        super(EfficientNetUNet, self).__init__()
        
        # 1. EfficientNet 编码器部分
        efficient_net = getattr(models, model_name)(pretrained=True)
        self.encoder = efficient_net.features

        # 替换第一个卷积层，使其适应多通道输入
        self.encoder[0][0] = nn.Conv2d(input_channels, 
                                       self.encoder[0][0].out_channels, 
                                       kernel_size=self.encoder[0][0].kernel_size, 
                                       stride=self.encoder[0][0].stride, 
                                       padding=self.encoder[0][0].padding, 
                                       bias=False)
        
        # 2. 解码器部分（U-Net 风格的上采样解码器）
        self.decoder1 = self._decoder_block(320 + 112, 128)  # 对应 Encoder 中的两个特征图
        self.decoder2 = self._decoder_block(128 + 40, 64)
        self.decoder3 = self._decoder_block(64 + 24, 32)
        self.decoder4 = self._decoder_block(32 + 16, 16)
        
        # 3. 最后的1x1卷积用于生成预测
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)
    
    def _decoder_block(self, in_channels, out_channels):
        """解码器的单个解码块，包括反卷积（上采样）和两个卷积层."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),  # 上采样
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 编码器前向传播，提取各个阶段的特征
        features = []
        for layer in self.encoder:
            x = layer(x)
            features.append(x)
        
        # 特征层的选择与跳跃连接
        d4 = self.decoder1(features[-1], features[-4])  # 与 encoder 第 -4 层进行跳跃连接
        d3 = self.decoder2(d4, features[-5])            # 与 encoder 第 -5 层进行跳跃连接
        d2 = self.decoder3(d3, features[-6])            # 与 encoder 第 -6 层进行跳跃连接
        d1 = self.decoder4(d2, features[-7])            # 与 encoder 第 -7 层进行跳跃连接
        
        # 最终的预测输出
        out = self.final_conv(d1)
        
        return out

# 测试模型
model = EfficientNetUNet(num_classes=1, input_channels=6, model_name='efficientnet_b0')
x = torch.randn(1, 6, 256, 256)  # 输入是 6 通道，256x256 大小的图像
output = model(x)
print(output.shape)  # 检查输出形状是否正确
