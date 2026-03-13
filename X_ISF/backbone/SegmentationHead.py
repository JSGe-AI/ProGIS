import torch
import torch.nn as nn
import torch.nn.functional as F
# 定义分割头的结构
class SegmentationHead(nn.Module):
    def __init__(self, in_features, out_classes, scale_factor=2):
        super(SegmentationHead, self).__init__()
        # 1x1 卷积用于特征维度变换
        self.conv1x1 = nn.Conv2d(in_features, out_classes, kernel_size=1)  
        # 双线性上采样恢复分辨率
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)  
 
    def forward(self, x):
        # 应用1x1卷积进行特征变换
        x = self.conv1x1(x) 
        # 上采样到目标分辨率 
        x = self.upsample(x)  
        return x
 
# 示例初始化
# 假设从Swin Transformer提取的特征图通道数为512，目标分割类别数为20
seg_head = SegmentationHead(in_features=512, out_classes=20, scale_factor=4)