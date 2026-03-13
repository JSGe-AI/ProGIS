import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import models

import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm


import scipy.ndimage as ndi


import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class EfficientNetUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=5, model_name='efficientnet_b0'):
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
        
        # 2. 解码器部分（包含上采样操作）
        self.decoder1 = self._decoder_block(320 + 112, 128)  # 对应 Encoder 中的两个特征图
        self.decoder2 = self._decoder_block(128 + 40, 64)
        self.decoder3 = self._decoder_block(64 + 24, 32)
        self.decoder4 = self._decoder_block(32 + 16, 16)
        
        # 3. 最后的1x1卷积用于生成预测
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)
    
    def _decoder_block(self, in_channels, out_channels):
        """解码器的单个解码块，包括上采样（反卷积）和两个卷积层."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),  # 上采样
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x ,aux_input):
        x = torch.cat([x, aux_input], dim=1)
        # 编码器前向传播，提取各个阶段的特征
        features = []
        for layer in self.encoder:
            x = layer(x)
            features.append(x)
        
        # 特征层的选择与跳跃连接，拼接特征图
        d4 = self.decoder1(torch.cat([self._upsample(features[-1], features[-4]), features[-4]], dim=1))
        d3 = self.decoder2(torch.cat([self._upsample(d4, features[-5]), features[-5]], dim=1))
        d2 = self.decoder3(torch.cat([self._upsample(d3, features[-6]), features[-6]], dim=1))
        d1 = self.decoder4(torch.cat([self._upsample(d2, features[-7]), features[-7]], dim=1))
        
        # 最终的预测输出
        out = self.final_conv(d1)
        
        return out
    
    def _upsample(self, tensor, target_tensor):
        """确保特征图的空间维度匹配."""
        return F.interpolate(tensor, size=target_tensor.shape[2:], mode='bilinear', align_corners=False)



### 评价指标

def dice_coeff(y_true, y_pred, a=1., b=1.):
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)
    intersection = torch.sum(y_true_flat * y_pred_flat)
    return (2. * intersection + a) / (torch.sum(y_true_flat) + torch.sum(y_pred_flat) + b)

def dice_loss(y_true, y_pred, a=1., b=1.):
    return 1.0 - dice_coeff(y_true, y_pred, a=a, b=b)


def compute_iou(pred, target, cls):
    pred_cls = (pred == cls)
    target_cls = (target == cls)
    
    intersection = np.logical_and(pred_cls, target_cls).sum()
    union = np.logical_or(pred_cls, target_cls).sum()
    
    if union == 0:
        return float('nan')  # 如果类别在预测和实际中都不存在，忽略此类别
    else:
        return intersection / union

def compute_miou_binary(pred, target):
    # 计算背景类别（0）的IOU
    iou_background = compute_iou(pred, target, 0)
    # 计算前景类别（1）的IOU
    iou_foreground = compute_iou(pred, target, 1)
    
    # 计算mIOU，忽略nan值
    miou = np.nanmean([iou_background, iou_foreground])
    return miou

def calculate_binary_segmentation_accuracy(preds, labels):
    """
    计算前景背景分割的像素级准确率
    :param preds: 模型的预测值，形状为 [batch_size, height, width] 或 [batch_size, 1, height, width]
    :param labels: 真实标签，形状为 [batch_size, 1, height, width]
    :return: 每个样本的准确率和平均准确率
    """
    if preds.dim() == 4:
        preds = preds.squeeze(1)  # 去掉频道维度
    
    if labels.dim() == 4:
        labels = labels.squeeze(1)  # 去掉频道维度
    
    assert preds.shape == labels.shape, "预测值和标签的形状必须一致"

    preds = (preds > 0.5).float()  # 阈值 0.5，用于二分类
    
    correct = (preds == labels).float().sum(dim=[1, 2])  # 每个样本的正确预测像素数
    total_pixels_per_sample = labels.size(1) * labels.size(2)
    
    accuracy_per_sample = correct / total_pixels_per_sample  # 每个样本的准确率
    mean_accuracy = accuracy_per_sample.mean().item()  # 批次中的平均准确率
    
    return accuracy_per_sample, mean_accuracy





# Define the dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, signal_dir, filenames):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.signal_dir = signal_dir
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # 根据索引获取当前文件名
        filename = self.filenames[idx]
        
        # 加载图像、掩模和超像素文件
        image_path = os.path.join(self.images_dir, filename)
        mask_path = os.path.join(self.masks_dir, filename)
        signal_path = os.path.join(self.signal_dir, filename)

        image = np.load(image_path)
        mask = np.load(mask_path)
        signal = np.load(signal_path)

        # 转换为 PyTorch 张量
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)  # (channels, height, width)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # (1, height, width)
        signal = torch.tensor(signal.transpose(2, 0, 1), dtype=torch.float32)  # (channels, height, width)

        return image, mask, signal

# 设置数据文件夹路径
images_dir = "/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/tumor_1_ROI_train_data/image_npy"
masks_dir = "/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/tumor_1_ROI_train_data/mask_npy"
signal_dir = '/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/tumor_1_ROI_train_data/signal_random_npy'

# 获取文件夹中的文件名
def get_filenames_from_folder(folder_path):
    return [filename for filename in os.listdir(folder_path) if filename.endswith('.npy')]

# 获取所有文件名
all_filenames = get_filenames_from_folder(images_dir)

# 打乱文件顺序
random.shuffle(all_filenames)

# 按8:2比例划分训练集和验证集
split_index = int(len(all_filenames) * 0.8)
train_filenames = all_filenames[:split_index]
val_filenames = all_filenames[split_index:]

# 打印数据集划分情况
print(f"训练集数量: {len(train_filenames)}")
print(f"验证集数量: {len(val_filenames)}")

# 将文件路径写入 txt 文件
train_txt_path = "/home/gjs/ISF_nuclick/ROI_Segmentor/train_filename.txt"
val_txt_path = "/home/gjs/ISF_nuclick/ROI_Segmentor/val_filename.txt"

def write_filenames_to_txt(txt_path, filenames):
    with open(txt_path, 'w') as f:
        for filename in filenames:
            f.write(f"{filename}\n")

# 写入训练集和验证集文件名到 txt 文件
write_filenames_to_txt(train_txt_path, train_filenames)
write_filenames_to_txt(val_txt_path, val_filenames)

# 创建自定义数据集类的实例
train_dataset = CustomDataset(images_dir, masks_dir, signal_dir, train_filenames)
val_dataset = CustomDataset(images_dir, masks_dir, signal_dir, val_filenames)


train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True ,  num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=8 ,shuffle=False,  num_workers=8)



# 创建第二个模型实例
model = EfficientNetUNet()




# model.load_state_dict(torch.load('/home/gjs/ISF_nuclick/checkpoints/resunet_suppixel_best_model_test.pth'))



  

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=4e-4)



criterion = nn.BCEWithLogitsLoss()

# Training function
def train_model(model, train_loader, val_loader, optimizer,  epochs=50):
    best_dice = 0.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_dice_score = 0.0
        train_accuracy = 0.0  # 用于累积准确率
        
        
        for item,(images, masks, aux_inputs) in enumerate(train_loader):
            
            
            images, aux_inputs, masks = images.to(device), aux_inputs.to(device), masks.to(device)
            
            
            optimizer.zero_grad()

            # outputs, all_superpixel_features = model(images, aux_inputs, superpixels)
            outputs = model(images, aux_inputs)
            # first_seg = model(images, aux_inputs, superpixels)
            

            
            outputs = outputs.unsqueeze(1)
           
            
            loss = dice_loss(outputs, masks)
            # loss = calc_dc_loss_sp(masks, all_pixel_features, superpixels , fg_proto_features)
            
            # loss = dice_loss(first_seg, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
            train_dice_score += dice_coeff(outputs, masks).item() * images.size(0)
            
            # 计算准确率
            _, batch_accuracy = calculate_binary_segmentation_accuracy(outputs, masks)
            train_accuracy += batch_accuracy * images.size(0)  # 将每个批次的准确率加权累积
            
            
        train_loss /= len(train_loader.dataset)
        
        train_dice_score /= len(train_loader.dataset)
        
        train_accuracy /= len(train_loader.dataset)  # 计算平均准确率

        ###
        model.eval()
        val_loss = 0.0
        dice_score = 0.0
        val_accuracy = 0.0  # 用于累积准确率
        
        

        with torch.no_grad():
            iou_scores = []
            for images, aux_inputs, masks, superpixels, filenames in val_loader:
                images, aux_inputs, masks, superpixels= images.to(device), aux_inputs.to(device), masks.to(device), superpixels.to(device)
                # outputs, all_superpixel_features = model(images, aux_inputs, superpixels)

                outputs = model(images)
                
                # first_seg_val = model(images, aux_inputs, superpixels)
                
                # loss = dice_loss(outputs, masks) + (1-signal_protofg_similar.mean()) + signal_protobg_similar.mean()
                #loss = criterion(outputs, masks)
                
                outputs = outputs.unsqueeze(1)
                
                loss = dice_loss(outputs , masks)
                # loss = calc_dc_loss_sp(masks, all_pixel_features, superpixels , fg_proto_features)
                val_loss += loss.item() * images.size(0)
                
                dice_score += dice_coeff(outputs, masks).item() * images.size(0)
                
                # 计算准确率
                _, batch_accuracy = calculate_binary_segmentation_accuracy(outputs, masks)
                val_accuracy += batch_accuracy * images.size(0)  # 将每个批次的准确率加权累积
                
                outputs = outputs.squeeze(1).cpu().numpy()
                preds = (outputs >= 0.5).astype(int)
                
                masks = masks.cpu().numpy()
                for pred, mask in zip(preds, masks):
                    miou = compute_miou_binary(pred, mask)
                    if not np.isnan(miou):
                        iou_scores.append(miou)
            mean_iou = np.mean(iou_scores)
                

        val_loss /= len(val_loader.dataset)
        dice_score /= len(val_loader.dataset)
        val_accuracy /= len(val_loader.dataset)  # 计算平均准确率

        print(f'Epoch {epoch+1}/{epochs}, Train Loss (CombinedLoss): {train_loss:.4f}, Train Dice: {train_dice_score:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss : {val_loss:.4f}, Val Dice: {dice_score:.4f}, Val_Mean IOU: {mean_iou:.4f}, Val_Acc: {val_accuracy:.4f}')

        # scheduler.step()
        
        # Save the best model
        if dice_score > best_dice:
            best_dice = dice_score
            torch.save(model.state_dict(), '/home/gjs/ISF_nuclick/checkpoints/resunet_suppixel_best_model_test_200.pth')


train_model(model, train_loader, val_loader, optimizer,  epochs=200)

