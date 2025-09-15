from UNet import *
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


# device_id =  1 # Specify which GPU to use, e.g., GPU 0
# torch.cuda.set_device(device_id)

# 设置随机种子
random.seed(42)  # 你可以选择任何整数作为种子

#################  nuclick  ###################

def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, actv='relu', use_bias=False, use_regularizer=False, do_batch_norm=True):
    layers = []
    padding = (kernel_size - 1) // 2 * dilation
    if use_regularizer:
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=use_bias, padding_mode='zeros'))
    else:
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=use_bias))
    if do_batch_norm:
        layers.append(BatchNorm2d(out_channels))
    if actv != 'None':
        if actv == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif actv == 'selu':
            layers.append(nn.SELU(inplace=True))
    return nn.Sequential(*layers)

class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sizes, dilatation_rates, is_dense=True):
        super(MultiScaleConvBlock, self).__init__()
        self.is_dense = is_dense
        if is_dense:
            self.conv0 = conv_bn_relu(in_channels, 4*out_channels, kernel_size=1)
        else:
            self.conv0 = nn.Identity()
        
        self.conv1 = conv_bn_relu(4*out_channels if is_dense else in_channels, out_channels, kernel_size=sizes[0], dilation=dilatation_rates[0])
        self.conv2 = conv_bn_relu(4*out_channels if is_dense else in_channels, out_channels, kernel_size=sizes[1], dilation=dilatation_rates[1])
        self.conv3 = conv_bn_relu(4*out_channels if is_dense else in_channels, out_channels, kernel_size=sizes[2], dilation=dilatation_rates[2])
        self.conv4 = conv_bn_relu(4*out_channels if is_dense else in_channels, out_channels, kernel_size=sizes[3], dilation=dilatation_rates[3])
        
        if is_dense:
            self.conv_out = conv_bn_relu(4*out_channels, out_channels, kernel_size=3)
    
    def forward(self, x):
        if self.is_dense:
            x = self.conv0(x)
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        conv3_out = self.conv3(x)
        conv4_out = self.conv4(x)
        output_map = torch.cat([conv1_out, conv2_out, conv3_out, conv4_out], dim=1)
        if self.is_dense:
            output_map = self.conv_out(output_map)
            output_map = torch.cat([x, output_map], dim=1)
        return output_map

class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, actv='relu', use_bias=False, use_regularizer=False, dilation=1):
        super(ResidualConv, self).__init__()
        self.actv = actv
        self.conv1 = conv_bn_relu(in_channels, out_channels, kernel_size, dilation=dilation, actv='None', use_bias=use_bias, use_regularizer=use_regularizer, do_batch_norm=True)
        self.conv2 = conv_bn_relu(out_channels, out_channels, kernel_size, dilation=dilation, actv='None', use_bias=use_bias, use_regularizer=use_regularizer, do_batch_norm=True)

        
    def forward(self, x):
        out_1 = self.conv1(x)
        out_2 = self.conv2(out_1)
        out = out_1 + out_2
        if self.actv == 'relu':
            out = F.relu(out)
        elif self.actv == 'selu':
            out = F.selu(out)
        return out

class MultiScaleResUnet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MultiScaleResUnet, self).__init__()
        self.conv1_1 = conv_bn_relu(in_channels, 64, kernel_size=7)
        self.conv1_2 = conv_bn_relu(64, 32, kernel_size=5)
        self.conv1_3 = conv_bn_relu(32, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.res2_1 = ResidualConv(32, 64)
        self.res2_2 = ResidualConv(64, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.res3_1 = ResidualConv(64, 128)
        self.msconv3 = MultiScaleConvBlock(128, 32, sizes=[3, 3, 5, 5], dilatation_rates=[1, 3, 3, 6], is_dense=False)
        self.res3_2 = ResidualConv(128, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.res4_1 = ResidualConv(128, 256)
        self.res4_2 = ResidualConv(256, 256)
        self.res4_3 = ResidualConv(256, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.res5_1 = ResidualConv(256, 512)
        self.res5_2 = ResidualConv(512, 512)
        self.res5_3 = ResidualConv(512, 512)
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        
        self.res51_1 = ResidualConv(512, 1024)
        self.res51_2 = ResidualConv(1024, 1024)
        
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.res6_1 = ResidualConv(1024, 512)
        self.res6_2 = ResidualConv(512, 256)
        
        self.up7 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.res7_1 = ResidualConv(512, 256)
        self.msconv7 = MultiScaleConvBlock(256, 64, sizes=[3, 3, 5, 5], dilatation_rates=[1, 3, 2, 3], is_dense=False)
        self.res7_2 = ResidualConv(256, 256)
        
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.res8_1 = ResidualConv(256, 128)
        self.res8_2 = ResidualConv(128, 128)
        
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.res9_1 = ResidualConv(128, 64)
        self.msconv9 = MultiScaleConvBlock(64, 16, sizes=[3, 3, 5, 7], dilatation_rates=[1, 3, 3, 6], is_dense=False)
        self.res9_2 = ResidualConv(64, 64)
        
        self.up10 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv10_1 = conv_bn_relu(64, 64)
        self.conv10_2 = conv_bn_relu(64, 32)
        self.conv10_3 = conv_bn_relu(32, 32)
        
        self.conv11 = nn.Conv2d(32, num_classes, kernel_size=1)
    
    def forward(self, x, aux_input):
        
        # inputs = torch.cat([x, aux_input], dim=1)
        
        conv1 = self.conv1_1(x)
        conv1 = self.conv1_2(conv1)
        conv1 = self.conv1_3(conv1)
        pool1 = self.pool1(conv1)
        
        conv2 = self.res2_1(pool1)
        conv2 = self.res2_2(conv2)
        pool2 = self.pool2(conv2)
        
        conv3 = self.res3_1(pool2)
        conv3 = self.msconv3(conv3)
        conv3 = self.res3_2(conv3)
        pool3 = self.pool3(conv3)
        
        conv4 = self.res4_1(pool3)
        conv4 = self.res4_2(conv4)
        conv4 = self.res4_3(conv4)
        pool4 = self.pool4(conv4)
        
        conv5 = self.res5_1(pool4)
        conv5 = self.res5_2(conv5)
        conv5 = self.res5_3(conv5)
        pool5 = self.pool5(conv5)
        
        conv51 = self.res51_1(pool5)
        conv51 = self.res51_2(conv51)
        
        up6 = self.up6(conv51)
        up6 = torch.cat([up6, conv5], dim=1)
        conv6 = self.res6_1(up6)
        conv6 = self.res6_2(conv6)
        
        up7 = self.up7(conv6)
        up7 = torch.cat([up7, conv4], dim=1)
        conv7 = self.res7_1(up7)
        conv7 = self.msconv7(conv7)
        conv7 = self.res7_2(conv7)
        
        up8 = self.up8(conv7)
        up8 = torch.cat([up8, conv3], dim=1)
        conv8 = self.res8_1(up8)
        conv8 = self.res8_2(conv8)
        
        up9 = self.up9(conv8)
        up9 = torch.cat([up9, conv2], dim=1)
        conv9 = self.res9_1(up9)
        conv9 = self.msconv9(conv9)
        conv9 = self.res9_2(conv9)
        
        up10 = self.up10(conv9)
        up10 = torch.cat([up10, conv1], dim=1)
        conv10 = self.conv10_1(up10)
        conv10 = self.conv10_2(conv10)
        conv10 = self.conv10_3(conv10)
        
        conv11 = self.conv11(conv10)
        output = F.sigmoid(conv11)
        
        # min_val = torch.amin(output, dim=(2, 3), keepdim=True)[0]  # 找到每个通道的最小值
        # max_val = torch.amax(output, dim=(2, 3), keepdim=True)[0]  # 找到每个通道的最大值
        # normalized_output = (output - min_val) / (max_val - min_val + 1e-8)  # 防止除以0
        
        # 使用SLIC进行超像素分割
        # segments_slic = slic(x, n_segments = 100, sigma = 5)
        
        return output
    
    
class MultiClassDiceLoss(nn.Module):
    def __init__(self, weight=None, smooth=1e-6):
        super(MultiClassDiceLoss, self).__init__()
        self.weight = weight
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs已是 softmax 概率分布，targets 已是 one-hot 编码
        intersection = torch.sum(inputs * targets, dim=(2, 3))
        union = torch.sum(inputs, dim=(2, 3)) + torch.sum(targets, dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # 对各类别的Dice系数进行加权求和
        if self.weight is not None:
            dice_loss = (1 - dice) * self.weight
        else:
            dice_loss = 1 - dice
        return dice_loss.mean()

class MultiClassLoss(nn.Module):
    def __init__(self, weight=None, dice_weight=0.5):
        super(MultiClassLoss, self).__init__()
        self.dice_loss = MultiClassDiceLoss(weight=weight)
        self.dice_weight = dice_weight
        self.ce_weight = weight

    def forward(self, inputs, targets):
        # 计算Dice损失
        dice_loss = self.dice_loss(inputs, targets)

        # 将inputs和targets转换为交叉熵损失可接受的格式
        inputs_ce = inputs.log()  # 使用 log-softmax 使之适应交叉熵格式
        targets_ce = targets.argmax(dim=1)  # 将 one-hot targets 转换回类别索引

        # 计算交叉熵损失
        ce_loss = F.nll_loss(inputs_ce, targets_ce, weight=self.ce_weight)

        # 组合交叉熵和Dice损失
        total_loss = (1 - self.dice_weight) * ce_loss + self.dice_weight * dice_loss
        return total_loss
'''

###未加权

def multi_class_dice_loss(pred, target, num_classes=5, epsilon=1e-6):
    batch_size = pred.size(0)
    dice_loss = 0.0
    total_valid_class_count = 0  # 记录batch中所有有效类别的数量

    for b in range(batch_size):
        # 对于 batch 中的每张图片分别计算
        image_loss = 0.0
        valid_class_count = 0  # 当前图片中的有效类别数量

        for c in range(num_classes):
            pred_c = pred[b, c, :, :]  # 当前图片的第c个类别预测图
            target_c = target[b, c, :, :]  # 当前图片的第c个类别真实标签图
            
            # 仅对包含该类别的图片进行计算
            if target_c.sum() > 0:
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()
                dice_c = (2 * intersection + epsilon) / (union + epsilon)
                image_loss += 1 - dice_c
                valid_class_count += 1

        # 如果图片中有有效类别，则计算该图片的平均损失
        if valid_class_count > 0:
            dice_loss += image_loss / valid_class_count
            total_valid_class_count += 1

    # 返回 batch 内所有图片的平均 Dice loss
    if total_valid_class_count > 0:
        return dice_loss / total_valid_class_count
    else:
        return torch.zeros(1, device=pred.device, requires_grad=True)
'''

def multi_class_dice_loss(pred, target, num_classes=5, class_weights=None, epsilon=1e-6):
    """
    计算多类别加权 Dice 损失。

    参数:
        pred (Tensor): 预测张量，形状为 [batch_size, num_classes, H, W]。
        target (Tensor): 真实标签张量，形状为 [batch_size, num_classes, H, W]。
        num_classes (int): 类别数，默认值为 5。
        class_weights (Tensor or list): 类别权重，形状为 [num_classes]。
        epsilon (float): 防止除零的小值，默认值为 1e-6。

    返回:
        Tensor: 平均加权 Dice 损失。
    """
    batch_size = pred.size(0)
    dice_loss = 0.0
    total_valid_class_count = 0  # 记录batch中所有有效类别的数量

    # 如果未提供类别权重，则使用均等权重
    if class_weights is None:
        class_weights = torch.ones(num_classes, device=pred.device)

    for b in range(batch_size):
        # 对于 batch 中的每张图片分别计算
        image_loss = 0.0
        valid_class_count = 0  # 当前图片中的有效类别数量

        for c in range(num_classes):
            pred_c = pred[b, c, :, :]  # 当前图片的第c个类别预测图
            target_c = target[b, c, :, :]  # 当前图片的第c个类别真实标签图
            
            # 仅对包含该类别的图片进行计算
            if target_c.sum() > 0:
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()
                dice_c = (2 * intersection + epsilon) / (union + epsilon)
                weighted_dice_loss = (1 - dice_c) * class_weights[c]  # 加权
                image_loss += weighted_dice_loss
                valid_class_count += 1

        # 如果图片中有有效类别，则计算该图片的平均损失
        if valid_class_count > 0:
            dice_loss += image_loss / valid_class_count
            total_valid_class_count += 1

    # 返回 batch 内所有图片的平均 Dice loss
    if total_valid_class_count > 0:
        return dice_loss / total_valid_class_count
    else:
        return torch.zeros(1, device=pred.device, requires_grad=True)


'''
class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(MultiClassDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 输入为预测概率分布，确保经过 softmax 处理
        # inputs = F.softmax(inputs, dim=1)
        
        # 计算交集和并集
        intersection = torch.sum(inputs * targets, dim=(2, 3))
        union = torch.sum(inputs, dim=(2, 3)) + torch.sum(targets, dim=(2, 3))
        
        # 计算每个类别的Dice系数
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 计算平均Dice Loss
        dice_loss = 1 - dice.mean()
        
        return dice_loss
 '''   
    
    
### 评价指标
# 定义计算accuracy, dice, IoU的函数
def calculate_acc(pred, target, num_classes, all_accs):
    
    for i in range(num_classes):
        # 计算每个类别的准确率
        pred_i = pred[i, :, :]  # 获取当前类别的预测概率图
        target_i = target[i, :, :]  # 获取当前类别的真实标签二值图
        if target_i.sum() > 0:
            intersection = (pred_i * target_i).sum()
            total = target_i.sum() + 1e-6
            acc = intersection / total
            all_accs[i].append(acc.item())
    return all_accs

def calculate_dice(pred, target, num_classes, all_dices):
    for i in range(num_classes):
        # 计算每个类别的Dice系数
        pred_i = pred[i, :, :]  # 获取当前类别的预测概率图
        target_i = target[i, :, :]  # 获取当前类别的真实标签二值图
        if target_i.sum() > 0:
            intersection = (pred_i * target_i).sum()
            union = pred_i.sum() + target_i.sum() + 1e-6
            dice = 2 * intersection / union
            all_dices[i].append(dice.item())
    return all_dices

def calculate_iou(pred, target, num_classes, all_ious):
    for i in range(num_classes):
        # 计算每个类别的IoU
        pred_i = pred[i, :, :]  # 获取当前类别的预测概率图
        target_i = target[i, :, :]  # 获取当前类别的真实标签二值图
        if target_i.sum() > 0:
            intersection = (pred_i * target_i).sum()
            union = (pred_i + target_i).sum() - intersection + 1e-6
            iou = intersection / union
            all_ious[i].append(iou.item())
    return all_ious


### 计算outputs中预测值为1且masks中值也为1的像素点所占的比例，计算outputs中预测值为1但masks中值为0的像素点所占的比例
def calculate_metrics(outputs, masks):
    
    # 最大最小归一化
    outputs_min = outputs.min()
    outputs_max = outputs.max()
    
    # 处理极端情况，防止除以0
    if outputs_max - outputs_min > 0:
        outputs = (outputs - outputs_min) / (outputs_max - outputs_min)
    else:
        outputs = torch.zeros_like(outputs)  # 如果没有变化，直接设为0

    # 确保outputs和masks的值为0或1
    outputs = (outputs > 0.6).float()  # 阈值设定为0.5，预测值大于0.5的被认为是1
    masks = masks.float()
    
    # 预测为1的像素点
    outputs_1 = outputs == 1
    # 真实值为1的像素点
    masks_1 = masks == 1
    # 真实值为0的像素点
    masks_0 = masks == 0
    
    # 计算预测为1且真实值为1的像素数
    true_positive = (outputs_1 & masks_1).sum().item()
    
    # 计算预测为1且真实值为0的像素数
    false_positive = (outputs_1 & masks_0).sum().item()
    
    # 计算总的预测为1的像素数
    total_pred_1 = outputs_1.sum().item()
    
    # 防止除以0的情况
    if total_pred_1 == 0:
        return 0, 0
    
    # 计算比例
    true_positive_ratio = true_positive / total_pred_1
    false_positive_ratio = false_positive / total_pred_1
    
    return true_positive_ratio, false_positive_ratio





def one_hot_encode(mask, num_classes=5):
    # 获取图像的高度和宽度
    _, height, width = mask.shape
    # 创建一个新的全零的张量，shape为(num_classes, height, width)
    one_hot = torch.zeros(num_classes, height, width)
    
    # 遍历mask中的每个像素，设置对应类别通道的值为1
    for c in range(1, num_classes + 1):  # 类别从1到num_classes
        one_hot[c - 1] = (mask == c).float()  # 将mask中为c的像素设置为1
    return one_hot


# Define the dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, signal_dir, masks_dir,  filenames, cls_number):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.signal_dir = signal_dir
        self.filenames = filenames
        self.num_classes = cls_number

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
        # image = torch.tensor(image, dtype=torch.float32)  # (channels, height, width)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # (1, height, width)
        # mask = torch.tensor(mask, dtype=torch.float32)  # (1, height, width)
        # signal = torch.tensor(signal, dtype=torch.float32)  # (channels, height, width)
        signal = torch.tensor(signal.transpose(2, 0, 1), dtype=torch.float32)  # (channels, height, width)
        # 将mask进行One-Hot编码
        mask_one_hot = one_hot_encode(mask, self.num_classes)  # (num_classes, height, width)
        

        return image, signal, mask_one_hot
    
# 获取文件夹中的文件名
def get_filenames_from_folder(folder_path):
    return [filename for filename in os.listdir(folder_path) if filename.endswith('.npy')]   


# 设置训练集和验证集的文件夹路径
cls_number = 5

i = 1
cls = 'Contrast_learning'
train_images_dir = f"/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/150/step256_patch512_nofilling/fold_{i}/train/{cls}/image_npy"
train_masks_dir = f"/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/150/step256_patch512_nofilling/fold_{i}/train/{cls}/mask_npy"
train_signal_dir = f'/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/150/step256_patch512_nofilling/fold_{i}/train/{cls}/signal_max_point_npy'


val_images_dir = f"/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/150/step256_patch512_nofilling/fold_{i}/val/{cls}/image_npy"
val_masks_dir = f"/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/150/step256_patch512_nofilling/fold_{i}/val/{cls}/mask_npy"
val_signal_dir = f'/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/150/step256_patch512_nofilling/fold_{i}/val/{cls}/signal_max_point_npy'


# 获取训练集和验证集的文件名
train_filenames = get_filenames_from_folder(train_images_dir)
val_filenames = get_filenames_from_folder(val_images_dir)

# 创建自定义数据集类的实例
train_dataset = CustomDataset(train_images_dir, train_signal_dir, train_masks_dir, train_filenames, cls_number)
val_dataset = CustomDataset(val_images_dir, val_signal_dir, val_masks_dir, val_filenames, cls_number)



train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=4)



# 创建模型实例
# model = get_efficientunet_b0(out_channels=1, concat_input=True, pretrained=False).cuda()

# model = MultiScaleResUnet(in_channels=3, num_classes=5)

# model = ResNetUNet_proto()
model = UNet(n_channels=3, n_classes=5)

# model.load_state_dict(torch.load('/home/gjs/ISF_nuclick/checkpoints/ROI_segmentor/nuclick_tumor_1_best.pth'))



  

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=4e-4)


# class_weights = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.4], device=pred.device)
loss_fn = MultiClassLoss()

# Training function
def train_model(model, train_loader, val_loader, loss_fn, optimizer,  epochs=50):
    best_dice = 0.0
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)
    cls_number = 5

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_all_accs = [[] for _ in range(cls_number)]  # 每个类别的准确率
        train_all_dices = [[] for _ in range(cls_number)]  # 每个类别的Dice系数
        train_all_ious = [[] for _ in range(cls_number)]  # 每个类别的IoU
        train_true_positive_ratio, train_false_positive_ratio = 0.0 , 0.0
        
        
        for item,(images, aux_inputs, masks) in enumerate(train_loader):
            
            
            images, aux_inputs, masks_one_hot = images.to(device), aux_inputs.to(device), masks.to(device)
            # roi_input , roi_aux_input , roi_mask = ROI_crop(images , aux_inputs, masks)
            
            # input = torch.cat([roi_input , roi_aux_input], dim=1)
            
            optimizer.zero_grad()

            # outputs, all_superpixel_features = model(images, aux_inputs, superpixels)
            outputs = model(images)
            
            # outputs = model(roi_input , roi_aux_input , images, aux_inputs, masks)
           
            class_weights = torch.tensor([0.3, 0.3, 0.6, 0.7, 0.5], device=outputs.device)
            # loss = multi_class_dice_loss(outputs, masks_one_hot)
            loss = multi_class_dice_loss(outputs, masks_one_hot, num_classes=5, class_weights=class_weights)
            # loss = calc_dc_loss_sp(masks, all_pixel_features, superpixels , fg_proto_features)
            
            # loss = dice_loss(first_seg, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
            # 找到每个像素位置的最大值对应的通道索引
            max_indices = torch.argmax(outputs, dim=1, keepdim=True)  # 结果 shape 为 [b, 1, 512, 512]

            # 将最大值所在的通道置为 1，其余置为 0
            out_put_one_hot = torch.zeros_like(outputs)
            out_put_one_hot.scatter_(1, max_indices, 1)
            
            # 遍历batch中的每张图片，计算各个类别的指标
            batch_size = images.size(0)
            
            for b in range(batch_size):
                # 获取当前图片的预测结果和真实标签
                pred = out_put_one_hot[b].squeeze(0)  # 预测结果
                target = masks_one_hot[b].squeeze(0)  # 真实标签

                # 计算每个类别的准确率、Dice系数和IoU
                train_all_accs = calculate_acc(pred, target, cls_number, train_all_accs)
                train_all_dices = calculate_dice(pred, target, cls_number, train_all_dices)
                train_all_ious = calculate_iou(pred, target, cls_number, train_all_ious)
            
            
        # 计算每个类别的平均值
        train_avg_accs = [np.mean(accs) for accs in train_all_accs]
        train_avg_dices = [np.mean(dices) for dices in train_all_dices]
        train_avg_ious = [np.mean(ious) for ious in train_all_ious]
        
        # 计算所有类别的总平均值
        train_total_avg_acc = np.mean(train_avg_accs)
        train_total_avg_dice = np.mean(train_avg_dices)
        train_total_avg_iou = np.mean(train_avg_ious)  
        train_loss /= len(train_loader.dataset) 
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, train_total_avg_dice: {train_total_avg_dice:.4f}, train_total_avg_acc: {train_total_avg_acc:.4f}, train_total_avg_iou: {train_total_avg_iou:.4f}') 
        print("     train Average Accuracies:", train_avg_accs)
        print("     train Average Dices:", train_avg_dices)
        print("     train Average IoUs:", train_avg_ious)

        ###
        model.eval()
        val_loss = 0.0
        # 用来存储各个类别的指标
        val_all_accs = [[] for _ in range(cls_number)]  # 每个类别的准确率
        val_all_dices = [[] for _ in range(cls_number)]  # 每个类别的Dice系数
        val_all_ious = [[] for _ in range(cls_number)]  # 每个类别的IoU
        

        with torch.no_grad():
            iou_scores = []
            for images, aux_inputs, masks in val_loader:
                images, aux_inputs, masks_one_hot = images.to(device), aux_inputs.to(device), masks.to(device)
                # roi_input , roi_aux_input , roi_mask = ROI_crop(images , aux_inputs, masks)
                # outputs, all_superpixel_features = model(images, aux_inputs, superpixels)
                # input = torch.cat([roi_input , roi_aux_input], dim=1)

                outputs = model(images)
                # outputs = model(roi_input , roi_aux_input , images, aux_inputs, masks)
                

                
                # loss = multi_class_dice_loss(outputs, masks_one_hot)
                class_weights = torch.tensor([0.3, 0.4, 0.5, 0.7, 0.4], device=outputs.device)
                # loss = multi_class_dice_loss(outputs, masks_one_hot)
                loss = multi_class_dice_loss(outputs, masks_one_hot, num_classes=5, class_weights=class_weights)
                val_loss += loss.item() * images.size(0)
                
                # 找到每个像素位置的最大值对应的通道索引
                max_indices = torch.argmax(outputs, dim=1, keepdim=True)  # 结果 shape 为 [b, 1, 512, 512]

                # 将最大值所在的通道置为 1，其余置为 0
                out_put_one_hot = torch.zeros_like(outputs)
                out_put_one_hot.scatter_(1, max_indices, 1)

                # 遍历batch中的每张图片，计算各个类别的指标
                batch_size = images.size(0)
                
                for b in range(batch_size):
                    # 获取当前图片的预测结果和真实标签
                    pred = out_put_one_hot[b].squeeze(0)  # 预测结果
                    target = masks_one_hot[b].squeeze(0)  # 真实标签

                    # 计算每个类别的准确率、Dice系数和IoU
                    val_all_accs = calculate_acc(pred, target, cls_number, val_all_accs)
                    val_all_dices = calculate_dice(pred, target, cls_number, val_all_dices)
                    val_all_ious = calculate_iou(pred, target, cls_number, val_all_ious)

                        
            # 计算每个类别的平均值
            val_avg_accs = [np.mean(accs) for accs in val_all_accs]
            val_avg_dices = [np.mean(dices) for dices in val_all_dices]
            val_avg_ious = [np.mean(ious) for ious in val_all_ious]
            
            # 计算所有类别的总平均值
            val_total_avg_acc = np.mean(val_avg_accs)
            val_total_avg_dice = np.mean(val_avg_dices)
            val_total_avg_iou = np.mean(val_avg_ious) 
            val_loss /= len(val_loader.dataset)  
            
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss : {val_loss:.4f}, val_total_avg_dice: {val_total_avg_dice:.4f}, val_total_avg_acc: {val_total_avg_acc:.4f}, val total_avg_iou: {val_total_avg_iou:.4f}') 
            print("     Val Average Accuracies:", val_avg_accs)
            print("     Val Average Dices:", val_avg_dices)
            print("     Val Average IoUs:", val_avg_ious)

        
        # # Save the best model
        # if val_total_avg_dice > best_dice:
        #     best_dice = val_total_avg_dice
        #     torch.save(model.state_dict(), f'/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/150/step256_patch512_nofilling/checkpoints/fold_{i}/fold_{i}_unet_256_nofilling_best.pth')


train_model(model, train_loader, val_loader, loss_fn, optimizer,  epochs=200)
print(f"fold_{i}")
