from efficientunet import *
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

from scipy.ndimage.morphology import distance_transform_edt
import scipy.ndimage as ndi
from skimage.morphology import skeletonize_3d


from skimage.measure import label as label_1
from skimage.measure import regionprops


def get_dist_maps_batch(binary_tensor_batch, norm_delimeter=260.0):
    """
    批量计算每个像素到二值张量中指定点的归一化平方距离。

    Args:
        binary_tensor_batch: 形状为 (batch_size, 2, H, W) 的 NumPy 数组。
        norm_delimeter: 归一化距离的除数。

    Returns:
        形状为 (batch_size, 2, H, W) 的距离图。
    """
    device = binary_tensor_batch.device  # 保存设备信息
    binary_tensor_batch = binary_tensor_batch.cpu().numpy()

    batch_size, num_layers, height, width = binary_tensor_batch.shape
    dist_maps_batch = np.full((batch_size, num_layers, height, width), 1e6, dtype=np.float32)


    for batch_idx in range(batch_size):
        for layer in range(num_layers):
            binary_layer = binary_tensor_batch[batch_idx, layer]
            if np.any(binary_layer == 255):

                # 将前景（255）转换为布尔值 (True 表示前景)
                foreground = (binary_layer == 255)

                # 计算每个像素到前景的距离变换
                dist_map = distance_transform_edt(~foreground)  # 计算背景到前景的距离

                # 归一化距离
                dist_map = (dist_map / norm_delimeter)** 2  # 计算平方距离

                dist_maps_batch[batch_idx, layer] = dist_map
            else:
                continue

    dist_maps_batch = torch.from_numpy(dist_maps_batch).to(device)
    return dist_maps_batch


#########################################################################################################
#计算误差区域的形态学骨架

def generateGuidingSignal(binaryMask):
    # binaryMask = binaryMask.squeeze(0)  # Remove the batch dimension if it's (1, H, W)
    binaryMask = binaryMask.to(torch.uint8)
    
    if binaryMask.sum() > 1:
        # Compute distance transform (move to CPU for NumPy operations)
        distance_map = distance_transform_edt(binaryMask.cpu().numpy())
        # distance_map = torch.tensor(distance_map, dtype=torch.float32, device=binaryMask.device)
        
        # Calculate mean and std (ensure they are on CPU before NumPy operations)
        tempMean = distance_map.mean()
        tempStd = distance_map.std()
        
        # Random threshold based on mean and std
        tempThresh = np.random.uniform(tempMean - tempStd, tempMean + tempStd)
        # tempThresh = torch.tensor(tempThresh, device=binaryMask.device)
        
        if tempThresh < 0:
            tempThresh = np.random.uniform(tempMean / 2, tempMean + tempStd / 2)
            # tempThresh = torch.tensor(tempThresh, device=binaryMask.device)
        
        # Apply threshold to get new mask
        newMask = distance_map > tempThresh
        if newMask.sum() == 0:
            newMask = distance_map > (tempThresh / 2)
        
        if newMask.sum() == 0:
            newMask = binaryMask

        # Skeletonize (use skimage and convert back to tensor)
        skel = skeletonize_3d(newMask)
        skel = torch.tensor(skel, dtype=torch.float32, device=binaryMask.device)
    else:
        skel = torch.zeros_like(binaryMask, dtype=torch.float32, device=binaryMask.device)

    return skel

def processMasks_old(pred_mask_all, GT_mask_all):
    """
    批量处理GT_mask和pred_mask，计算每个样本的前景和背景骨架信号。
    参数:
        pred_masks: 预测的mask，形状为 [batch_size, 1, H, W]。
        GT_masks: 真值mask，形状为 [batch_size, 1, H, W]。
    返回:
        输出张量，形状为 [batch_size, 2, H, W]。
        每个样本的第0通道为前景区域骨架信号，第1通道为背景区域骨架信号。
    """
    pred_mask_all = (pred_mask_all > 0.5).float()

    batch_size, _, H, W = pred_mask_all.shape

    # 初始化输出张量
    output = torch.zeros(batch_size, 2, H, W, device=pred_mask_all.device, dtype=torch.float32)

    for i in range(batch_size):
        # 取出当前样本的预测和真值mask
        pred_mask = pred_mask_all[i].squeeze(0)  # [H, W]
        GT_mask = GT_mask_all[i].squeeze(0)      # [H, W]

        # 计算前景区域 (GT_mask为1且pred_mask为0)
        fg = (GT_mask == 1) & (pred_mask == 0)
        fg = fg.to(torch.float32)  # [H, W]

        # 计算背景区域 (GT_mask为0且pred_mask为1)
        bg = (GT_mask == 0) & (pred_mask == 1)
        bg = bg.to(torch.float32)  # [H, W]
        
        # 计算前景区域的骨架信号
        fg_skeleton = generateGuidingSignal(fg) if fg.sum() > 0 else torch.zeros_like(pred_mask, dtype=torch.float32, device=pred_mask.device)  # 如果fg为全0，创建一个全零张量
        
        # 计算背景区域的骨架信号
        bg_skeleton = generateGuidingSignal(bg) if bg.sum() > 0 else torch.zeros_like(pred_mask, dtype=torch.float32, device=pred_mask.device)  # 如果bg为全0，创建一个全零张量



        # 合并前景和背景骨架信号到输出
        output[i, 0] = fg_skeleton  # 前景骨架信号
        output[i, 1] = bg_skeleton  # 背景骨架信号

    return output




def processMasks(pred_mask_all, GT_mask_all):
    """
    批量处理GT_mask和pred_mask，计算每个样本的前景和背景骨架信号。
    参数:
        pred_masks: 预测的mask，形状为 [batch_size, 1, H, W]。
        GT_masks: 真值mask，形状为 [batch_size, 1, H, W]。
    返回:
        输出张量，形状为 [batch_size, 2, H, W]。
        每个样本的第0通道为前景区域骨架信号，第1通道为背景区域骨架信号。
    """
    pred_mask_all = (pred_mask_all > 0.5).float()
    batch_size, _, H, W = pred_mask_all.shape
    output = torch.zeros(batch_size, 2, H, W, device=pred_mask_all.device, dtype=torch.float32)

    # 提取整个批次的GT_mask和pred_mask
    GT_mask_all = GT_mask_all.squeeze(1)  # [batch_size, H, W]
    pred_mask_all = pred_mask_all.squeeze(1)  # [batch_size, H, W]

    # 计算前景区域和背景区域
    fg_all = (GT_mask_all == 1) & (pred_mask_all == 0)  # 前景区域: GT为1且预测为0
    bg_all = (GT_mask_all == 0) & (pred_mask_all == 1)  # 背景区域: GT为0且预测为1

    # 将前景和背景区域的张量转为浮点型
    fg_all = fg_all.float()
    bg_all = bg_all.float()

    # 批量处理每个样本
    for i in range(batch_size):
        fg = fg_all[i]  # [H, W]
        bg = bg_all[i]  # [H, W]

        # 找出前景和背景的最大连通域
        fg_largest = get_largest_connected_component(fg)
        bg_largest = get_largest_connected_component(bg)

        # 比较前景和背景的最大连通域面积
        fg_area = fg_largest.sum().item()
        bg_area = bg_largest.sum().item()

        # 根据面积选择较大的连通域
        if fg_area >= bg_area:
            largest_connected = fg_largest
            output[i, 0] = generateGuidingSignal(largest_connected) if largest_connected.sum() > 0 else torch.zeros_like(fg, dtype=torch.float32, device=fg.device)  # 前景骨架信号
        else:
            largest_connected = bg_largest
            output[i, 1] = generateGuidingSignal(largest_connected) if largest_connected.sum() > 0 else torch.zeros_like(bg, dtype=torch.float32, device=bg.device)  # 背景骨架信号

    return output


def get_largest_connected_component(mask):
    """
    获取mask中最大连通区域
    参数:
        mask: 二值化mask，形状为 [H, W]
    返回:
        最大连通域的二值mask，形状为 [H, W]
    """
    if mask.sum() > 0:
        labeled_mask = label_1(mask.cpu().numpy(), connectivity=1)
        regions = regionprops(labeled_mask)
        if regions:
            largest_region = max(regions, key=lambda r: r.area)
            largest_component = (labeled_mask == largest_region.label)
            return torch.from_numpy(largest_component).to(mask.device, dtype=torch.float32)
    return torch.zeros_like(mask, device=mask.device)


##########################################################################################################





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

# Define the dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, signal_dir,  filenames):
    # def __init__(self, images_dir, masks_dir,   filenames):
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
        signal = torch.tensor(signal, dtype=torch.float32)  # (channels, height, width)

        return image, mask, signal
        # return image, mask
    
# 获取文件夹中的文件名
def get_filenames_from_folder(folder_path):
    return [filename for filename in os.listdir(folder_path) if filename.endswith('.npy')]    
    


# 设置训练集和验证集的文件夹路径
# train_images_dir = "/data_nas2/gjs/ISF_pixel_level_data/Gastric/train/ROI_data/all_connect/image_npy"
# train_masks_dir = "/data_nas2/gjs/ISF_pixel_level_data/Gastric/train/ROI_data/all_connect/mask_npy"
# train_signal_dir = '/data_nas2/gjs/ISF_pixel_level_data/Gastric/train/ROI_data/all_connect/signal_line_npy'

# val_images_dir = "/data_nas2/gjs/ISF_pixel_level_data/Gastric/val/ROI_data/all_connect/image_npy"
# val_masks_dir = "/data_nas2/gjs/ISF_pixel_level_data/Gastric/val/ROI_data/all_connect/mask_npy"
# val_signal_dir = '/data_nas2/gjs/ISF_pixel_level_data/Gastric/val/ROI_data/all_connect/signal_line_npy'
i = 5

path = "/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/125WSI"

train_images_dir = f"{path}/fold_{i}/train/ROI_data/all_class/image_npy"
train_masks_dir = f"{path}/fold_{i}/train/ROI_data/all_class/mask_npy"
train_signal_dir = f'{path}/fold_{i}/train/ROI_data/all_class/signal_maxconnect_line_npy'

val_images_dir = f"{path}/fold_{i}/val/ROI_data/all_class/image_npy"
val_masks_dir = f"{path}/fold_{i}/val/ROI_data/all_class/mask_npy"
val_signal_dir = f'{path}/fold_{i}/val/ROI_data/all_class/signal_maxconnect_line_npy'


# 获取训练集和验证集的文件名
train_filenames = get_filenames_from_folder(train_images_dir)
val_filenames = get_filenames_from_folder(val_images_dir)

# 创建自定义数据集类的实例
train_dataset = CustomDataset(train_images_dir, train_masks_dir, train_signal_dir, train_filenames)
val_dataset = CustomDataset(val_images_dir, val_masks_dir, val_signal_dir, val_filenames)

# train_dataset = CustomDataset(train_images_dir, train_masks_dir,  train_filenames)
# val_dataset = CustomDataset(val_images_dir, val_masks_dir,  val_filenames)




train_loader = DataLoader(train_dataset, batch_size=42, shuffle=True ,  num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=42 ,shuffle=False,  num_workers=8)



# 创建模型实例
model = get_efficientunet_b0(out_channels=1, concat_input=True, pretrained=False).cuda()

# model = MultiScaleResUnet(in_channels=5, num_classes=1)

# model = UNet(n_channels=3, n_classes=1)

# model.load_state_dict(torch.load('/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/125WSI/fold_2/ROI_ckpt/BCSS_effi-Unet_roi_best_1+1_threod_allmask.pth'))



  

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=4e-4)



loss_fn = nn.BCELoss()

# Training function
def train_model(model, train_loader, val_loader, loss_fn, optimizer,  epochs=50):
    best_dice = 0.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_dice_score = 0.0
        train_accuracy = 0.0  # 用于累积准确率
        train_true_positive_ratio, train_false_positive_ratio = 0.0 , 0.0
        
        train_loader_tqdm = tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs} [train]', unit='batch')
        
        for images, masks, aux_inputs in train_loader_tqdm:
        # for item,(images, masks) in enumerate(train_loader):
            
            
            images, aux_inputs, masks= images.to(device), aux_inputs.to(device), masks.to(device)
            # images,  masks= images.to(device), masks.to(device)
            
            # dist_map = get_dist_maps_batch(aux_inputs)
            # 生成与 masks 形状相同的全零张量
            pred_mask = torch.zeros_like(masks)
            # aux_inputs = processMasks(pred_mask, masks)
            
            input = torch.cat((images, pred_mask, aux_inputs), dim=1)
            
            optimizer.zero_grad()
            # outputs, all_superpixel_features = model(images, aux_inputs, superpixels)
            pred_mask_1 = model(input)
            
            signal = processMasks(pred_mask_1, masks)
            union_signal = torch.bitwise_or(signal.to(torch.uint8), aux_inputs.to(torch.uint8))
            
            pre_mask_1_threod = (pred_mask_1 >= 0.5).int()
            # # dist_map = get_dist_maps_batch(union_signal)
            input = torch.cat((images, pre_mask_1_threod, union_signal ), dim=1)
            pred_mask_2 = model(input)
 
            loss = dice_loss(pred_mask_1, masks) + dice_loss(pred_mask_2, masks)
            # loss = calc_dc_loss_sp(masks, all_pixel_features, superpixels , fg_proto_features)
            
            # loss = dice_loss(first_seg, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
            outputs = pred_mask_2
            
            outputs = (outputs >= 0.5).int()
            
            train_dice_score += dice_coeff(outputs, masks).item() * images.size(0)
            
            # 计算准确率
            _, batch_accuracy = calculate_binary_segmentation_accuracy(outputs, masks)
            train_accuracy += batch_accuracy * images.size(0)  # 将每个批次的准确率加权累积
            
            # true_positive_ratio_1 = calculate_spp(outputs, masks , suppixels)* images.size(0)
            # train_true_positive_ratio += true_positive_ratio_1 
            
            
            # true_positive_ratio_1, false_positive_ratio_1 = calculate_metrics(outputs, masks)
            # train_true_positive_ratio += true_positive_ratio_1* images.size(0)
            # train_false_positive_ratio += false_positive_ratio_1* images.size(0)
            
            
        train_loss /= len(train_loader.dataset)
        
        train_dice_score /= len(train_loader.dataset)
        
        train_accuracy /= len(train_loader.dataset)  # 计算平均准确率
        
        train_true_positive_ratio /= len(train_loader.dataset)
        
        # train_false_positive_ratio /= len(train_loader.dataset)

        ###
        model.eval()
        val_loss = 0.0
        dice_score = 0.0
        val_accuracy = 0.0  # 用于累积准确率
        val_true_positive_ratio, val_false_positive_ratio = 0.0 , 0.0
        

        with torch.no_grad():
            iou_scores = []
            
            val_loader_tqdm = tqdm(val_loader, total=len(val_loader), desc=f'Epoch {epoch+1}/{epochs} [val]', unit='batch')
            for images, masks, aux_inputs in val_loader_tqdm:
            # for images, masks in val_loader:
                
                
                images, aux_inputs, masks = images.to(device), aux_inputs.to(device), masks.to(device)
                images, masks = images.to(device),  masks.to(device)
                
                # dist_map = get_dist_maps_batch(aux_inputs)
                # 生成与 masks 形状相同的全零张量
                pred_mask = torch.zeros_like(masks)
                # aux_inputs = processMasks(pred_mask, masks)
                input = torch.cat((images, pred_mask, aux_inputs), dim=1)
                
                # optimizer.zero_grad()
                # outputs, all_superpixel_features = model(images, aux_inputs, superpixels)
                pred_mask_1 = model(input)
                
                signal = processMasks(pred_mask_1, masks)
                union_signal = torch.bitwise_or(signal.to(torch.uint8), aux_inputs.to(torch.uint8))
                pre_mask_1_threod = (pred_mask_1 >= 0.5).int()
                # # dist_map = get_dist_maps_batch(union_signal)
                input = torch.cat((images, pre_mask_1_threod, union_signal ), dim=1)
                pred_mask_2 = model(input)
    
                loss = dice_loss(pred_mask_1, masks) + dice_loss(pred_mask_2, masks)
                # loss = calc_dc_loss_sp(masks, all_pixel_features, superpixels , fg_proto_features)
                val_loss += loss.item() * images.size(0)
                
                outputs = pred_mask_2
                outputs = (outputs >= 0.5).int()
                
                dice_score += dice_coeff(outputs, masks).item() * images.size(0)
                
                # 计算准确率
                _, batch_accuracy = calculate_binary_segmentation_accuracy(outputs, masks)
                val_accuracy += batch_accuracy * images.size(0)  # 将每个批次的准确率加权累积
                
                
                # true_positive_ratio_1 = calculate_spp(outputs, masks , suppixels)
                # val_true_positive_ratio += true_positive_ratio_1 * images.size(0)
                
                # true_positive_ratio_1, false_positive_ratio_1 = calculate_metrics(outputs, masks)
                # val_true_positive_ratio += true_positive_ratio_1* images.size(0)
                # val_false_positive_ratio += false_positive_ratio_1* images.size(0)
                
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
        val_true_positive_ratio /= len(val_loader.dataset)
        
        # val_false_positive_ratio /= len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss (CombinedLoss): {train_loss:.4f}, Train Dice: {train_dice_score:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss : {val_loss:.4f}, Val Dice: {dice_score:.4f}, Val_Mean IOU: {mean_iou:.4f}, Val_Acc: {val_accuracy:.4f}')
        # print(f'Epoch {epoch+1}/{epochs}, Train_true_positive_ratio: {train_true_positive_ratio:.4f}, Train_false_positive_ratio:{train_false_positive_ratio:.4f},  Val_true_positive_ratio: {val_true_positive_ratio:.4f}, Val_false_positive_ratio:{val_false_positive_ratio:.4f}')
        # print(f'Epoch {epoch+1}/{epochs}, Train_true_positive_ratio: {train_true_positive_ratio:.4f}, Val_true_positive_ratio: {val_true_positive_ratio:.4f}')
        # scheduler.step()
        
        # # Save the best model
        if dice_score > best_dice:
            best_dice = dice_score
            torch.save(model.state_dict(), f'{path}/fold_{i}/ROI_ckpt_35/BCSS_effi-Unet_roi_best_1+1_threod_allmask.pth')



# print("模型：", model.__class__.__name__)
print(train_images_dir)
print("fold:",i)

train_model(model, train_loader, val_loader, loss_fn, optimizer,  epochs=200)




