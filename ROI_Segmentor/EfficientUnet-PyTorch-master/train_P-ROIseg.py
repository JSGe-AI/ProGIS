import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from scipy.ndimage.morphology import distance_transform_edt
from skimage.morphology import skeletonize_3d
from skimage.measure import label as label_1
from skimage.measure import regionprops

# 确保这两个自定义模块在同一级目录下或在你的 Python Path 中
from efficientunet import *
from UNet import *


# =========================================================================
# 1. 用户配置参数区域 (User Configuration Area) - 所有需要修改的参数都在这里
# =========================================================================

# 数据和路径配置
BASE_PATH = "/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/125WSI"
FOLD = 5  # 当前训练的折数

# 数据集子文件夹配置 (保持相对路径即可)
IMAGE_SUBDIR = "ROI_data/all_class/image_npy"
MASK_SUBDIR = "ROI_data/all_class/mask_npy"
SIGNAL_SUBDIR = "ROI_data/all_class/signal_maxconnect_line_npy"

# 模型保存配置
SAVE_CKPT_DIR_NAME = "ROI_ckpt_35"  # 检查点保存文件夹名称
SAVE_MODEL_NAME = "BCSS_effi-Unet_roi_best_1+1_threod_allmask.pth" # 最佳模型命名

# 超参数配置 (Hyperparameters)
BATCH_SIZE = 42
NUM_WORKERS = 8
LEARNING_RATE = 4e-4
EPOCHS = 200

# =========================================================================


def get_dist_maps_batch(binary_tensor_batch, norm_delimeter=260.0):
    """批量计算每个像素到二值张量中指定点的归一化平方距离"""
    device = binary_tensor_batch.device
    binary_tensor_batch = binary_tensor_batch.cpu().numpy()

    batch_size, num_layers, height, width = binary_tensor_batch.shape
    dist_maps_batch = np.full((batch_size, num_layers, height, width), 1e6, dtype=np.float32)

    for batch_idx in range(batch_size):
        for layer in range(num_layers):
            binary_layer = binary_tensor_batch[batch_idx, layer]
            if np.any(binary_layer == 255):
                foreground = (binary_layer == 255)
                dist_map = distance_transform_edt(~foreground)
                dist_map = (dist_map / norm_delimeter) ** 2
                dist_maps_batch[batch_idx, layer] = dist_map
            else:
                continue

    dist_maps_batch = torch.from_numpy(dist_maps_batch).to(device)
    return dist_maps_batch


def generateGuidingSignal(binaryMask):
    """计算误差区域的形态学骨架"""
    binaryMask = binaryMask.to(torch.uint8)
    
    if binaryMask.sum() > 1:
        distance_map = distance_transform_edt(binaryMask.cpu().numpy())
        tempMean = distance_map.mean()
        tempStd = distance_map.std()
        
        tempThresh = np.random.uniform(tempMean - tempStd, tempMean + tempStd)
        if tempThresh < 0:
            tempThresh = np.random.uniform(tempMean / 2, tempMean + tempStd / 2)
        
        newMask = distance_map > tempThresh
        if newMask.sum() == 0:
            newMask = distance_map > (tempThresh / 2)
        if newMask.sum() == 0:
            newMask = binaryMask

        skel = skeletonize_3d(newMask)
        skel = torch.tensor(skel, dtype=torch.float32, device=binaryMask.device)
    else:
        skel = torch.zeros_like(binaryMask, dtype=torch.float32, device=binaryMask.device)

    return skel


def processMasks(pred_mask_all, GT_mask_all):
    """批量处理GT_mask和pred_mask，计算每个样本的前景和背景骨架信号"""
    pred_mask_all = (pred_mask_all > 0.5).float()
    batch_size, _, H, W = pred_mask_all.shape
    output = torch.zeros(batch_size, 2, H, W, device=pred_mask_all.device, dtype=torch.float32)

    GT_mask_all = GT_mask_all.squeeze(1)
    pred_mask_all = pred_mask_all.squeeze(1)

    fg_all = ((GT_mask_all == 1) & (pred_mask_all == 0)).float()
    bg_all = ((GT_mask_all == 0) & (pred_mask_all == 1)).float()

    for i in range(batch_size):
        fg = fg_all[i]
        bg = bg_all[i]

        fg_largest = get_largest_connected_component(fg)
        bg_largest = get_largest_connected_component(bg)

        fg_area = fg_largest.sum().item()
        bg_area = bg_largest.sum().item()

        if fg_area >= bg_area:
            largest_connected = fg_largest
            output[i, 0] = generateGuidingSignal(largest_connected) if largest_connected.sum() > 0 else torch.zeros_like(fg, dtype=torch.float32, device=fg.device)
        else:
            largest_connected = bg_largest
            output[i, 1] = generateGuidingSignal(largest_connected) if largest_connected.sum() > 0 else torch.zeros_like(bg, dtype=torch.float32, device=bg.device)

    return output


def get_largest_connected_component(mask):
    """获取mask中最大连通区域"""
    if mask.sum() > 0:
        labeled_mask = label_1(mask.cpu().numpy(), connectivity=1)
        regions = regionprops(labeled_mask)
        if regions:
            largest_region = max(regions, key=lambda r: r.area)
            largest_component = (labeled_mask == largest_region.label)
            return torch.from_numpy(largest_component).to(mask.device, dtype=torch.float32)
    return torch.zeros_like(mask, device=mask.device)


# ------------------------- 评价指标 -------------------------

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
        return float('nan')
    else:
        return intersection / union


def compute_miou_binary(pred, target):
    iou_background = compute_iou(pred, target, 0)
    iou_foreground = compute_iou(pred, target, 1)
    return np.nanmean([iou_background, iou_foreground])


def calculate_binary_segmentation_accuracy(preds, labels):
    if preds.dim() == 4:
        preds = preds.squeeze(1)
    if labels.dim() == 4:
        labels = labels.squeeze(1)
    
    assert preds.shape == labels.shape, "预测值和标签的形状必须一致"
    preds = (preds > 0.5).float()
    
    correct = (preds == labels).float().sum(dim=[1, 2])
    total_pixels_per_sample = labels.size(1) * labels.size(2)
    
    accuracy_per_sample = correct / total_pixels_per_sample
    mean_accuracy = accuracy_per_sample.mean().item()
    
    return accuracy_per_sample, mean_accuracy


def calculate_metrics(outputs, masks):
    outputs_min = outputs.min()
    outputs_max = outputs.max()
    
    if outputs_max - outputs_min > 0:
        outputs = (outputs - outputs_min) / (outputs_max - outputs_min)
    else:
        outputs = torch.zeros_like(outputs)
    
    outputs = (outputs > 0.6).float()
    masks = masks.float()
    
    outputs_1 = outputs == 1
    masks_1 = masks == 1
    masks_0 = masks == 0
    
    true_positive = (outputs_1 & masks_1).sum().item()
    false_positive = (outputs_1 & masks_0).sum().item()
    total_pred_1 = outputs_1.sum().item()
    
    if total_pred_1 == 0:
        return 0, 0
    
    true_positive_ratio = true_positive / total_pred_1
    false_positive_ratio = false_positive / total_pred_1
    return true_positive_ratio, false_positive_ratio


# ------------------------- 数据集定义 -------------------------

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, signal_dir, filenames):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.signal_dir = signal_dir
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_dir, filename)
        mask_path = os.path.join(self.masks_dir, filename)
        signal_path = os.path.join(self.signal_dir, filename)

        image = np.load(image_path)
        mask = np.load(mask_path)
        signal = np.load(signal_path)

        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        signal = torch.tensor(signal, dtype=torch.float32)

        return image, mask, signal


def get_filenames_from_folder(folder_path):
    return [filename for filename in os.listdir(folder_path) if filename.endswith('.npy')]


# ------------------------- 训练逻辑 -------------------------

def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs, save_path):
    best_dice = 0.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    for epoch in range(epochs):
        # ----------------- 训练阶段 -----------------
        model.train()
        train_loss = 0.0
        train_dice_score = 0.0
        train_accuracy = 0.0
        
        train_loader_tqdm = tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs} [train]', unit='batch')
        
        for images, masks, aux_inputs in train_loader_tqdm:
            images, aux_inputs, masks = images.to(device), aux_inputs.to(device), masks.to(device)
            
            # 第一次推断
            pred_mask = torch.zeros_like(masks)
            input_tensor = torch.cat((images, pred_mask, aux_inputs), dim=1)
            
            optimizer.zero_grad()
            pred_mask_1 = model(input_tensor)
            
            # 处理信号与第二次推断
            signal = processMasks(pred_mask_1, masks)
            union_signal = torch.bitwise_or(signal.to(torch.uint8), aux_inputs.to(torch.uint8))
            pre_mask_1_threod = (pred_mask_1 >= 0.5).int()
            
            input_tensor_2 = torch.cat((images, pre_mask_1_threod, union_signal), dim=1)
            pred_mask_2 = model(input_tensor_2)

            loss = dice_loss(pred_mask_1, masks) + dice_loss(pred_mask_2, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
            outputs = pred_mask_2
            outputs = (outputs >= 0.5).int()
            
            train_dice_score += dice_coeff(outputs, masks).item() * images.size(0)
            _, batch_accuracy = calculate_binary_segmentation_accuracy(outputs, masks)
            train_accuracy += batch_accuracy * images.size(0) 
            
        train_loss /= len(train_loader.dataset)
        train_dice_score /= len(train_loader.dataset)
        train_accuracy /= len(train_loader.dataset)

        # ----------------- 验证阶段 -----------------
        model.eval()
        val_loss = 0.0
        dice_score = 0.0
        val_accuracy = 0.0
        mean_iou = 0.0

        with torch.no_grad():
            iou_scores = []
            val_loader_tqdm = tqdm(val_loader, total=len(val_loader), desc=f'Epoch {epoch+1}/{epochs} [val]', unit='batch')
            
            for images, masks, aux_inputs in val_loader_tqdm:
                images, aux_inputs, masks = images.to(device), aux_inputs.to(device), masks.to(device)
                
                # 第一次推断
                pred_mask = torch.zeros_like(masks)
                input_tensor = torch.cat((images, pred_mask, aux_inputs), dim=1)
                pred_mask_1 = model(input_tensor)
                
                # 处理信号与第二次推断
                signal = processMasks(pred_mask_1, masks)
                union_signal = torch.bitwise_or(signal.to(torch.uint8), aux_inputs.to(torch.uint8))
                pre_mask_1_threod = (pred_mask_1 >= 0.5).int()
                
                input_tensor_2 = torch.cat((images, pre_mask_1_threod, union_signal), dim=1)
                pred_mask_2 = model(input_tensor_2)
    
                loss = dice_loss(pred_mask_1, masks) + dice_loss(pred_mask_2, masks)
                val_loss += loss.item() * images.size(0)
                
                outputs = pred_mask_2
                outputs = (outputs >= 0.5).int()
                
                dice_score += dice_coeff(outputs, masks).item() * images.size(0)
                _, batch_accuracy = calculate_binary_segmentation_accuracy(outputs, masks)
                val_accuracy += batch_accuracy * images.size(0)
                
                # 计算 mIOU
                outputs_np = outputs.squeeze(1).cpu().numpy()
                preds = (outputs_np >= 0.5).astype(int)
                masks_np = masks.cpu().numpy()
                
                for pred, mask in zip(preds, masks_np):
                    miou = compute_miou_binary(pred, mask)
                    if not np.isnan(miou):
                        iou_scores.append(miou)
            
            if iou_scores:
                mean_iou = np.mean(iou_scores)

        val_loss /= len(val_loader.dataset)
        dice_score /= len(val_loader.dataset)
        val_accuracy /= len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Dice: {train_dice_score:.4f} | Train Acc: {train_accuracy:.4f}')
        print(f'             | Val Loss: {val_loss:.4f}   | Val Dice: {dice_score:.4f}   | Val Mean IOU: {mean_iou:.4f} | Val Acc: {val_accuracy:.4f}')
        
        # 保存最优模型
        if dice_score > best_dice:
            best_dice = dice_score
            os.makedirs(os.path.dirname(save_path), exist_ok=True) # 确保目标文件夹存在
            torch.save(model.state_dict(), save_path)
            print(f'>>> Best model saved to: {save_path}')


# ------------------------- 主函数入口 -------------------------

if __name__ == '__main__':
    # 1. 构建最终路径
    train_images_dir = f"{BASE_PATH}/fold_{FOLD}/train/{IMAGE_SUBDIR}"
    train_masks_dir = f"{BASE_PATH}/fold_{FOLD}/train/{MASK_SUBDIR}"
    train_signal_dir = f"{BASE_PATH}/fold_{FOLD}/train/{SIGNAL_SUBDIR}"

    val_images_dir = f"{BASE_PATH}/fold_{FOLD}/val/{IMAGE_SUBDIR}"
    val_masks_dir = f"{BASE_PATH}/fold_{FOLD}/val/{MASK_SUBDIR}"
    val_signal_dir = f"{BASE_PATH}/fold_{FOLD}/val/{SIGNAL_SUBDIR}"
    
    save_model_path = f"{BASE_PATH}/fold_{FOLD}/{SAVE_CKPT_DIR_NAME}/{SAVE_MODEL_NAME}"

    print("="*50)
    print(f"Training on Fold: {FOLD}")
    print(f"Train images directory: {train_images_dir}")
    print("="*50)

    # 2. 准备数据集和 DataLoader
    train_filenames = get_filenames_from_folder(train_images_dir)
    val_filenames = get_filenames_from_folder(val_images_dir)

    train_dataset = CustomDataset(train_images_dir, train_masks_dir, train_signal_dir, train_filenames)
    val_dataset = CustomDataset(val_images_dir, val_masks_dir, val_signal_dir, val_filenames)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 3. 初始化模型、优化器和损失函数
    model = get_efficientunet_b0(out_channels=1, concat_input=True, pretrained=False).cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    loss_fn = nn.BCELoss()

    # 4. 开始训练
    train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs=EPOCHS, save_path=save_model_path)