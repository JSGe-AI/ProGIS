import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from scipy.ndimage.morphology import distance_transform_edt
from skimage.morphology import skeletonize_3d

# 需要确保 networks.vit_seg_modeling 模块在你当前的工作目录下
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


# =========================================================================
# 1. 用户配置参数区域 (User Configuration Area) - 所有需要修改的参数都在这里
# =========================================================================

# --- 基础设置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MULTI_GPU = False
FOLD = 5  # 当前训练的折数

# --- 训练超参数 ---
LEARNING_RATE = 4e-4
BATCH_SIZE = 8
NUM_WORKERS = 4
EPOCHS = 80
WEIGHT_DECAY = 5e-5
TEMPERATURE = 0.3  # 对比学习温度系数

# --- 模型参数 (ViT / TransUNet) ---
VIT_NAME = 'R50-ViT-B_16'
IMG_SIZE = 512
VIT_PATCHES_SIZE = 16
NUM_CLASSES = 1
N_SKIP = 3

# --- 路径与数据配置 ---
# 基础目录
BASE_PATH = "/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/125WSI"

# 训练集路径
TRAIN_IMAGES_DIR = f"{BASE_PATH}/fold_{FOLD}/train/Contrast_learning/image_npy"
TRAIN_MASKS_DIR = f"{BASE_PATH}/fold_{FOLD}/train/Contrast_learning/mask_npy"
TRAIN_SUPERPIXEL_DIR = f"{BASE_PATH}/fold_{FOLD}/train/Contrast_learning/image_SLIC_500"

# 验证集路径
VAL_IMAGES_DIR = f"{BASE_PATH}/fold_{FOLD}/val/Contrast_learning/image_npy"
VAL_MASKS_DIR = f"{BASE_PATH}/fold_{FOLD}/val/Contrast_learning/mask_npy"
VAL_SUPERPIXEL_DIR = f"{BASE_PATH}/fold_{FOLD}/val/Contrast_learning/image_SLIC_500"

# 模型保存配置
SAVE_DIR = f"{BASE_PATH}/fold_{FOLD}/Trans_unet"

# =========================================================================


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
    miou = np.nanmean([iou_background, iou_foreground])
    return miou

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


# ------------------------- 损失函数 -------------------------

def calc_dc_loss_sp(masks, all_perpixel_features, superpixels):
    b, c, h, w = all_perpixel_features.shape
    loss_dc = 0
    temperature = TEMPERATURE

    def get_sps_features(all_perpixel_features, superpixels, sps_list):
        one_sps = superpixels[bi, 0, ...] 
        one_outputs = all_perpixel_features[bi, ...] 
        one_sps_feature_list = []
        for spi in sps_list:
            sp_index = torch.where(one_sps == spi)
            sp_feature = one_outputs[:, sp_index[0], sp_index[1]]
            sp_feature = torch.mean(sp_feature, -1)
            one_sps_feature_list.append(sp_feature)
        sps_feature_list = torch.stack(one_sps_feature_list)
        return sps_feature_list

    def get_sp_contrast_loss(fg_sps_labels_list, bg_sps_labels_list, affinity_matrix_ff, affinity_matrix_fb):
        loss = 0
        for i in range(len(fg_sps_labels_list)):
            cos_smi_sp_ff = affinity_matrix_ff[i, :]
            cos_avg_ff = torch.sum(cos_smi_sp_ff) / len(fg_sps_labels_list)
            
            cos_smi_sp_fb = affinity_matrix_fb[i, :]
            cos_avg_fb = torch.sum(cos_smi_sp_fb) / len(bg_sps_labels_list)
            
            loss += -torch.log(cos_avg_ff / (cos_avg_ff + cos_avg_fb + 1e-8))
        return loss / len(fg_sps_labels_list)
    
    def get_loss_per_pixels_with_fg_bg_sps(all_perpixel_features, superpixels, fg_sps_labels_list, bg_sps_labels_list, fg_sps_feature_list1_norm, bg_sps_feature_list1_norm, affinity_matrix):
        fg_total_loss = 0
        
        one_sps = superpixels[bi, 0, ...]
        one_image_feature = all_perpixel_features[bi, ...]
        affinity_matrix_fg_sum = torch.sum(affinity_matrix, 1)

        # 前景像素与前景超像素块特征相似性增加
        for i in range(len(fg_sps_labels_list)):
            sp_index_fg = torch.where(one_sps == fg_sps_labels_list[i])
            pixels_feature_fg = one_image_feature[:, sp_index_fg[0], sp_index_fg[1]].T
            
            pixels_norm = pixels_feature_fg.norm(dim=1).clone()
            pixels_norm[pixels_norm == 0] = 1e-8
            pixels_feature_fg_norm = pixels_feature_fg / pixels_norm[:, None]
            
            sp_feature_fg_norm = fg_sps_feature_list1_norm[i:i+1] 
            affinity_matrix_pixels = torch.mm(sp_feature_fg_norm, pixels_feature_fg_norm.t())
            affinity_matrix_pixels = torch.exp(affinity_matrix_pixels / temperature)
            e_sim_fg = affinity_matrix_pixels.mean()
            e_dis_fg = affinity_matrix_fg_sum[i]
            
            fg_total_loss += -torch.log(e_sim_fg / (e_sim_fg + e_dis_fg + 1e-8))
            
        total_loss = fg_total_loss / len(fg_sps_labels_list) 
        return total_loss

    num_batch = b
    for bi in range(b): 
        target_sps = superpixels[bi, 0, ...] 
        sps_labels_list = torch.unique(target_sps)
        sps_feature_list = get_sps_features(all_perpixel_features, superpixels, sps_labels_list).to(DEVICE) 

        loss_img = 0
        mask_flat = masks[bi].view(-1) 
        superpixels_flat = superpixels[bi].view(-1) 

        unique_labels = mask_flat.unique()
        fg_labels = unique_labels[unique_labels > 0] 
        fg_labels_num = len(fg_labels)

        for fg_label in fg_labels:
            fg_sps_labels_list = []
            bg_sps_labels_list = []

            for label in sps_labels_list:
                mask_for_superpixel = mask_flat[superpixels_flat == label]
                num_fg_pixels = torch.sum(mask_for_superpixel == fg_label)
                num_bg_pixels = torch.sum(mask_for_superpixel != fg_label)

                if num_fg_pixels > num_bg_pixels:
                    fg_sps_labels_list.append(label)
                else:
                    bg_sps_labels_list.append(label)

            fg_sps_labels_list = torch.tensor(fg_sps_labels_list).long().to(DEVICE)
            bg_sps_labels_list = torch.tensor(bg_sps_labels_list).long().to(DEVICE)
            
            fg_sps_feature_list = [sps_feature_list[label] for label in fg_sps_labels_list] 
            bg_sps_feature_list = [sps_feature_list[label] for label in bg_sps_labels_list] 
           
            if len(fg_sps_feature_list) == 0 or len(bg_sps_feature_list) == 0:
                fg_labels_num -= 1
                continue    
                        
            fg_sps_feature_list1 = torch.stack(fg_sps_feature_list)
            bg_sps_feature_list1 = torch.stack(bg_sps_feature_list)

            fg_sps_feature_list1_norm = fg_sps_feature_list1 / fg_sps_feature_list1.norm(dim=1)[:, None] 
            bg_sps_feature_list1_norm = bg_sps_feature_list1 / bg_sps_feature_list1.norm(dim=1)[:, None] 

            affinity_matrix_ff = torch.mm(fg_sps_feature_list1_norm, fg_sps_feature_list1_norm.t())
            affinity_matrix_ff = torch.exp(affinity_matrix_ff / temperature)
            
            affinity_matrix_fb = torch.mm(fg_sps_feature_list1_norm, bg_sps_feature_list1_norm.t())
            affinity_matrix_fb = torch.exp(affinity_matrix_fb / temperature)
            
            loss_sp1 = get_sp_contrast_loss(fg_sps_labels_list, fg_sps_labels_list, affinity_matrix_ff, affinity_matrix_fb)
            loss_px = get_loss_per_pixels_with_fg_bg_sps(all_perpixel_features, superpixels, fg_sps_labels_list, bg_sps_labels_list, fg_sps_feature_list1_norm, bg_sps_feature_list1_norm, affinity_matrix_fb) 
            
            loss_img += loss_sp1 + loss_px  
            
        if fg_labels_num != 0:    
            loss_dc += loss_img / fg_labels_num
        else:
            num_batch -= 1
            continue
            
    if num_batch != 0:
        loss_dc = loss_dc / num_batch
    else:
        loss_dc = torch.tensor(1e-8, device=DEVICE)

    return loss_dc


# ------------------------- 数据集定义 -------------------------

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, suppixel_dir, filenames):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.suppixel_dir = suppixel_dir
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_dir, filename)
        mask_path = os.path.join(self.masks_dir, filename)
        suppixel_path = os.path.join(self.suppixel_dir, filename)

        image = np.load(image_path)
        mask = np.load(mask_path)
        suppixel = np.load(suppixel_path)

        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) 
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) 
        suppixel = torch.tensor(suppixel.transpose(2, 0, 1), dtype=torch.float32) 

        return image, mask, suppixel, filename
    
def get_filenames_from_folder(folder_path):
    return [filename for filename in os.listdir(folder_path) if filename.endswith('.npy')]


# ------------------------- 训练逻辑 -------------------------

def train_model(model, train_loader, val_loader, optimizer, epochs=50):
    best_loss = 10.0
    model.to(DEVICE)
    os.makedirs(SAVE_DIR, exist_ok=True)  # 确保保存目录存在

    epoch_pbar = tqdm(range(epochs), desc="Overall Training Progress", unit="epoch")

    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        train_batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False, unit="batch")
        
        for images, masks, superpixels, filenames in train_batch_pbar:
            images, masks, superpixels = images.to(DEVICE), masks.to(DEVICE), superpixels.to(DEVICE)
            optimizer.zero_grad()

            all_pixel_features = model(images)
            loss = calc_dc_loss_sp(masks, all_pixel_features, superpixels)

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_batch_pbar.set_postfix(batch_loss=f"{loss.item():.4f}")
            
        train_loss /= len(train_loader.dataset)
        
        # ----------------- 验证阶段 -----------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            val_batch_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]", leave=False, unit="batch")
            
            for images, masks, superpixels, filenames in val_batch_pbar:
                images, masks, superpixels = images.to(DEVICE), masks.to(DEVICE), superpixels.to(DEVICE)
                all_pixel_features = model(images)

                loss = calc_dc_loss_sp(masks, all_pixel_features, superpixels)
                val_loss += loss.item() * images.size(0)
                
                val_batch_pbar.set_postfix(batch_loss=f"{loss.item():.4f}")
            
            val_loss /= len(val_loader.dataset)
            epoch_pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")
            print(f'Epoch {epoch+1}/{epochs}, Train Loss (CombinedLoss): {train_loss:.4f}, Val Loss : {val_loss:.4f}')
            
        # ----------------- 模型保存 -----------------
        if val_loss < best_loss or (epoch + 1) % 1 == 0:   # 每隔1个epoch保存一次
            if val_loss < best_loss:
                best_loss = val_loss
                filename = os.path.join(SAVE_DIR, 'transunet_best.pth')
                print(f"Epoch {epoch+1}: Model saved with lowest Val Loss: {val_loss:.4f}")
            else:
                filename = os.path.join(SAVE_DIR, f'transunet_{epoch+1}_loss{val_loss:.4f}.pth')
                print(f"Epoch {epoch+1}: Model saved at epoch interval.")
            
            # 使用 model.module 保存，以防之前被 DataParallel 包装过导致后续无法单卡推理
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(state_dict, filename)


# ------------------------- 主函数入口 -------------------------

if __name__ == '__main__':
    # 1. 获取训练集和验证集的文件名
    train_filenames = get_filenames_from_folder(TRAIN_IMAGES_DIR)
    val_filenames = get_filenames_from_folder(VAL_IMAGES_DIR)

    # 2. 创建自定义数据集类的实例
    train_dataset = CustomDataset(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, TRAIN_SUPERPIXEL_DIR, train_filenames)
    val_dataset = CustomDataset(VAL_IMAGES_DIR, VAL_MASKS_DIR, VAL_SUPERPIXEL_DIR, val_filenames)

    # 3. 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 4. 初始化 TransUNet (ViT) 模型
    config_vit = CONFIGS_ViT_seg[VIT_NAME]
    config_vit.n_classes = NUM_CLASSES
    config_vit.n_skip = N_SKIP
    if 'R50' in VIT_NAME:
        config_vit.patches.grid = (int(IMG_SIZE / VIT_PATCHES_SIZE), int(IMG_SIZE / VIT_PATCHES_SIZE))
        
    model = ViT_seg(config_vit, img_size=IMG_SIZE, num_classes=config_vit.n_classes)
    model.load_from(weights=np.load(config_vit.pretrained_path))

    if MULTI_GPU:
        model = nn.DataParallel(model)

    # 5. 初始化优化器
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 6. 开始训练
    train_model(model, train_loader, val_loader, optimizer, epochs=EPOCHS)