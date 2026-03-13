import os
import time
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import models

import scipy.ndimage as ndi


# =========================================================================
# 1. 用户配置参数区域 (User Configuration Area)
# =========================================================================

# 基础设置
SEED = 42
MULTI_GPU = True

# 训练超参数
LEARNING_RATE = 4e-4
BATCH_SIZE = 20
NUM_WORKERS = 4
EPOCHS = 50

# 路径与数据配置
BASE_PATH = "/data_nas2/gjs/ISF_pixel_level_data/Gastric_new"
FOLD = 4  # 当前训练折数

# 数据集子文件夹配置 (相对路径)
IMAGE_SUBDIR = "Contrast_learning/image_npy"
MASK_SUBDIR = "Contrast_learning/mask_npy"
SUPERPIXEL_SUBDIR = "Contrast_learning/image_SLIC_500"

# 模型保存配置
SAVE_CKPT_DIR_NAME = "ckpt"
SAVE_MODEL_PREFIX = "cos2_Gastric_resunet"

# =========================================================================

# 设置随机种子
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ------------------------- 模型定义 -------------------------

def convbnrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(out_channels, eps=1e-5),
        nn.ReLU(inplace=True),
    )

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class ResNetUNetHead(nn.Module):
    def __init__(self, freeze=False, pretrained=True):
        super().__init__()
        self.freeze = freeze
        base_model = models.resnet18(pretrained=pretrained)
        self.base_layers = list(base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) 
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) 
        self.layer2 = self.base_layers[5]  
        self.layer3 = self.base_layers[6]  
        self.layer4 = self.base_layers[7]  
        
        self.upsample3 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv_up31 = convbnrelu(256 + 512, 512, 3, 1)
        self.conv_up32 = convbnrelu(512, 512, 3, 1)
        
        self.upsample2 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv_up21 = convbnrelu(128 + 512, 256, 3, 1)
        self.conv_up22 = convbnrelu(256, 256, 3, 1)
        
        self.upsample1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv_up11 = convbnrelu(64 + 256, 128, 3, 1)
        self.conv_up12 = convbnrelu(128, 128, 3, 1)
        
        self.upsample0 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.conv_up01 = convbnrelu(64 + 128, 64, 3, 1)
        self.conv_up02 = convbnrelu(64, 64, 3, 1)
        
        # projection head
        self.conv_proh1 = nn.Conv2d(64, 64, 1)
        self.conv_proh2 = nn.Conv2d(64, 32, 1)
        self.upsample_last = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)

    def forward(self, input):
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = self.upsample3(layer4)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up31(x)
        x = self.conv_up32(x)

        x = self.upsample2(x)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up21(x)
        x = self.conv_up22(x)

        x = self.upsample1(x)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up11(x)
        x = self.conv_up12(x)

        x = self.upsample0(x)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up01(x)
        x = self.conv_up02(x)

        x = self.conv_proh1(x)
        x = self.conv_proh2(x)
        x = self.upsample_last(x)

        return x

class ResNetUNet(nn.Module):
    def __init__(self, freeze=False, pretrained=True):
        super().__init__()
        self.freeze = freeze
        base_model = models.resnet18(pretrained=pretrained)
        self.base_layers = list(base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) 
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) 
        self.layer2 = self.base_layers[5]  
        self.layer3 = self.base_layers[6]  
        self.layer4 = self.base_layers[7]  
        
        self.upsample3 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv_up31 = convbnrelu(256 + 512, 512, 3, 1)
        self.conv_up32 = convbnrelu(512, 512, 3, 1)
        
        self.upsample2 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv_up21 = convbnrelu(128 + 512, 256, 3, 1)
        self.conv_up22 = convbnrelu(256, 256, 3, 1)
        
        self.upsample1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv_up11 = convbnrelu(64 + 256, 128, 3, 1)
        self.conv_up12 = convbnrelu(128, 128, 3, 1)
        
        self.upsample0 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.conv_up01 = convbnrelu(64 + 128, 64, 3, 1)
        self.conv_up02 = convbnrelu(64, 64, 3, 1)
        
        self.upsample = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        
        # projection head
        self.conv_proh1 = nn.Conv2d(64, 64, 1)
        self.conv_proh2 = nn.Conv2d(64, 32, 1)

    def forward(self, input):
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = self.upsample3(layer4)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up31(x)
        x = self.conv_up32(x)

        x = self.upsample2(x)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up21(x)
        x = self.conv_up22(x)

        x = self.upsample1(x)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up11(x)
        x = self.conv_up12(x)

        x = self.upsample0(x)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up01(x)
        x = self.conv_up02(x)

        x = self.upsample(x)
        x = self.conv_proh1(x)
        x = self.conv_proh2(x)

        return x 


# ------------------------- 损失函数 -------------------------

def calc_dc_loss_sp(masks, all_perpixel_features, superpixels):
    b, c, h, w = all_perpixel_features.shape
    loss_dc = 0
    temperature = 0.3

    def get_sps_features(all_perpixel_features, superpixels, sps_list):
        one_sps = superpixels[bi,0, ...] 
        one_outputs = all_perpixel_features[bi, ...] 
        one_sps_feature_list = []
        for spi in sps_list:
            sp_index = torch.where(one_sps==spi)
            sp_feature = one_outputs[:, sp_index[0], sp_index[1]]
            sp_feature = torch.mean(sp_feature, -1)
            one_sps_feature_list.append(sp_feature)
        sps_feature_list = torch.stack(one_sps_feature_list)
        return sps_feature_list

    def get_sp_contrast_loss(fg_sps_labels_list, bg_sps_labels_list, affinity_matrix_ff, affinity_matrix_fb):
        loss = 0
        for i in range(len(fg_sps_labels_list)):
            cos_smi_sp_ff = affinity_matrix_ff[i, :]
            cos_all_ff = torch.sum(cos_smi_sp_ff)    
            cos_avg_ff = cos_all_ff/len(fg_sps_labels_list)
            
            cos_smi_sp_fb = affinity_matrix_fb[i, :]
            cos_all_fb = torch.sum(cos_smi_sp_fb)    
            cos_avg_fb = cos_all_fb/len(bg_sps_labels_list)
            
            loss += -torch.log(cos_avg_ff/(cos_avg_ff + cos_avg_fb + 1e-8))
        return loss/len(fg_sps_labels_list)
    
    def get_loss_per_pixels_with_fg_bg_sps(all_perpixel_features, superpixels, fg_sps_labels_list, bg_sps_labels_list, fg_sps_feature_list1_norm, bg_sps_feature_list1_norm, affinity_matrix):
        fg_total_loss = 0
        bg_total_loss = 0
        
        one_sps = superpixels[bi, 0, ...]  
        one_image_feature = all_perpixel_features[bi, ...] 
        affinity_matrix_fg_sum = torch.sum(affinity_matrix, 1)
        affinity_matrix_bg_sum = torch.sum(affinity_matrix, 0)
        
        for i in range(len(fg_sps_labels_list)):
            sp_index_fg = torch.where(one_sps == fg_sps_labels_list[i])
            pixels_feature_fg = one_image_feature[:, sp_index_fg[0], sp_index_fg[1]].T
            pixels_feature_fg_norm = pixels_feature_fg / pixels_feature_fg.norm(dim=1)[:, None]
            
            sp_feature_fg_norm = fg_sps_feature_list1_norm[i:i+1] 
            affinity_matrix_pixels = (torch.mm(sp_feature_fg_norm, pixels_feature_fg_norm.t()))**2
            affinity_matrix_pixels = torch.exp(affinity_matrix_pixels / temperature)
            e_sim_fg = affinity_matrix_pixels.mean()
            e_dis_fg = affinity_matrix_fg_sum[i]
            fg_total_loss += -torch.log(e_sim_fg /(e_sim_fg + e_dis_fg + 1e-8))
            
        for i in range(len(bg_sps_labels_list)):
            sp_index_bg = torch.where(one_sps == bg_sps_labels_list[i])
            pixels_feature_bg = one_image_feature[:, sp_index_bg[0], sp_index_bg[1]].T
            pixels_feature_bg_norm = pixels_feature_bg / pixels_feature_bg.norm(dim=1)[:, None]
            
            sp_feature_bg_norm = bg_sps_feature_list1_norm[i:i+1] 
            affinity_matrix_pixels = (torch.mm(sp_feature_bg_norm, pixels_feature_bg_norm.t()))**2
            affinity_matrix_pixels = torch.exp(affinity_matrix_pixels / temperature)
            e_sim_bg = affinity_matrix_pixels.mean()
            e_dis_bg = affinity_matrix_bg_sum[i]
            bg_total_loss += -torch.log(e_sim_bg /(e_sim_bg + e_dis_bg + 1e-8))
        
        total_loss = fg_total_loss / len(fg_sps_labels_list) + bg_total_loss / len(bg_sps_labels_list)
        return total_loss
        
    num_batch = b
    for bi in range(b): 
        target_sps = superpixels[bi,0,...] 
        sps_labels_list = torch.unique(target_sps)
        sps_feature_list = get_sps_features(all_perpixel_features, superpixels, sps_labels_list).cuda() 
        
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

            fg_sps_labels_list = torch.tensor(fg_sps_labels_list).long()
            bg_sps_labels_list = torch.tensor(bg_sps_labels_list).long()
            
            fg_sps_feature_list = [sps_feature_list[label] for label in fg_sps_labels_list]   
            bg_sps_feature_list = [sps_feature_list[label] for label in bg_sps_labels_list]   
           
            if len(fg_sps_feature_list) == 0 or len(bg_sps_feature_list) == 0:
                fg_labels_num -= 1
                continue    
                        
            fg_sps_feature_list1 = torch.stack(fg_sps_feature_list)
            bg_sps_feature_list1 = torch.stack(bg_sps_feature_list)

            fg_sps_feature_list1_norm = fg_sps_feature_list1 / fg_sps_feature_list1.norm(dim=1)[:, None]  
            bg_sps_feature_list1_norm = bg_sps_feature_list1 / bg_sps_feature_list1.norm(dim=1)[:, None]  

            affinity_matrix_ff = (torch.mm(fg_sps_feature_list1_norm, fg_sps_feature_list1_norm.t()))**2
            affinity_matrix_ff = torch.exp(affinity_matrix_ff / temperature)
            
            affinity_matrix_fb = (torch.mm(fg_sps_feature_list1_norm, bg_sps_feature_list1_norm.t()))**2
            affinity_matrix_fb = torch.exp(affinity_matrix_fb / temperature)
            
            loss_sp1 = get_sp_contrast_loss(fg_sps_labels_list, fg_sps_labels_list, affinity_matrix_ff , affinity_matrix_fb)
            loss_px = get_loss_per_pixels_with_fg_bg_sps(all_perpixel_features, superpixels, fg_sps_labels_list, bg_sps_labels_list, fg_sps_feature_list1_norm , bg_sps_feature_list1_norm , affinity_matrix_fb) 
            
            loss_img += loss_sp1 + loss_px  
            
        if fg_labels_num != 0 :    
            loss_dc += loss_img / fg_labels_num
        else :
            num_batch -= 1
            continue
            
    if num_batch != 0:
        loss_dc = loss_dc / num_batch
    else:
        loss_dc = torch.tensor(1e-8, device=all_perpixel_features.device)

    return loss_dc


def Euclidean_distance_loss_sp(masks, all_perpixel_features, superpixels):
    b, c, h, w = all_perpixel_features.shape
    loss_dc = 0
    temperature = 0.3

    def get_sps_features(all_perpixel_features, superpixels, sps_list):
        one_sps = superpixels[bi,0, ...] 
        one_outputs = all_perpixel_features[bi, ...] 
        one_sps_feature_list = []
        for spi in sps_list:
            sp_index = torch.where(one_sps==spi)
            sp_feature = one_outputs[:, sp_index[0], sp_index[1]]
            sp_feature = torch.mean(sp_feature, -1)
            one_sps_feature_list.append(sp_feature)
        sps_feature_list = torch.stack(one_sps_feature_list)
        return sps_feature_list

    def get_sp_contrast_loss(fg_sps_labels_list, bg_sps_labels_list, affinity_matrix_ff, affinity_matrix_fb):
        loss = 0
        for i in range(len(fg_sps_labels_list)):
            cos_smi_sp_ff =  affinity_matrix_ff[i, :]
            cos_all_ff = torch.sum(cos_smi_sp_ff)    
            cos_avg_ff = cos_all_ff/len(fg_sps_labels_list)
            
            cos_smi_sp_fb =  affinity_matrix_fb[i, :]
            cos_all_fb = torch.sum(cos_smi_sp_fb)    
            cos_avg_fb = cos_all_fb/len(bg_sps_labels_list)
            
            loss += -torch.log(cos_avg_fb/(cos_avg_ff + cos_avg_fb + 1e-8))
        return loss/len(fg_sps_labels_list)
    
    def get_loss_per_pixels_with_fg_bg_sps(all_perpixel_features, superpixels, fg_sps_labels_list, bg_sps_labels_list, fg_sps_feature_list1_norm, bg_sps_feature_list1_norm, affinity_matrix):
        fg_total_loss = 0
        bg_total_loss = 0
        
        one_sps = superpixels[bi, 0, ...]  
        one_image_feature = all_perpixel_features[bi, ...] 
        affinity_matrix_fg_sum = torch.mean(affinity_matrix, dim=1)
        affinity_matrix_bg_sum = torch.mean(affinity_matrix, dim=0)
        
        for i in range(len(fg_sps_labels_list)):
            sp_index_fg = torch.where(one_sps == fg_sps_labels_list[i])
            pixels_feature_fg = one_image_feature[:, sp_index_fg[0], sp_index_fg[1]].T
            pixels_feature_fg_norm = pixels_feature_fg 
            
            sp_feature_fg_norm = fg_sps_feature_list1_norm[i:i+1] 
            affinity_matrix_pixels = torch.cdist(sp_feature_fg_norm, pixels_feature_fg_norm, p=2)
            affinity_matrix_pixels = torch.exp(affinity_matrix_pixels / temperature)
            e_sim_fg = affinity_matrix_pixels.mean()
            e_dis_fg = affinity_matrix_fg_sum[i]
            fg_total_loss += -torch.log(e_dis_fg /(e_sim_fg + e_dis_fg + 1e-8))
            
        for i in range(len(bg_sps_labels_list)):
            sp_index_bg = torch.where(one_sps == bg_sps_labels_list[i])
            pixels_feature_bg = one_image_feature[:, sp_index_bg[0], sp_index_bg[1]].T
            pixels_feature_bg_norm = pixels_feature_bg
            
            sp_feature_bg_norm = bg_sps_feature_list1_norm[i:i+1] 
            affinity_matrix_pixels = torch.cdist(sp_feature_bg_norm, pixels_feature_bg_norm, p=2)
            affinity_matrix_pixels = torch.exp(affinity_matrix_pixels / temperature)
            e_sim_bg = affinity_matrix_pixels.mean()
            e_dis_bg = affinity_matrix_bg_sum[i]
            bg_total_loss += -torch.log(e_dis_bg /(e_sim_bg + e_dis_bg + 1e-8))
        
        total_loss = fg_total_loss / len(fg_sps_labels_list) + bg_total_loss / len(bg_sps_labels_list)
        return total_loss

    num_batch = b
    for bi in range(b):
        target_sps = superpixels[bi,0,...]
        sps_labels_list = torch.unique(target_sps)
        sps_feature_list = get_sps_features(all_perpixel_features, superpixels, sps_labels_list).cuda() 
        
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

            fg_sps_labels_list = torch.tensor(fg_sps_labels_list).long()
            bg_sps_labels_list = torch.tensor(bg_sps_labels_list).long()
            
            fg_sps_feature_list = [sps_feature_list[label] for label in fg_sps_labels_list]   
            bg_sps_feature_list = [sps_feature_list[label] for label in bg_sps_labels_list]   
           
            if len(fg_sps_feature_list) == 0 or len(bg_sps_feature_list) == 0:
                fg_labels_num -= 1
                continue    
                        
            fg_sps_feature_list1 = torch.stack(fg_sps_feature_list)
            bg_sps_feature_list1 = torch.stack(bg_sps_feature_list)

            fg_sps_feature_list1_norm = fg_sps_feature_list1
            bg_sps_feature_list1_norm = bg_sps_feature_list1
            
            affinity_matrix_ff = torch.cdist(fg_sps_feature_list1_norm, fg_sps_feature_list1_norm, p=2)
            affinity_matrix_ff = torch.exp(affinity_matrix_ff)

            affinity_matrix_fb = torch.cdist(fg_sps_feature_list1_norm, bg_sps_feature_list1_norm, p=2)
            affinity_matrix_fb = torch.exp(affinity_matrix_fb)

            loss_sp1 = get_sp_contrast_loss(fg_sps_labels_list, fg_sps_labels_list, affinity_matrix_ff , affinity_matrix_fb)
            loss_px = get_loss_per_pixels_with_fg_bg_sps(all_perpixel_features, superpixels, fg_sps_labels_list, bg_sps_labels_list, fg_sps_feature_list1_norm , bg_sps_feature_list1_norm , affinity_matrix_fb)
            
            loss_img += loss_sp1 + loss_px  
            
        if fg_labels_num != 0 :    
            loss_dc += loss_img / fg_labels_num
        else :
            num_batch -= 1
            continue
            
    if num_batch != 0:
        loss_dc = loss_dc / num_batch
    else:
        loss_dc = torch.tensor(1e-8, device=all_perpixel_features.device)

    return loss_dc


def soft_dice_loss(logits, targets, epsilon=1e-6):
    num_classes = logits.size(1)
    probs = torch.softmax(logits, dim=1) 
    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    
    intersection = (probs * targets_one_hot).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
    
    dice_score = (2. * intersection + epsilon) / (union + epsilon)
    dice_loss = 1 - dice_score.mean() 
    return dice_loss

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss() 

    def forward(self, logits, targets):
        targets = targets.squeeze(1)
        targets = targets - 1  # 调整到从 0 开始
        ce_loss = self.ce_loss(logits, targets)
        dice_loss = soft_dice_loss(logits, targets)
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * dice_loss
        return total_loss


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

        return image, mask, suppixel
    
def get_filenames_from_folder(folder_path):
    return [filename for filename in os.listdir(folder_path) if filename.endswith('.npy')]   


# ------------------------- 训练逻辑 -------------------------

def train_model(model, loss_fn, train_loader, val_loader, optimizer, epochs, save_path):
    best_val_loss = float('inf')
    
    if dist.get_rank() == 0:
        print(f'Train dataset size: {len(train_loader.dataset)}')
        print(f'Train loader batches: {len(train_loader)}')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # 只在主进程中显示训练进度条
        if dist.get_rank() == 0:
            train_loader_tqdm = tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs} [Training]', unit='batch')
        else:
            train_loader_tqdm = train_loader  
        
        for images, masks, superpixels in train_loader_tqdm:
            images, masks, superpixels = images.cuda(), masks.cuda(), superpixels.cuda()
            
            optimizer.zero_grad()
            all_pixel_features = model(images)
            
            # 使用 calc_dc_loss_sp
            loss = calc_dc_loss_sp(masks, all_pixel_features, superpixels) 

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
            if dist.get_rank() == 0:
                train_loader_tqdm.set_postfix(loss=loss.item())
                
        if dist.get_rank() == 0:
            train_loader_tqdm.close()
            
        # 计算总训练损失并同步
        train_loss_tensor = torch.tensor(train_loss).cuda()
        dist.reduce(train_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
        if dist.get_rank() == 0:
            train_loss = train_loss_tensor.item() / len(train_loader.dataset)
        
        # ----------------- 验证阶段 -----------------
        model.eval()
        val_loss = 0.0
        
        if dist.get_rank() == 0:
            val_loader_tqdm = tqdm(val_loader, total=len(val_loader), desc=f'Epoch {epoch+1}/{epochs} [Validation]', unit='batch')
        else:
            val_loader_tqdm = val_loader  

        with torch.no_grad():
            for images, masks, superpixels in val_loader_tqdm:
                images, masks, superpixels = images.cuda(), masks.cuda(), superpixels.cuda()
                all_pixel_features = model(images)

                loss = calc_dc_loss_sp(masks, all_pixel_features, superpixels) 
                val_loss += loss.item() * images.size(0)
                
                if dist.get_rank() == 0:
                    val_loader_tqdm.set_postfix(loss=loss.item())
                    
            if dist.get_rank() == 0:
                val_loader_tqdm.close()
            
            # 计算总验证损失并同步
            val_loss_tensor = torch.tensor(val_loss).cuda()
            dist.reduce(val_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
            if dist.get_rank() == 0:
                val_loss = val_loss_tensor.item() / len(val_loader.dataset)
            
        # 只在主进程打印损失并保存模型
        if dist.get_rank() == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            os.makedirs(save_path, exist_ok=True)
            
            if val_loss < best_val_loss or (epoch + 1) > 10:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_filename = os.path.join(save_path, f'{SAVE_MODEL_PREFIX}_{epoch+1}_loss{val_loss:.4f}_best.pth')
                    print(f"Epoch {epoch+1}: Model saved with lowest Val Loss: {val_loss:.4f}")
                else:
                    model_filename = os.path.join(save_path, f'{SAVE_MODEL_PREFIX}_{epoch+1}_loss{val_loss:.4f}.pth')
                    print(f"Epoch {epoch+1}: Model saved with Val Loss: {val_loss:.4f}")
                
                # 在DDP模式下，推荐保存 model.module.state_dict() 以便后续在单卡上也能方便加载
                torch.save(model.module.state_dict(), model_filename)

        # 清空缓存以防止显存溢出
        torch.cuda.empty_cache()


# ------------------------- 主函数入口 -------------------------

def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(rank % torch.cuda.device_count()) 
        print(f"Initialized process group with rank {rank} and world size {world_size}")
    else:
        print("Environment variables RANK and WORLD_SIZE are not set. This script requires torchrun.")
        exit(1)
    return rank


if __name__ == '__main__':
    # 1. 路径构建
    train_images_dir = f"{BASE_PATH}/fold_{FOLD}/train/{IMAGE_SUBDIR}"
    train_masks_dir = f"{BASE_PATH}/fold_{FOLD}/train/{MASK_SUBDIR}"
    train_superpixel_dir = f"{BASE_PATH}/fold_{FOLD}/train/{SUPERPIXEL_SUBDIR}"

    val_images_dir = f"{BASE_PATH}/fold_{FOLD}/val/{IMAGE_SUBDIR}"
    val_masks_dir = f"{BASE_PATH}/fold_{FOLD}/val/{MASK_SUBDIR}"
    val_superpixel_dir = f"{BASE_PATH}/fold_{FOLD}/val/{SUPERPIXEL_SUBDIR}"
    
    save_model_path = f"{BASE_PATH}/fold_{FOLD}/{SAVE_CKPT_DIR_NAME}"

    # 2. 初始化 DDP
    if MULTI_GPU:
        rank = init_distributed_mode()
    else:
        rank = 0
        print("Running in Single GPU mode is not fully configured, please use torchrun.")
        exit(1)

    # 3. 数据集准备
    train_filenames = get_filenames_from_folder(train_images_dir)
    val_filenames = get_filenames_from_folder(val_images_dir)

    train_dataset = CustomDataset(train_images_dir, train_masks_dir, train_superpixel_dir, train_filenames)
    val_dataset = CustomDataset(val_images_dir, val_masks_dir, val_superpixel_dir, val_filenames)

    # DDP 需要使用 DistributedSampler
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS)
    
    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=NUM_WORKERS)

    # 4. 模型初始化
    model = ResNetUNet().cuda()
    model = DDP(model, device_ids=[rank], output_device=rank)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    loss_fn = CombinedLoss()  # 虽然原代码没显式加上，但保留了该类的定义和初始化

    # 5. 开始训练
    train_model(model, loss_fn, train_loader, val_loader, optimizer, epochs=EPOCHS, save_path=save_model_path)
    
    
#### torchrun --nproc_per_node=2 train_resunet_DDP.py