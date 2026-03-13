import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 假设该模块在你的本地环境中存在
from efficientunet import *
from UNet import *

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# ==============================================================================
# 用户配置区 (User Configuration) - 请在此处修改所有超参数
# ==============================================================================
class Config:
    # 1. 基础环境与设备配置
    MULTI_GPU = False
    
    # 2. 核心训练超参数
    LR = 4e-4
    EPOCHS = 50
    TEMPERATURE = 0.3
    
    # 3. 数据集与折数配置
    BASE_PATH = "/data_nas2/gjs/ISF_pixel_level_data/Gastric_new"
    FOLD = 5
    
    # 4. DataLoader 参数
    TRAIN_BATCH_SIZE = 20 if MULTI_GPU else 16
    VAL_BATCH_SIZE = 20 if MULTI_GPU else 16
    NUM_WORKERS = 4

# ==============================================================================
# 分布式训练初始化
# ==============================================================================
def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(rank % torch.cuda.device_count())  
        print(f"Initialized process group with rank {rank} and world size {world_size}")
    else:
        print("Environment variables RANK and WORLD_SIZE are not set.")
        rank = 0
    return rank

# ==============================================================================
# 核心对比学习损失函数 (Contrastive Learning Loss)
# ==============================================================================
def calc_dc_loss_sp(masks, all_perpixel_features, superpixels):
    device = all_perpixel_features.device 
    b, c, h, w = all_perpixel_features.shape
    
    loss_dc = 0
    temperature = Config.TEMPERATURE

    # 嵌套函数依赖外部的 bi 变量，保留原逻辑以确保计算一致性
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
            cos_all_ff = torch.sum(cos_smi_sp_ff)    
            cos_avg_ff = cos_all_ff / len(fg_sps_labels_list)
            
            cos_smi_sp_fb = affinity_matrix_fb[i, :]
            cos_all_fb = torch.sum(cos_smi_sp_fb)    
            cos_avg_fb = cos_all_fb / len(bg_sps_labels_list)
            
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
            
            # 避免除以 0 的情况
            pixels_norm = pixels_feature_fg.norm(dim=1).clone()
            pixels_norm[pixels_norm == 0] = 1e-8
            pixels_feature_fg_norm = pixels_feature_fg / pixels_norm[:, None]
            
            # 计算像素与当前前景超像素块特征的相似性
            sp_feature_fg_norm = fg_sps_feature_list1_norm[i:i+1]  
            affinity_matrix_pixels = torch.mm(sp_feature_fg_norm, pixels_feature_fg_norm.t())
            affinity_matrix_pixels = torch.exp(affinity_matrix_pixels / temperature)
            e_sim_fg = affinity_matrix_pixels.mean()
            e_dis_fg = affinity_matrix_fg_sum[i]
            
            # 计算总损失
            fg_total_loss += -torch.log(e_sim_fg / (e_sim_fg + e_dis_fg + 1e-8))
            
        total_loss = fg_total_loss / len(fg_sps_labels_list) 
        return total_loss
        
    for bi in range(b): 
        num_batch = b
        target_sps = superpixels[bi, 0, ...] 
        sps_labels_list = torch.unique(target_sps)

        sps_feature_list = get_sps_features(all_perpixel_features, superpixels, sps_labels_list).to(device) 
        
        loss_img = 0
        
        # 将 masks 和 superpixels 展平
        mask_flat = masks[bi].view(-1)  
        superpixels_flat = superpixels[bi].view(-1)  

        # 获取除背景外的所有类别，假设背景类为0
        unique_labels = mask_flat.unique()
        fg_labels = unique_labels[unique_labels > 0]  
        fg_labels_num = len(fg_labels)

        # 遍历每个前景类别，依次将一个类别作为前景，其余类别作为背景
        for fg_label in fg_labels:
            fg_sps_labels_list = []
            bg_sps_labels_list = []

            # 遍历超像素块标签列表
            for label in sps_labels_list:
                mask_for_superpixel = mask_flat[superpixels_flat == label]

                num_fg_pixels = torch.sum(mask_for_superpixel == fg_label)
                num_bg_pixels = torch.sum(mask_for_superpixel != fg_label)

                if num_fg_pixels > num_bg_pixels:
                    fg_sps_labels_list.append(label)
                else:
                    bg_sps_labels_list.append(label)

            fg_sps_labels_list = torch.tensor(fg_sps_labels_list).long().to(device)
            bg_sps_labels_list = torch.tensor(bg_sps_labels_list).long().to(device)
            
            fg_sps_feature_list = [sps_feature_list[label] for label in fg_sps_labels_list]   
            bg_sps_feature_list = [sps_feature_list[label] for label in bg_sps_labels_list]   
                    
            if len(fg_sps_feature_list) == 0 or len(bg_sps_feature_list) == 0:
                fg_labels_num -= 1
                continue    
                        
            fg_sps_feature_list1 = torch.stack(fg_sps_feature_list)
            bg_sps_feature_list1 = torch.stack(bg_sps_feature_list)

            # 归一化超像素特征
            fg_sps_feature_list1_norm = fg_sps_feature_list1 / fg_sps_feature_list1.norm(dim=1)[:, None]  
            bg_sps_feature_list1_norm = bg_sps_feature_list1 / bg_sps_feature_list1.norm(dim=1)[:, None]  
            
            affinity_matrix_ff = torch.mm(fg_sps_feature_list1_norm, fg_sps_feature_list1_norm.t())
            affinity_matrix_ff = torch.exp(affinity_matrix_ff / temperature)
            
            affinity_matrix_fb = torch.mm(fg_sps_feature_list1_norm, bg_sps_feature_list1_norm.t())
            affinity_matrix_fb = torch.exp(affinity_matrix_fb / temperature)
            
            loss_sp1 = get_sp_contrast_loss(fg_sps_labels_list, fg_sps_labels_list, affinity_matrix_ff, affinity_matrix_fb)
                            
            # 让前景超像素块中的像素特征与所在超像素块的特征更相似，与背景超像素块的特征远离
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
        loss_dc = 1e-8

    return loss_dc

# ==============================================================================
# 数据集定义 (Dataset)
# ==============================================================================
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

# ==============================================================================
# 训练与验证循环 (Training & Validation)
# ==============================================================================
def train_model(model, train_loader, val_loader, optimizer, epochs=50):
    best_val_loss = 10.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    epoch_pbar = tqdm(range(epochs), desc="Overall Training Progress", unit="epoch")

    # 预先创建保存权重的目录
    save_dir = f'{Config.BASE_PATH}/fold_{Config.FOLD}/efficientUnet'
    os.makedirs(save_dir, exist_ok=True)

    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0

        train_batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False, unit="batch")
        
        for images, masks, superpixels in train_batch_pbar:
            images, masks, superpixels = images.to(device), masks.to(device), superpixels.to(device)
            optimizer.zero_grad()

            all_pixel_features = model(images)
            loss = calc_dc_loss_sp(masks, all_pixel_features, superpixels)

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_batch_pbar.set_postfix(batch_loss=f"{loss.item():.4f}")
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            val_batch_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]", leave=False, unit="batch")
            
            for images, masks, superpixels in val_batch_pbar:
                images, masks, superpixels = images.to(device), masks.to(device), superpixels.to(device)
                all_pixel_features = model(images)

                loss = calc_dc_loss_sp(masks, all_pixel_features, superpixels)
                val_loss += loss.item() * images.size(0)
                
                val_batch_pbar.set_postfix(batch_loss=f"{loss.item():.4f}")
            
            val_loss /= len(val_loader.dataset)
            epoch_pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")
            print(f'Epoch {epoch+1}/{epochs}, Train Loss (CombinedLoss): {train_loss:.4f}, Val Loss : {val_loss:.4f}')
            
            # 保存最优模型和最新一个epoch的模型
            if val_loss < best_val_loss or (epoch + 1) > 10:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_filename = f'{save_dir}/efficientUnet_{epoch+1}_loss{val_loss:.4f}_best.pth'
                    print(f"Epoch {epoch+1}: Model saved with lowest Val Loss: {val_loss:.4f}")
                else:
                    model_filename = f'{save_dir}/efficientUnet_{epoch+1}_loss{val_loss:.4f}.pth'
                    print(f"Epoch {epoch+1}: Model saved with Val Loss: {val_loss:.4f}")
                
                torch.save(model.state_dict(), model_filename)

        # 清空缓存以防止显存溢出
        torch.cuda.empty_cache()

# ==============================================================================
# 主执行入口 (Main Execution)
# ==============================================================================
if __name__ == "__main__":
    # 构建数据路径
    train_images_dir = f"{Config.BASE_PATH}/fold_{Config.FOLD}/train/Contrast_learning/image_npy"
    train_masks_dir = f"{Config.BASE_PATH}/fold_{Config.FOLD}/train/Contrast_learning/mask_npy"
    train_superpixel_dir = f'{Config.BASE_PATH}/fold_{Config.FOLD}/train/Contrast_learning/image_SLIC_500'

    val_images_dir = f"{Config.BASE_PATH}/fold_{Config.FOLD}/val/Contrast_learning/image_npy"
    val_masks_dir = f"{Config.BASE_PATH}/fold_{Config.FOLD}/val/Contrast_learning/mask_npy"
    val_superpixel_dir = f'{Config.BASE_PATH}/fold_{Config.FOLD}/val/Contrast_learning/image_SLIC_500'

    # 获取训练集和验证集的文件名
    train_filenames = get_filenames_from_folder(train_images_dir)
    val_filenames = get_filenames_from_folder(val_images_dir)

    # 创建自定义数据集类的实例
    train_dataset = CustomDataset(train_images_dir, train_masks_dir, train_superpixel_dir, train_filenames)
    val_dataset = CustomDataset(val_images_dir, val_masks_dir, val_superpixel_dir, val_filenames)

    # 创建模型实例
    model = get_efficientunet_b0(out_channels=1, concat_input=True, pretrained=False).cuda()

    if Config.MULTI_GPU:
        rank = init_distributed_mode()
        model = DDP(model, device_ids=[rank], output_device=rank)
        
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=Config.TRAIN_BATCH_SIZE, sampler=train_sampler, num_workers=Config.NUM_WORKERS)
        
        val_sampler = DistributedSampler(val_dataset)
        val_loader = DataLoader(val_dataset, batch_size=Config.VAL_BATCH_SIZE, sampler=val_sampler, num_workers=Config.NUM_WORKERS)
    else:
        train_loader = DataLoader(train_dataset, batch_size=Config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=Config.VAL_BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.LR)

    # 开始训练
    train_model(model, train_loader, val_loader, optimizer, epochs=Config.EPOCHS)