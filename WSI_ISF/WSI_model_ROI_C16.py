import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from einops import rearrange as _rearrange
from tqdm import tqdm 

from scipy.ndimage.morphology import distance_transform_edt
from skimage.morphology import skeletonize_3d
from skimage.measure import label as label_1
from skimage.measure import regionprops


###########################################################################################################
def generateGuidingSignal(binaryMask):
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
    """
    批量处理GT_mask和pred_mask，计算每个样本的前景和背景骨架信号。
    """
    pred_mask_all = (pred_mask_all > 0.5).float()
    batch_size, _, H, W = pred_mask_all.shape
    output = torch.zeros(batch_size, 1, H, W, device=pred_mask_all.device, dtype=torch.float32)

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
            output[i,:] = generateGuidingSignal(largest_connected) if largest_connected.sum() > 0 else torch.zeros_like(fg, dtype=torch.float32, device=fg.device)
            output[i, output[i,:] != 0] = 1
        else:
            largest_connected = bg_largest
            output[i,:] = generateGuidingSignal(largest_connected) if largest_connected.sum() > 0 else torch.zeros_like(bg, dtype=torch.float32, device=bg.device)
            output[i, output[i,:] != 0] = -1
    return output


def get_largest_connected_component(mask):
    """
    获取mask中最大连通区域
    """
    if mask.sum() > 0:
        labeled_mask = label_1(mask.cpu().numpy(), connectivity=1)
        regions = regionprops(labeled_mask)
        if regions:
            largest_region = max(regions, key=lambda r: r.area)
            largest_component = (labeled_mask == largest_region.label)
            return torch.from_numpy(largest_component).to(mask.device, dtype=torch.float32)
    return torch.zeros_like(mask, device=mask.device)


#############################################################################################################

def rearrange(*args, **kwargs):
    return _rearrange(*args, **kwargs).contiguous()


class CustomDataset(Dataset):
    def __init__(self, h5_dir, signal_dir, mask_dir):
        self.h5_files = sorted([os.path.join(h5_dir, f) for f in os.listdir(h5_dir) if f.endswith('.h5')])
        self.signal_files = sorted([os.path.join(signal_dir, f) for f in os.listdir(signal_dir) if f.endswith('.npy')])
        self.mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.npy')])
        
        if len(self.h5_files) != len(self.signal_files) or len(self.h5_files) != len(self.mask_files):
            raise ValueError(f"File count mismatch: {len(self.h5_files)} h5 files, {len(self.signal_files)} signal files, {len(self.mask_files)} mask files.")

    def load_h5_data(self, path):
        with h5py.File(path, 'r') as f:
            coordinates = f['coords'][:]
            features = f['features'][:]
        return coordinates, features

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, idx):
        coordinates, features = self.load_h5_data(self.h5_files[idx])
        signal_coords = np.load(self.signal_files[idx])
        mask = np.load(self.mask_files[idx])
        return coordinates, features, signal_coords, mask, self.signal_files[idx]


class SimilarityModel(nn.Module):
    def __init__(self, feature_dim, num_ref_modes=5, mapped_dim=1024):
        super(SimilarityModel, self).__init__()
        self.feature_dim = feature_dim
        self.num_ref_modes = num_ref_modes
        self.mapped_dim = mapped_dim
        
        patch_embed = torch.empty(num_ref_modes, mapped_dim)
        patch_embed = nn.init.xavier_normal_(patch_embed)
        self.patch_embed = nn.Parameter(patch_embed)
        
        self.W_q = nn.Linear(mapped_dim, mapped_dim)
        self.W_k = nn.Linear(mapped_dim, mapped_dim)
        self.W_v = nn.Linear(mapped_dim, mapped_dim)
        
        self.classification_head = nn.Conv2d(mapped_dim, 1, kernel_size=1)

    def forward(self, roi_signal, roi_features, roi_mask):
        roi_features_A = self.ref_embed(roi_features, roi_signal)
        batch_size, _, ROI_H, ROI_W = roi_mask.shape
        
        roi_features_A_flat = roi_features_A.reshape(self.mapped_dim, -1).permute(1, 0)
        roi_features_O_flat = roi_features.reshape(self.mapped_dim, -1).permute(1, 0)
        
        Q = self.W_q(roi_features_A_flat)
        K = self.W_k(roi_features_O_flat)
        V = self.W_v(roi_features_O_flat)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.mapped_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.permute(1, 0).view(self.mapped_dim, ROI_H, ROI_W)
        attention_output = attention_output.unsqueeze(0)
        
        segmentation_map = self.classification_head(attention_output) 
        segmentation_map = segmentation_map.float()
        segmentation_map = torch.sigmoid(segmentation_map) 

        return segmentation_map

    def ref_embed(self, x, ref_label):
        ref_label = ref_label.long()
        ref_mask = F.one_hot(ref_label, self.num_ref_modes)
        patch_embed = ref_mask.float() @ self.patch_embed
        patch_embed = rearrange(patch_embed, 'b () h w c -> b c h w')
        
        if patch_embed.size()[-2:] != x.size()[-2:]:
            patch_embed = F.interpolate(patch_embed, x.size()[-2:], mode='bilinear', align_corners=False)
        return x + patch_embed

    def min_max_normalize(self, matrix):
        min_val = torch.min(matrix)
        max_val = torch.max(matrix)
        return (matrix - min_val) / (max_val - min_val + 1e-8)
    
    def top_30(self, out_mask_clone):
        flattened_mask = out_mask_clone.view(-1)
        num_elements = flattened_mask.numel()
        top_30_percent_count = int(0.3 * num_elements)
        _, top_indices = torch.topk(flattened_mask, top_30_percent_count)

        binary_mask = torch.zeros_like(flattened_mask)
        binary_mask[top_indices] = 1
        return binary_mask.view_as(out_mask_clone)


def dice_score(y_true, y_pred, coordinates, a=1., b=1.):
    coords = coordinates.squeeze(0).long().to(y_pred.device)
    y_true = y_true.squeeze(0)
    y_pred = y_pred.squeeze(0)

    selected_preds = y_pred[coords[:, 0], coords[:, 1]]
    selected_targets = y_true[coords[:, 0], coords[:, 1]]

    intersection = torch.sum(selected_targets * selected_preds)
    return (2. * intersection + a) / (torch.sum(selected_targets) + torch.sum(selected_preds) + b)
    
def dice_coeff(y_true, y_pred, a=1., b=1.):
    y_true_flat = y_true.view(-1).float()
    y_pred_flat = y_pred.reshape(-1).float()
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
    return intersection / union

def compute_miou_binary(pred, target):
    return compute_iou(pred, target, 1)

def calculate_accuracy(pred, target):
    TP = ((pred == 1) & (target == 1)).sum().item()
    TN = ((pred == 0) & (target == 0)).sum().item()
    FP = ((pred == 1) & (target == 0)).sum().item()
    FN = ((pred == 0) & (target == 1)).sum().item()
    
    total_pixels = TP + TN + FP + FN
    return (TP + TN) / total_pixels if total_pixels > 0 else 0.1

def cut_roi(coordinates, features, signal_coords, mask):
    batch_size, H, W = mask.shape
    num_coords = coordinates.shape[1]
    num_signal_coords = signal_coords.shape[1]
    
    output_tensor = torch.zeros((batch_size, 1024, H, W), device=mask.device)
    coordinates = coordinates.long()
    
    flat_indices = (coordinates[..., 0] * W + coordinates[..., 1]).view(-1)
    flat_batch_indices = torch.arange(batch_size, device=mask.device).repeat_interleave(num_coords)
    output_tensor_flat = output_tensor.view(batch_size, 1024, -1)
    output_tensor_flat[flat_batch_indices, :, flat_indices] = features.view(-1, 1024)
    output_tensor = output_tensor_flat.view(batch_size, 1024, H, W)
                
    signal = torch.zeros((batch_size, 1, H, W), device=mask.device)
    for i in range(batch_size):
        for j in range(num_signal_coords):
            x, y = signal_coords[i][j]
            if 0 <= x < H and 0 <= y < W:
                signal[i, 0, x, y] = 1            

    signal_coords = signal_coords.long()
    
    x_min_1 = signal_coords[:, :, 0].min(dim=1)[0]
    x_max_1 = signal_coords[:, :, 0].max(dim=1)[0]
    y_min_1 = signal_coords[:, :, 1].min(dim=1)[0]
    y_max_1 = signal_coords[:, :, 1].max(dim=1)[0]
    
    center_x = (x_min_1 + x_max_1) // 2
    center_y = (y_min_1 + y_max_1) // 2
    
    start_y = torch.tensor(max(center_y - 50, 0), dtype=torch.float)
    start_x = torch.tensor(max(center_x - 50, 0), dtype=torch.float)
    end_y = torch.tensor(start_y + 100, dtype=torch.float)
    end_x = torch.tensor(start_x + 100, dtype=torch.float)
    
    if end_y > W:
        end_y = torch.tensor(W, dtype=torch.float)
        start_y = torch.tensor(end_y - 100, dtype=torch.float)
    if end_x > H:
        end_x = torch.tensor(H, dtype=torch.float)
        start_x = torch.tensor(end_x - 100, dtype=torch.float) 
        
    x_min = torch.floor(start_x).to(torch.int)
    x_max = torch.floor(end_x).to(torch.int)
    y_min = torch.floor(start_y).to(torch.int)
    y_max = torch.floor(end_y).to(torch.int)

    roi_signal = signal[:, :, x_min:x_max, y_min:y_max]
    roi_features = output_tensor[:, :, x_min:x_max, y_min:y_max]
    roi_mask = mask[:, x_min:x_max, y_min:y_max]
    
    return roi_signal, roi_features, roi_mask


def preprocess_and_train(path, fold, train_h5_dir, train_signal_dir, train_mask_dir, val_h5_dir, val_signal_dir, val_mask_dir, num_epochs=50, learning_rate=1e-4, batch_size=1, device='cuda:1', feature_dim=1024):
    best_roi_dice = 0
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    train_dataset = CustomDataset(train_h5_dir, train_signal_dir, train_mask_dir)
    val_dataset = CustomDataset(val_h5_dir, val_signal_dir, val_mask_dir)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    model = SimilarityModel(feature_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_dice_scores_roi = []
        train_iou_scores_roi = []
        train_acc_scores_roi = []
        
        train_loader_tqdm = tqdm(train_dataloader, total=len(train_dataloader), desc=f'Epoch {epoch+1}/{num_epochs} [Training]', unit='batch')

        for coordinates, features, signal_coords, mask, signal_file in train_loader_tqdm:
            coordinates, features, signal_coords, mask = coordinates.to(device), features.to(device), signal_coords.to(device), mask.to(device)
            roi_signal, roi_features, roi_mask = cut_roi(coordinates, features, signal_coords, mask)
            roi_signal[roi_signal == 1] = 2  
            roi_signal[roi_signal == 0] = 4 
            
            roi_mask = roi_mask.unsqueeze(1).float()            
            optimizer.zero_grad()

            with autocast(): 
                roi_segmap_1 = model(roi_signal, roi_features, roi_mask)
                roi_seg_1 =  (roi_segmap_1 >= 0.5).float()

                signal_2 = processMasks(roi_seg_1, roi_mask)
                roi_seg_1[signal_2 == 1] = 2
                roi_seg_1[signal_2 == -1] = 3
                roi_seg_1[roi_signal == 2] = 2  
                
                roi_segmap_2 = model(roi_seg_1, roi_features, roi_mask)
                roi_seg_2 =  (roi_segmap_2 >= 0.5).float()
                
                signal_3 = processMasks(roi_seg_2, roi_mask)
                roi_seg_2[signal_3 == 1] = 2
                roi_seg_2[signal_3 == -1] = 3
                roi_seg_2[signal_2 == 1] = 2     
                roi_seg_2[signal_2 == -1] = 3    
                roi_seg_2[roi_signal == 2] = 2   
                
                condition1 = (roi_seg_2 == 0) & (roi_seg_1 == 1)
                roi_seg_2[condition1] = 4

                condition2 = (roi_seg_2 == 1) & (roi_seg_1 == 0)
                roi_seg_2[condition2] = 4
                
                roi_segmap_3 = model(roi_seg_2, roi_features, roi_mask)
                roi_seg_3 =  (roi_segmap_3 >= 0.5).float()
                
                loss = dice_loss(roi_segmap_1, roi_mask) + dice_loss(roi_segmap_2, roi_mask) + dice_loss(roi_segmap_3, roi_mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * mask.size(0)
            
            train_dice_score_roi = dice_coeff(roi_seg_3, roi_mask).item()
            train_dice_scores_roi.append(train_dice_score_roi)

            train_acc_roi = calculate_accuracy(roi_seg_3, roi_mask)
            train_acc_scores_roi.append(train_acc_roi)
            
            roi_seg_3 = roi_seg_3.detach().cpu().numpy()
            preds = (roi_seg_3).astype(int)
            
            roi_mask = roi_mask.cpu().numpy()
            for pred, r_mask in zip(preds, roi_mask):
                miou = compute_miou_binary(pred, r_mask)
                if not np.isnan(miou):
                    train_iou_scores_roi.append(miou)
            
        train_mean_iou_roi = np.mean(train_iou_scores_roi)
        train_mean_dice_roi = np.mean(train_dice_scores_roi)
        train_mean_acc_roi = np.mean(train_acc_scores_roi)
        
        model.eval()
        val_loss = 0
        val_dice_scores_roi = []
        val_iou_scores_roi = []
        val_acc_scores_roi = []
        
        with torch.no_grad():
            val_loader_tqdm = tqdm(val_dataloader, total=len(val_dataloader), desc=f'Epoch {epoch+1}/{num_epochs} [val]', unit='batch')

            for coordinates, features, signal_coords, mask, signal_file in val_loader_tqdm:
                coordinates, features, signal_coords, mask = coordinates.to(device), features.to(device), signal_coords.to(device), mask.to(device)
                
                roi_signal, roi_features, roi_mask = cut_roi(coordinates, features, signal_coords, mask)
                roi_signal[roi_signal == 1] = 2
                roi_signal[roi_signal == 0] = 4
                roi_mask = roi_mask.unsqueeze(1).float()
                
                with autocast():
                    roi_segmap_1 = model(roi_signal, roi_features, roi_mask)
                    roi_seg_1 =  (roi_segmap_1 >= 0.5).float()
                    
                    signal_2 = processMasks(roi_seg_1, roi_mask)
                    roi_seg_1[signal_2 == 1] = 2
                    roi_seg_1[signal_2 == -1] = 3
                    roi_seg_1[roi_signal == 2] = 2  
                    
                    roi_segmap_2 = model(roi_seg_1, roi_features, roi_mask)
                    roi_seg_2 =  (roi_segmap_2 >= 0.5).float()
                    
                    signal_3 = processMasks(roi_seg_2, roi_mask)
                    roi_seg_2[signal_3 == 1] = 2
                    roi_seg_2[signal_3 == -1] = 3
                    roi_seg_2[signal_2 == 1] = 2     
                    roi_seg_2[signal_2 == -1] = 3    
                    roi_seg_2[roi_signal == 2] = 2   
                    
                    condition1 = (roi_seg_2 == 0) & (roi_seg_1 == 1)
                    roi_seg_2[condition1] = 4

                    condition2 = (roi_seg_2 == 1) & (roi_seg_1 == 0)
                    roi_seg_2[condition2] = 4
                    
                    roi_segmap_3 = model(roi_seg_2, roi_features, roi_mask)
                    roi_seg_3 =  (roi_segmap_3 >= 0.5).float()
                    
                    loss = dice_loss(roi_segmap_1, roi_mask) + dice_loss(roi_segmap_2, roi_mask) + dice_loss(roi_segmap_3, roi_mask)
                
                val_loss += loss.item() * roi_mask.size(0)
                
                val_dice_score_roi = dice_coeff(roi_seg_3, roi_mask).item()
                val_dice_scores_roi.append(val_dice_score_roi)

                val_acc_roi = calculate_accuracy(roi_seg_3, roi_mask)
                val_acc_scores_roi.append(val_acc_roi)
                
                roi_seg_3 = roi_seg_3.detach().cpu().numpy()
                preds = (roi_seg_3).astype(int)
                
                roi_mask = roi_mask.cpu().numpy()
                for pred, r_mask in zip(preds, roi_mask):
                    miou = compute_miou_binary(pred, r_mask)
                    if not np.isnan(miou):
                        val_iou_scores_roi.append(miou)
                
            val_mean_iou_roi = np.mean(val_iou_scores_roi)
            val_mean_dice_roi = np.mean(val_dice_scores_roi)
            val_mean_acc_roi = np.mean(val_acc_scores_roi)
        
        if val_mean_dice_roi > best_roi_dice:
            best_roi_dice = val_mean_dice_roi
            torch.save(model.state_dict(), f'{path}/C16_best_model_fold_{fold}.pth')
            print(f'Model checkpoint saved at epoch {epoch + 1}')
            
        print(f'Epoch [{epoch + 1}/{num_epochs}], train_ROI_Dice: {train_mean_dice_roi:.4f} ,train_ROI Loss: {train_loss / len(train_dataloader):.4f}, train all Dice_roi: {train_mean_dice_roi:.4f}, train all Acc_roi: {train_mean_acc_roi:.4f}, train all mIOU_roi: {train_mean_iou_roi:.4f}')
        print(f'Epoch [{epoch + 1}/{num_epochs}], val_ROI_Dice: {val_mean_dice_roi:.4f} ,val_ROI Loss: {val_loss / len(val_dataloader):.4f}, val all Dice_roi: {val_mean_dice_roi:.4f}, val all Acc_roi: {val_mean_acc_roi:.4f}, val all mIOU_roi: {val_mean_iou_roi:.4f}')


# ==============================================================================
# 用户配置区 (所有需要你修改的路径、超参数都集中在这里)
# ==============================================================================
if __name__ == "__main__":
    
    # --- 1. 路径与设备配置 ---
    BASE_PATH = "/data_nas2/gjs/Camelyon16/UNI_5_fold"  # 数据集根目录
    FOLD_LIST = [4]                                     # 需要运行的折数列表
    DEVICE = "cuda:3"                                   # 运行设备 (例如 'cuda:0', 'cuda:1' 等)

    # --- 2. 模型训练超参数 ---
    NUM_EPOCHS = 50                                     # 训练轮数
    LEARNING_RATE = 1e-4                                # 学习率
    BATCH_SIZE = 1                                      # 批次大小
    FEATURE_DIM = 1024                                  # 特征维度

    # ==============================================================================
    # 运行逻辑 (无需修改)
    # ==============================================================================
    for fold in FOLD_LIST:
        print(f"========== 开始训练 Fold: {fold} ==========")
        
        # 自动拼接训练集和验证集路径
        train_h5_dir = f'{BASE_PATH}/fold_{fold}/train/feature_h5'
        train_signal_dir = f'{BASE_PATH}/fold_{fold}/train/signals_max_XY'
        train_mask_dir = f'{BASE_PATH}/fold_{fold}/train/mask'

        val_h5_dir = f'{BASE_PATH}/fold_{fold}/test/feature_h5'
        val_signal_dir = f'{BASE_PATH}/fold_{fold}/test/signals_max_XY'
        val_mask_dir = f'{BASE_PATH}/fold_{fold}/test/mask'

        # 调用训练函数
        preprocess_and_train(
            path=BASE_PATH, 
            fold=fold, 
            train_h5_dir=train_h5_dir, 
            train_signal_dir=train_signal_dir, 
            train_mask_dir=train_mask_dir, 
            val_h5_dir=val_h5_dir, 
            val_signal_dir=val_signal_dir, 
            val_mask_dir=val_mask_dir,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            feature_dim=FEATURE_DIM
        )