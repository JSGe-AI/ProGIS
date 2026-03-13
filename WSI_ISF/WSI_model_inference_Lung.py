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

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

###########################################################################################################
def generateGuidingSignal(binaryMask):
    binaryMask = binaryMask.to(torch.uint8)
    device = binaryMask.device
    
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
        skel = torch.tensor(skel, dtype=torch.float32, device=device)
    else:
        skel = torch.zeros_like(binaryMask, dtype=torch.float32, device=device)

    return skel


def processMasks(pred_mask_all, GT_mask_all):
    pred_mask_all = (pred_mask_all > 0.5).float()
    batch_size, _, H, W = GT_mask_all.shape
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
            indices = torch.nonzero(largest_connected)
            center = indices.float().mean(dim=0)
        else:
            largest_connected = bg_largest
            output[i,:] = generateGuidingSignal(largest_connected) if largest_connected.sum() > 0 else torch.zeros_like(bg, dtype=torch.float32, device=bg.device)
            output[i, output[i,:] != 0] = -1
            indices = torch.nonzero(largest_connected)
            center = indices.float().mean(dim=0)
    return output, center


def get_largest_connected_component(mask):
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
    def __init__(self, feature_dim, num_ref_modes=5):
        super(SimilarityModel, self).__init__()
        self.feature_dim = feature_dim
        self.num_ref_modes = num_ref_modes
        
        patch_embed = torch.empty(num_ref_modes, feature_dim)
        patch_embed = nn.init.xavier_normal_(patch_embed)
        self.patch_embed = nn.Parameter(patch_embed)
        
        self.W_q = nn.Linear(feature_dim, feature_dim)
        self.W_k = nn.Linear(feature_dim, feature_dim)
        self.W_v = nn.Linear(feature_dim, feature_dim)
        
        self.classification_head = nn.Conv2d(feature_dim, 1, kernel_size=1)

    def forward(self, roi_signal, roi_features, roi_mask):
        roi_features_A = self.ref_embed(roi_features, roi_signal)
        batch_size, _, ROI_H, ROI_W = roi_mask.shape
        
        roi_features_A_flat = roi_features_A.reshape(self.feature_dim, -1).permute(1, 0)
        roi_features_O_flat = roi_features.reshape(self.feature_dim, -1).permute(1, 0)
        
        Q = self.W_q(roi_features_A_flat)
        K = self.W_k(roi_features_O_flat)
        V = self.W_v(roi_features_O_flat)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.feature_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.permute(1, 0).view(self.feature_dim, ROI_H, ROI_W)
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


class Sim_Model(nn.Module):
    def __init__(self, feature_dim):
        super(Sim_Model, self).__init__()
        self.roi_segmodel = SimilarityModel(feature_dim)
        self.feature_dim = feature_dim

    def forward(self, feature_tensor, signal_coords, signal_file, mask, threod):
        signal, roi_signal, roi_features, roi_mask = self.cut_roi(feature_tensor, signal_coords, mask)
        roi_mask = roi_mask.unsqueeze(1) 
        roi_signal[roi_signal == 0] = 4 
        roi_segmap = self.roi_segmodel(roi_signal, roi_features, roi_mask)
        roi_seg_clone =  (roi_segmap >= 0.5).float()

        proto_feature_fg = torch.sum(roi_features * roi_seg_clone, dim=(2, 3)) \
                         / (roi_seg_clone.sum(dim=(2, 3)) + 1e-5)
                 
        feature_tensor_norm = F.normalize(feature_tensor.view(1, self.feature_dim, -1), p=2, dim=1) 
        proto_feature_fg_norm = F.normalize(proto_feature_fg, p=2, dim=1) 

        cosine_sim = (torch.matmul(feature_tensor_norm.permute(0, 2, 1), proto_feature_fg_norm.T))**2 

        cosine_sim_min = cosine_sim.min()
        cosine_sim_max = cosine_sim.max()
        similarity_fg = (cosine_sim - cosine_sim_min) / (cosine_sim_max - cosine_sim_min + 1e-6) 

        similarity_fg = similarity_fg.view(1, 1, feature_tensor.shape[2], feature_tensor.shape[3])
        segmentation_map =  similarity_fg.clone()
        
        similarity_fg[similarity_fg <= threod] = 0
        similarity_fg[similarity_fg > threod] = 1

        return similarity_fg, roi_seg_clone, roi_mask, signal, segmentation_map 

    def cut_roi(self, feature_tensor, signal_coords, mask):
        batch_size, H, W = mask.shape
        num_signal_coords = signal_coords.shape[1]
           
        signal = torch.zeros((batch_size, 1, H, W), device=mask.device)
        for i in range(batch_size):
            for j in range(num_signal_coords):
                x, y = signal_coords[i][j]
                if 0 <= x < H and 0 <= y < W:
                    signal[i, 0, x, y] = 2            

        signal_coords = signal_coords.long()
        if signal_coords.size(1) == 0:
            center_x = int(H // 2)
            center_y = int(W // 2)
        else:
            x_min_1 = signal_coords[:, :, 0].min(dim=1)[0]
            x_max_1 = signal_coords[:, :, 0].max(dim=1)[0]
            y_min_1 = signal_coords[:, :, 1].min(dim=1)[0]
            y_max_1 = signal_coords[:, :, 1].max(dim=1)[0]
    
            center_x = int((x_min_1 + x_max_1) // 2)
            center_y = int((y_min_1 + y_max_1) // 2)
        
        size = 100
        start_y = torch.tensor(max(center_y - size/2, 0), dtype=torch.float)
        start_x = torch.tensor(max(center_x - size/2, 0), dtype=torch.float)
        end_y = torch.tensor(start_y + size, dtype=torch.float)
        end_x = torch.tensor(start_x + size, dtype=torch.float)

        if end_y - start_y < size:
            start_y = torch.tensor((max(end_y - size, 0)), dtype=torch.float)
            end_y = torch.tensor(min(start_y + size, W), dtype=torch.float)
        if end_x - start_x < size:
            start_x = torch.tensor((max(end_x - size, 0)), dtype=torch.float)
            end_x = torch.tensor(min(start_x + size, H), dtype=torch.float)
        
        if (end_x-start_x ) != size or (end_y-start_y)!=size:
            start_x, end_x = torch.tensor(0, dtype=torch.float), torch.tensor(H, dtype=torch.float)
            start_y, end_y = torch.tensor(0, dtype=torch.float), torch.tensor(W, dtype=torch.float)

        x_min = torch.floor(start_x).to(torch.int)
        x_max = torch.floor(end_x).to(torch.int)
        y_min = torch.floor(start_y).to(torch.int)
        y_max = torch.floor(end_y).to(torch.int)

        roi_signal = signal[:, :, x_min:x_max, y_min:y_max]
        roi_features = feature_tensor[:, :, x_min:x_max, y_min:y_max]
        roi_mask = mask[:, x_min:x_max, y_min:y_max]
        
        return signal, roi_signal, roi_features, roi_mask
    

def dice_score(y_true, y_pred, coordinates, a=1., b=1.):
    coords = coordinates.squeeze(0).long().to(y_pred.device)
    y_true = y_true.squeeze(0).squeeze(0)
    y_pred = y_pred.squeeze(0).squeeze(0)

    selected_preds = y_pred[coords[:, 0], coords[:, 1]]
    selected_targets = y_true[coords[:, 0], coords[:, 1]]

    intersection = torch.sum(selected_targets * selected_preds)
    return (2. * intersection + a) / (torch.sum(selected_targets) + torch.sum(selected_preds) + b)
    

def dice_coeff(y_true, y_pred, a=1., b=1.):
    y_true_flat = y_true.reshape(-1).float()
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
    else:
        return intersection / union


def compute_miou_binary(pred, target):
    return compute_iou(pred, target, 1)


def calculate_accuracy(pred, target, coordinates):
    coords = coordinates.squeeze(0).to(pred.device)
    target = target.squeeze(0).squeeze(0)
    pred = pred.squeeze(0).squeeze(0)
    
    selected_preds = (pred[coords[:, 0], coords[:, 1]]).int()
    selected_targets = (target[coords[:, 0], coords[:, 1]]).int()
    
    TP = ((selected_preds == 1) & (selected_targets == 1)).sum().item()
    TN = ((selected_preds == 0) & (selected_targets == 0)).sum().item()
    FP = ((selected_preds == 1) & (selected_targets == 0)).sum().item()
    FN = ((selected_preds == 0) & (selected_targets == 1)).sum().item()
    
    total_pixels = TP + TN + FP + FN
    acc = (TP + TN) / total_pixels if total_pixels > 0 else 0
    return acc


def calculate_balanced_accuracy(pred, target, coordinates):
    coords = coordinates.squeeze(0).to(pred.device)
    target = target.squeeze(0).squeeze(0)
    pred = pred.squeeze(0).squeeze(0)
    
    selected_preds = (pred[coords[:, 0], coords[:, 1]]).int()
    selected_targets = (target[coords[:, 0], coords[:, 1]]).int()
    
    TP = ((selected_preds == 1) & (selected_targets == 1)).sum().item()
    TN = ((selected_preds == 0) & (selected_targets == 0)).sum().item()
    FP = ((selected_preds == 1) & (selected_targets == 0)).sum().item()
    FN = ((selected_preds == 0) & (selected_targets == 1)).sum().item()
    
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0 
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0 
    balanced_acc = (sensitivity + specificity) / 2

    return balanced_acc


def preprocess_and_inference(fold, ckpt, threod, val_h5_dir, val_signal_dir, val_mask_dir, num_epochs=1, learning_rate=1e-4, batch_size=1, device='cuda:1', feature_dim = 512, inter_steps=20):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    val_dataset = CustomDataset(val_h5_dir, val_signal_dir, val_mask_dir)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Sim_Model(feature_dim).to(device)
    model.roi_segmodel.load_state_dict(torch.load(ckpt))
    
    for epoch in range(num_epochs):
        val_dice_scores = []
        val_dice_scores_roi = []
        val_iou_scores = []
        val_acc_scores = []
        
        with torch.no_grad():
            for coordinates, features, signal_coords, mask, signal_file in val_dataloader:
                coordinates, features, signal_coords, mask = coordinates.to(device), features.to(device), signal_coords.to(device), mask.to(device)
                coordinates = coordinates[:, :, [1, 0]]
                
                batch_size, H, W = mask.shape
                num_coords = coordinates.shape[1]

                feature_tensor = torch.zeros((batch_size, feature_dim, H, W), device=mask.device)
                coordinates = coordinates.long()
                
                flat_indices = (coordinates[..., 0] * W + coordinates[..., 1]).view(-1)
                flat_batch_indices = torch.arange(batch_size, device=mask.device).repeat_interleave(num_coords)
                output_tensor_flat = feature_tensor.view(batch_size, feature_dim, -1)
                output_tensor_flat[flat_batch_indices, :, flat_indices] = features.view(-1, feature_dim)
                feature_tensor = output_tensor_flat.view(batch_size, feature_dim, H, W)
                
                with autocast():
                    out_mask, roi_seg, roi_mask_1, signal, segmentation_map = model(feature_tensor, signal_coords, signal_file, mask, threod)
                    
                    mask = mask.unsqueeze(1)
                    previous_mask = out_mask.clone()
                    out_mask_list = [previous_mask]
                    count = 0
                        
                    signal_2, centers = processMasks(out_mask, mask)
                    signal_2[signal == 2] = 1
                    out_mask[signal_2 == 1] = 2
                    out_mask[signal_2 == -1] = 3
                    
                    while count < inter_steps:
                        if count > 0:
                            signal_3, centers = processMasks(out_mask, mask)
                            signal_3[signal_2 == 1] = 1
                            signal_3[signal_2 == -1] = -1
                            
                            previous_mask_1 = out_mask.clone()
                            out_mask[signal_3 == 1] = 2
                            out_mask[signal_3 == -1] = 3
                            
                            condition1 = (out_mask == 0) & (previous_mask == 1)
                            condition2 = (out_mask == 1) & (previous_mask == 0)
                            
                            previous_mask = previous_mask_1
                            signal_2 = signal_3.clone()
                            
                            out_mask[condition1] = 4
                            out_mask[condition2] = 4
                        
                        batch_size, _, H, W = mask.shape
                        size = 100
                        half_size = int(size//2)
    
                        start_x = max(int(centers[0]) - half_size, 0)
                        start_y = max(int(centers[1]) - half_size, 0)
                        end_x = min(start_x + size, H)
                        end_y = min(start_y + size, W)

                        if end_y - start_y < size:
                            start_y = int(max(end_y - size, 0))
                            end_y = min(start_y + size, W)
                        if end_x - start_x < size:
                            start_x = int(max(end_x - size, 0))
                            end_x = min(start_x + size, H)
                            
                        start_x, end_x = int(start_x), int(end_x)
                        start_y, end_y = int(start_y), int(end_y)
                        
                        if (end_x-start_x ) != size or (end_y-start_y)!=size:
                            start_x, end_x = 0, min(100, H)
                            start_y, end_y = 0, min(100, W)

                        roi_signal = out_mask[:, :, start_x: end_x, start_y: end_y]
                        roi_features = feature_tensor[:, :, start_x: end_x, start_y: end_y]
                        roi_mask = mask[:, :, start_x: end_x, start_y: end_y]
                        
                        roi_segmap = model.roi_segmodel(roi_signal, roi_features, roi_mask)
                        fg_sigmoid_output = roi_segmap.clone() 
                        
                        segmentation_map[:, :, start_x: end_x, start_y: end_y] =  fg_sigmoid_output
                        
                        mask_greater_than_0_6 = fg_sigmoid_output > 0.5
                        mask_less_equal_0_6 = fg_sigmoid_output <= 0.5

                        fg_sigmoid_output[mask_greater_than_0_6] = 1
                        fg_sigmoid_output[mask_less_equal_0_6] = 0
                        
                        out_mask[:, :,  start_x: end_x, start_y: end_y] = fg_sigmoid_output
                        out_mask[out_mask == 2] = 1
                        out_mask[out_mask == 3] = 0
                        out_mask[out_mask == 4] = previous_mask[out_mask == 4]
                        
                        out_mask_clone = out_mask.clone()
                        out_mask_list.append(out_mask_clone)
                                
                        count += 1 
                    
                val_dice_score_roi = dice_coeff(roi_seg, roi_mask_1).item()
                val_dice_scores_roi.append(val_dice_score_roi)

                val_dice_score = dice_score(out_mask, mask, coordinates).item()
                val_dice_scores.append(val_dice_score)

                out_mask = out_mask.squeeze(0)
                mask = mask.squeeze(0)
                val_acc = calculate_balanced_accuracy(out_mask, mask, coordinates)
                val_acc_scores.append(val_acc)
                
                out_mask = out_mask.detach().cpu().numpy()
                preds = (out_mask >= 0.5).astype(int)
                
                mask = mask.cpu().numpy()
                for pred, r_mask in zip(preds, mask):
                    miou = compute_miou_binary(pred, r_mask)
                    if not np.isnan(miou):
                        val_iou_scores.append(miou)
                        
                torch.cuda.empty_cache()
                
            val_mean_iou = np.mean(val_iou_scores)
            val_mean_dice = np.mean(val_dice_scores)
            val_mean_dice_roi = np.mean(val_dice_scores_roi)
            val_mean_acc = np.mean(val_acc_scores)
            
        print("threod:", threod)
        print(ckpt)   
        print(f'Epoch [{epoch + 1}/{num_epochs}], val_ROI_Dice: {val_mean_dice_roi:.4f} ,val Dice: {val_mean_dice:.4f}, val balanced Acc: {val_mean_acc:.4f}, val mIOU: {val_mean_iou:.4f}')


# ==========================================================================================
# 用户配置区 (所有需要你修改的路径、超参数、设备配置都集中在这里)
# ==========================================================================================
if __name__ == "__main__":
    
    # ---------------- 1. 基础路径与设备配置 ----------------
    BASE_PATH = "/data_nas2/gjs/Lung/Conch_5_fold"       # 数据集总路径
    FOLD_START = 2                                       # 起始 Fold (包含)
    FOLD_END = 3                                         # 结束 Fold (不包含，比如 range(2,3) 实际只跑 fold 2)
    DEVICE = "cuda:2"                                    # 验证使用的 GPU 设备

    # ---------------- 2. 验证超参数配置 ----------------
    THRESHOLD = 0.999                                    # 分割阈值
    FEATURE_DIM = 512                                    # 模型特征维度
    NUM_EPOCHS = 1                                       # 验证轮次
    BATCH_SIZE = 1                                       # 批大小
    LEARNING_RATE = 1e-4                                 # 学习率 (由于是推理评估过程，这里仅作优化器初始化)
    Inter_steps_num = 20                                 # 交互次数


    # 模型权重路径模板（请保留 {i} 供循环自动填入 Fold 号）
    CKPT_PATH_TEMPLATE = "WSI_ISF/conch_5_fold_ckpt/Lung/Lung_best_model_fold_{i}.pth"


    # ==========================================================================================
    # 主运行逻辑 (通常无需修改)
    # ==========================================================================================
    for i in range(FOLD_START, FOLD_END):
        print(f"\n{'='*20} 开始评估 Fold: {i} {'='*20}")
        print(f"数据路径: {BASE_PATH}")
        
        # 自动拼接该折的验证集路径
        val_h5_dir = f'{BASE_PATH}/fold_{i}/test/feature_h5'
        val_signal_dir = f'{BASE_PATH}/fold_{i}/test/signals_max_XY'
        val_mask_dir = f'{BASE_PATH}/fold_{i}/test/mask'
        
        # 获取该折的模型权重文件
        ckpt_path = CKPT_PATH_TEMPLATE.format(i=i)

        # 启动评估
        preprocess_and_inference(
            fold=i, 
            ckpt=ckpt_path, 
            threod=THRESHOLD, 
            val_h5_dir=val_h5_dir, 
            val_signal_dir=val_signal_dir, 
            val_mask_dir=val_mask_dir, 
            num_epochs=NUM_EPOCHS, 
            learning_rate=LEARNING_RATE, 
            batch_size=BATCH_SIZE, 
            device=DEVICE, 
            feature_dim=FEATURE_DIM,
            inter_steps = Inter_steps_num
        )