import os
import time
import random
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import label, find_objects
from skimage.morphology import skeletonize_3d
from skimage.measure import label as label_1
from skimage.measure import regionprops

# 确保 efficientunet 模块在同级目录下
from efficientunet import *


# =========================================================================
# 1. 用户配置参数区域 (User Configuration Area) - 所有需要修改的参数都在这里
# =========================================================================

# ----------------- 基础与设备配置 -----------------
MULTI_GPU = False
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------- 推断/验证超参数 -----------------
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 1             # 验证通常跑一轮即可
ITERATION_STEPS = 20   # 掩膜修正迭代次数 (while count < 20)

FOLD_LIST = [1]        # 需要验证的折数列表
CHOICES = [1]          # 类别选择
THREOD_SIM_LIST = [0.85] # 相似度阈值列表

# ----------------- 路径与数据配置 -----------------
# 根目录与特定实验数据路径
BASE_TEST_PATH = '/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/125WSI/fold_1/user_study/TCGA-A2-A3XX-DX1_xmin69942_medium'

# 数据子文件夹名称
IMAGE_SUBDIR = "image_npy"
MASK_SUBDIR = "mask_npy"
SIGNAL_SUBDIR = "signal_max_point_npy"
SUPERPIXEL_SUBDIR = "image_SLIC_500"

# 模型权重路径配置 (支持 f-string 动态折数)
def get_ckpt_resunet(fold):
    return f'/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/125WSI/fold_{fold}/ckpt/cos2_BCSS_resunet_best.pth'

def get_ckpt_effiunet(fold):
    return f'/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/125WSI/fold_{fold}/ROI_ckpt/BCSS_effi-Unet_roi_best_1+1_threod_allmask.pth'

# =========================================================================


# ------------------------- 辅助函数 -------------------------

def generateGuidingSignal(binaryMask):
    binaryMask = binaryMask.to(torch.uint8)
    if binaryMask.sum() > 1:
        distance_map = distance_transform_edt(binaryMask.cpu().numpy())
        distance_map = torch.tensor(distance_map, dtype=torch.float32, device=binaryMask.device)
        
        tempMean = distance_map.mean().cpu().numpy()
        tempStd = distance_map.std().cpu().numpy()
        
        tempThresh = np.random.uniform(tempMean - tempStd, tempMean + tempStd)
        tempThresh = torch.tensor(tempThresh, device=binaryMask.device)
        
        if tempThresh < 0:
            tempThresh = np.random.uniform(tempMean / 2, tempMean + tempStd / 2)
            tempThresh = torch.tensor(tempThresh, device=binaryMask.device)
        
        newMask = distance_map > tempThresh
        if newMask.sum() == 0:
            newMask = distance_map > (tempThresh / 2)
        if newMask.sum() == 0:
            newMask = binaryMask

        skel = skeletonize_3d(newMask.cpu().numpy())
        skel = torch.tensor(skel, dtype=torch.float32, device=binaryMask.device)
    else:
        skel = torch.zeros_like(binaryMask, dtype=torch.float32, device=binaryMask.device)
    return skel


def generateGuidingSignal_1(mask, RandomizeGuidingSignalType):
    mask = mask.squeeze(0)
    if RandomizeGuidingSignalType == 'Skeleton':
        binaryMask = (mask > (0.5 * mask.max())).to(torch.uint8)
        if binaryMask.sum() > 1:
            distance_map = distance_transform_edt(binaryMask.cpu().numpy())
            distance_map = torch.tensor(distance_map, dtype=torch.float32, device=mask.device)
            tempMean = distance_map.mean().cpu().numpy()
            tempStd = distance_map.std().cpu().numpy()
            tempThresh = np.random.uniform(tempMean - tempStd, tempMean + tempStd)
            tempThresh = torch.tensor(tempThresh, device=mask.device)
            
            if tempThresh < 0:
                tempThresh = np.random.uniform(tempMean / 2, tempMean + tempStd / 2)
                tempThresh = torch.tensor(tempThresh, device=mask.device)
            
            newMask = distance_map > tempThresh
            if newMask.sum() == 0:
                newMask = distance_map > (tempThresh / 2)
            if newMask.sum() == 0:
                newMask = binaryMask

            skel = skeletonize_3d(newMask.cpu().numpy())
            skel = torch.tensor(skel, dtype=torch.float32, device=mask.device)
        else:
            skel = torch.zeros_like(binaryMask, dtype=torch.float32).unsqueeze(-1)
        return skel


def processMasks(pred_mask_all, GT_mask_all):
    pred_mask_all = (pred_mask_all > 0.5).float()
    batch_size, _, H, W = pred_mask_all.shape
    output = torch.zeros(batch_size, 2, H, W, device=pred_mask_all.device, dtype=torch.float32)
    centers = []

    for i in range(batch_size):
        pred_mask = pred_mask_all[i].squeeze(0)
        GT_mask = GT_mask_all[i].squeeze(0)

        fg = ((GT_mask == 1) & (pred_mask == 0)).to(torch.float32)
        bg = ((GT_mask == 0) & (pred_mask == 1)).to(torch.float32)
        
        # 找出前景的最大连通域
        if fg.sum() > 0:
            labeled_fg = label_1(fg.cpu().numpy(), connectivity=1)
            regions_fg = regionprops(labeled_fg)
            if regions_fg:
                largest_region_fg = max(regions_fg, key=lambda r: r.area)
                fg_largest = torch.from_numpy(labeled_fg == largest_region_fg.label).to(fg.device, dtype=torch.float32)
                fg_center = largest_region_fg.centroid
                fg_center = (round(fg_center[0]), round(fg_center[1]))
            else:
                fg_largest = torch.zeros_like(fg)
                fg_center = None
        else:
            fg_largest = torch.zeros_like(fg)
            fg_center = None

        # 找出背景的最大连通域
        if bg.sum() > 0:
            labeled_bg = label_1(bg.cpu().numpy(), connectivity=1)
            regions_bg = regionprops(labeled_bg)
            if regions_bg:
                largest_region_bg = max(regions_bg, key=lambda r: r.area)
                bg_largest = torch.from_numpy(labeled_bg == largest_region_bg.label).to(bg.device, dtype=torch.float32)
                bg_center = largest_region_bg.centroid
                bg_center = (round(bg_center[0]), round(bg_center[1]))
            else:
                bg_largest = torch.zeros_like(bg)
                bg_center = None
        else:
            bg_largest = torch.zeros_like(bg)
            bg_center = None
        
        fg_area = fg_largest.sum().item()
        bg_area = bg_largest.sum().item()

        if fg_area >= bg_area:
            output[i, 0] = generateGuidingSignal(fg_largest) if fg_largest.sum() > 0 else torch.zeros_like(pred_mask)
            centers.append(fg_center)
        else:
            output[i, 1] = generateGuidingSignal(bg_largest) if bg_largest.sum() > 0 else torch.zeros_like(pred_mask)
            centers.append(bg_center)
            
    return output, centers


def get_largest_connected_component(roi_mask):
    roi_mask_np = roi_mask.squeeze(0).cpu().numpy()
    structure = np.ones((3, 3), dtype=int)
    labeled_array, num_features = label(roi_mask_np, structure)

    if num_features == 0:
        return torch.zeros_like(roi_mask)

    regions = find_objects(labeled_array)
    region_sizes = [np.sum(labeled_array[region] == labeled_array[region][0, 0]) for region in regions]
    max_region_index = np.argmax(region_sizes)

    new_roi_mask_np = np.zeros_like(roi_mask_np)
    new_roi_mask_np[labeled_array == max_region_index + 1] = 1

    new_roi_mask = torch.tensor(new_roi_mask_np, dtype=torch.float32).unsqueeze(0)
    return new_roi_mask


def ROI_crop_signal_line(input, aux_input, superpixel, mask):
    batch_size, _, H, W = input.shape
    
    roi_inputs, roi_aux_inputs, roi_suppixels, roi_masks = [], [], [], []
    mask_box = torch.zeros_like(mask)
    all_aux_inputs = aux_input.clone()
    centers = []

    for b in range(batch_size):
        indices = (aux_input[b, 0] == 1).nonzero(as_tuple=True)
        if indices[0].numel() > 0:
            center_y = indices[0].float().mean().round().long().item()
            center_x = indices[1].float().mean().round().long().item()

            start_y, start_x = max(center_y - 128, 0), max(center_x - 128, 0)
            end_y, end_x = min(start_y + 256, H), min(start_x + 256, W)

            if end_y - start_y < 256: start_y = max(end_y - 256, 0)
            if end_x - start_x < 256: start_x = max(end_x - 256, 0)
        else:
            start_y = torch.randint(0, max(H - 256, 1), (1,)).item()
            start_x = torch.randint(0, max(W - 256, 1), (1,)).item()
            end_y, end_x = start_y + 256, start_x + 256
            center_x, center_y = start_x + 128, start_y + 128
            
        roi_input = input[b, :, start_y:end_y, start_x:end_x]
        roi_suppixel = superpixel[b, :, start_y:end_y, start_x:end_x]
        roi_mask = mask[b, :, start_y:end_y, start_x:end_x]
        roi_mask_new = get_largest_connected_component(roi_mask)
        
        mask_box[b, :, start_y:end_y, start_x:end_x] = 1
        
        guidingSignal = generateGuidingSignal_1(roi_mask_new, 'Skeleton').unsqueeze(0)
        guidingSignal_expanded = torch.cat([guidingSignal, torch.zeros_like(guidingSignal)], dim=0)
        if guidingSignal_expanded.dim() == 4:
            guidingSignal_expanded = guidingSignal_expanded.squeeze(-1)
        guidingSignal_expanded = guidingSignal_expanded.to(input.device)

        roi_inputs.append(roi_input)
        roi_aux_inputs.append(guidingSignal_expanded)
        roi_suppixels.append(roi_suppixel)
        roi_masks.append(roi_mask)
        centers.append([center_x, center_y])
        
    roi_inputs = torch.stack(roi_inputs) if roi_inputs else None
    roi_aux_inputs = torch.stack(roi_aux_inputs) if roi_aux_inputs else None
    roi_suppixels = torch.stack(roi_suppixels) if roi_suppixels else None
    roi_masks = torch.stack(roi_masks) if roi_masks else None
    
    roi_aux_inputs = roi_aux_inputs.to(input.device)
    all_aux_inputs = all_aux_inputs.to(input.device)
    
    return roi_inputs, roi_aux_inputs, roi_suppixels, roi_masks, mask_box, all_aux_inputs, centers


# ------------------------- 评价指标 -------------------------

def dice_coeff(y_true, y_pred, a=1., b=1.):
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)
    intersection = torch.sum(y_true_flat * y_pred_flat)
    return (2. * intersection + a) / (torch.sum(y_true_flat) + torch.sum(y_pred_flat) + b)

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
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    iou_foreground = compute_iou(pred, target, 1)
    return iou_foreground

def calculate_binary_segmentation_accuracy(preds, labels):
    if preds.dim() == 4: preds = preds.squeeze(1)
    if labels.dim() == 4: labels = labels.squeeze(1)
    preds = (preds > 0.5).float()
    
    correct = (preds == labels).float().sum(dim=[1, 2])
    total_pixels_per_sample = labels.size(1) * labels.size(2)
    accuracy_per_sample = correct / total_pixels_per_sample
    mean_accuracy = accuracy_per_sample.mean().item()
    return accuracy_per_sample, mean_accuracy


# ------------------------- 模型定义 -------------------------

def convbnrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(out_channels, eps=1e-5),
        nn.ReLU(inplace=True),
    )

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


class ResNetUNet_proto(nn.Module):
    def __init__(self, freeze=False, pretrained=True):
        super().__init__()
        self.freeze = freeze
        self.ResNetUNet = ResNetUNet(pretrained=pretrained)
        self.segment_part = get_efficientunet_b0(out_channels=1, concat_input=True, pretrained=False, backbone=False)

    def forward(self, roi_input, roi_aux_input, roi_suppixel, input, aux_input, superpixels, mask_box, threod):
        pred_mask = torch.zeros_like(roi_suppixel)
        roiseg_input = torch.cat((roi_input, pred_mask, roi_aux_input), dim=1)
        seg_orginal = self.segment_part(roiseg_input)
        x = self.ResNetUNet(input)
        
        fg_sigmoid_output = seg_orginal.clone()
        mask_greater_than_0_6 = fg_sigmoid_output > 0.95
        mask_less_equal_0_6 = fg_sigmoid_output <= 0.95
        fg_sigmoid_output[mask_greater_than_0_6] = 1
        fg_sigmoid_output[mask_less_equal_0_6] = 0
        
        crop_size = 256
        mask_box = mask_box.squeeze(1)
        x_cropped = x * mask_box.unsqueeze(1)
        cropped_regions = []

        for i in range(x.shape[0]):
            mask = mask_box[i]
            feature_map = x_cropped[i]
            nonzero_coords = torch.nonzero(mask, as_tuple=True)
            
            if len(nonzero_coords[0]) > 0:
                center_y = (nonzero_coords[0].min() + nonzero_coords[0].max()) // 2
                center_x = (nonzero_coords[1].min() + nonzero_coords[1].max()) // 2
                start_y = max(0, center_y - crop_size // 2)
                start_x = max(0, center_x - crop_size // 2)
                end_y, end_x = start_y + crop_size, start_x + crop_size

                start_y = min(start_y, feature_map.shape[1] - crop_size)
                start_x = min(start_x, feature_map.shape[2] - crop_size)
                end_y, end_x = start_y + crop_size, start_x + crop_size

                cropped = feature_map[:, start_y:end_y, start_x:end_x]
            else:
                cropped = torch.zeros((x.shape[1], crop_size, crop_size), dtype=x.dtype, device=x.device)
            cropped_regions.append(cropped)

        output_tensor = torch.stack(cropped_regions, dim=0)
        foreground_features = output_tensor * fg_sigmoid_output
        foreground_pixel_count = torch.clamp(fg_sigmoid_output.sum(dim=(2, 3), keepdim=True), min=1)
        
        fg_avg_features = foreground_features.sum(dim=(2, 3), keepdim=True) / foreground_pixel_count
        fg_avg_features = fg_avg_features.squeeze(3).squeeze(2)
        
        x_normalized = F.normalize(x, dim=1)
        prototype_normalized = F.normalize(fg_avg_features, dim=1)
        out_mask = (torch.einsum('bchw,bc->bhw', x_normalized, prototype_normalized))**2
        
        out_mask = out_mask.unsqueeze(1)
        min_val = out_mask.min()
        max_val = out_mask.max()
        out_mask_normalized = (out_mask - min_val) / (max_val - min_val + 1e-6)
        
        out_mask_thresh = out_mask_normalized.clone()
        out_mask_thresh[out_mask_thresh <= threod] = 0
        out_mask_thresh[out_mask_thresh > threod] = 1

        return x, out_mask_thresh, fg_sigmoid_output, x, out_mask_normalized, superpixels


# ------------------------- 数据集定义 -------------------------

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, signal_dir, masks_dir, suppixel_dir, filenames):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.signal_dir = signal_dir
        self.suppixel_dir = suppixel_dir
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = np.load(os.path.join(self.images_dir, filename))
        mask = np.load(os.path.join(self.masks_dir, filename))
        signal = np.load(os.path.join(self.signal_dir, filename))
        suppixel = np.load(os.path.join(self.suppixel_dir, filename))

        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        signal = torch.tensor(signal.transpose(2, 0, 1), dtype=torch.float32)
        suppixel = torch.tensor(suppixel.transpose(2, 0, 1), dtype=torch.float32)

        return image, signal, mask, suppixel, filename

def get_filenames_from_folder(folder_path):
    return [filename for filename in os.listdir(folder_path) if filename.endswith('.npy')]


# ------------------------- 推断逻辑 -------------------------

def inference_model(model, val_loader, epochs=1, threod=0.4, fold=1, cls_num="1"):
    model.to(DEVICE)
    for epoch in range(epochs):
        model.eval()
        
        number = ITERATION_STEPS + 1 # count from 0 to 24 + initial = 25
        val_dice_score = [0.0 for _ in range(number)]
        nuclick_val_dice = [0.0 for _ in range(number)]
        val_accuracy = [0.0 for _ in range(number)]  
        nuclick_val_acc = [0.0 for _ in range(number)]
        mean_IoU = [0.0 for _ in range(number)]
        mean_IoU_list = [[] for _ in range(number)]

        with torch.no_grad():
            for images, aux_inputs, masks, superpixels, filenames in val_loader:
                images, aux_inputs, masks, superpixels = images.to(DEVICE), aux_inputs.to(DEVICE), masks.to(DEVICE), superpixels.to(DEVICE)
                
                roi_input, roi_aux_input, roi_suppixel, roi_mask, mask_box, all_aux_inputs, centers_1 = ROI_crop_signal_line(images, aux_inputs, superpixels, masks)

                torch.cuda.synchronize() 
                all_perpixel_features, pre_masks, nuclick_out, all_pixel_features, out_mask_normalized, superpixels = model(roi_input, roi_aux_input, roi_suppixel, images, aux_inputs, superpixels, mask_box, threod)
                torch.cuda.synchronize() 

                out_put = pre_masks.clone()
                pre_mask_list = [out_put]
                
                count = 0
                signal, centers = processMasks(out_put, masks)
                union_signal = torch.bitwise_or(signal.to(torch.uint8), aux_inputs.to(torch.uint8))
                
                while count < ITERATION_STEPS:
                    if count > 0:
                        out_put = pre_masks
                        signal, centers = processMasks(out_put, masks)
                        union_signal = torch.bitwise_or(signal.to(torch.uint8), union_signal.to(torch.uint8))
                    
                    batch_size, _, H, W = out_put.shape
                    roi_inputs, roi_aux_inputs, roi_pred_masks = [], [], []

                    for b in range(batch_size):
                        start_y = max(centers[b][0] - 128, 0)
                        start_x = max(centers[b][1] - 128, 0)
                        end_y, end_x = min(start_y + 256, H), min(start_x + 256, W)

                        if end_y - start_y < 256: start_y = int(max(end_y - 256, 0))
                        if end_x - start_x < 256: start_x = int(max(end_x - 256, 0))

                        roi_inputs.append(images[b, :, start_y:end_y, start_x:end_x])
                        roi_aux_inputs.append(union_signal[b, :, start_y:end_y, start_x:end_x])
                        roi_pred_masks.append(out_put[b, :, start_y:end_y, start_x:end_x])

                    roi_inputs = torch.stack(roi_inputs) if roi_inputs else None
                    roi_aux_inputs = torch.stack(roi_aux_inputs) if roi_aux_inputs else None
                    roi_pred_masks = torch.stack(roi_pred_masks) if roi_pred_masks else None

                    roiseg_input = torch.cat((roi_inputs, roi_pred_masks, roi_aux_inputs), dim=1)
                    seg_orginal = model.segment_part(roiseg_input)

                    fg_sigmoid_output = seg_orginal.clone()
                    fg_sigmoid_output[fg_sigmoid_output > 0.5] = 1
                    fg_sigmoid_output[fg_sigmoid_output <= 0.5] = 0
                    
                    for b in range(batch_size):
                        start_y = max(centers[b][0] - 128, 0)
                        start_x = max(centers[b][1] - 128, 0)
                        end_y, end_x = min(start_y + 256, H), min(start_x + 256, W)

                        if end_y - start_y < 256: start_y = int(max(end_y - 256, 0))
                        if end_x - start_x < 256: start_x = int(max(end_x - 256, 0))
                            
                        pre_masks[b, :, start_y:end_y, start_x:end_x] = fg_sigmoid_output[b]
                        out_mask_normalized[b, :, start_y:end_y, start_x:end_x] = seg_orginal.clone()[b]
                        
                    pre_masks[pre_masks > 0.5] = 1
                    pre_masks[pre_masks <= 0.5] = 0
                    pre_mask_list.append(pre_masks.clone())
                    count += 1 

                for i in range(len(pre_mask_list)):
                    val_dice_score[i] += dice_coeff(pre_mask_list[i], masks).item() * images.size(0)
                    nuclick_val_dice[i] += dice_coeff(nuclick_out, roi_mask).item() * images.size(0)
                    
                    _, batch_accuracy = calculate_binary_segmentation_accuracy(pre_mask_list[i], masks)
                    val_accuracy[i] += batch_accuracy * images.size(0)
                    
                    _, nuclick_batch_accuracy = calculate_binary_segmentation_accuracy(nuclick_out, roi_mask)
                    nuclick_val_acc[i] += nuclick_batch_accuracy * images.size(0)
                    
                    for pred, mask in zip(pre_mask_list[i], masks):
                        miou = compute_miou_binary(pred, mask)
                        if not np.isnan(miou):
                            mean_IoU_list[i].append(miou)

            dataset_len = len(val_loader.dataset)
            for i in range(len(val_dice_score)):
                val_dice_score[i] /= dataset_len
                nuclick_val_dice[i] /= dataset_len
                val_accuracy[i] /= dataset_len
                nuclick_val_acc[i] /= dataset_len
                mean_IoU[i] = np.mean(mean_IoU_list[i])
        
        # 只打印前 21 次迭代以对齐原逻辑
        for i in range(min(21, len(val_dice_score))):
            print(f'Iteration: {i+1} | Val Dice: {val_dice_score[i]:.4f} | Val Mean IOU: {mean_IoU[i]:.4f} | Val Acc: {val_accuracy[i]:.4f} | roi_Val_Dice: {nuclick_val_dice[i]:.4f} | roi_Val_Acc: {nuclick_val_acc[i]:.4f}')
        
        return val_dice_score[-1], mean_IoU[-1], val_accuracy[-1]


# ------------------------- 主函数入口 -------------------------

if __name__ == '__main__':
    start_time = time.time()
    
    for fold_num in FOLD_LIST:
        print(f"\n{'='*50}")
        print(f"Starting Evaluation for Fold: {fold_num}")
        print(f"{'='*50}")
        
        # 1. 实例化模型
        model = ResNetUNet_proto().to(DEVICE)

        # 2. 加载 ResNetUNet (移除 DataParallel 保存时产生的 'module.' 前缀)
        ckpt_res = get_ckpt_resunet(fold_num)
        res_state_dict = torch.load(ckpt_res, map_location='cpu')
        new_res_state_dict = OrderedDict()
        for k, v in res_state_dict.items():
            if k.startswith('module.'):
                new_res_state_dict[k[7:]] = v
            else:
                new_res_state_dict[k] = v
        model.ResNetUNet.load_state_dict(new_res_state_dict)

        # 3. 加载 segment_part
        ckpt_effi = get_ckpt_effiunet(fold_num)
        model.segment_part.load_state_dict(torch.load(ckpt_effi, map_location='cpu'))

        # 4. 冻结所有模型参数 (推断模式)
        for param in model.segment_part.parameters():
            param.requires_grad = False 
        for param in model.ResNetUNet.parameters():
            param.requires_grad = False 
            
        if MULTI_GPU:
            model = nn.DataParallel(model)

        for threod_sim in THREOD_SIM_LIST:
            dice_list, iou_list, acc_list = [], [], []
            
            for cls in CHOICES:
                # 动态构建路径
                test_images_dir = os.path.join(BASE_TEST_PATH, IMAGE_SUBDIR)
                test_masks_dir = os.path.join(BASE_TEST_PATH, MASK_SUBDIR)
                test_signal_dir = os.path.join(BASE_TEST_PATH, SIGNAL_SUBDIR)
                test_superpixel_dir = os.path.join(BASE_TEST_PATH, SUPERPIXEL_SUBDIR)

                test_filenames = get_filenames_from_folder(test_images_dir)
                test_dataset = CustomDataset(test_images_dir, test_signal_dir, test_masks_dir, test_superpixel_dir, test_filenames)
                test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

                # 开始推断
                dice, iou, acc = inference_model(model, test_loader, epochs=EPOCHS, threod=threod_sim, fold=fold_num, cls_num=str(cls))
                dice_list.append(dice)
                iou_list.append(iou)
                acc_list.append(acc)

                print(f"Class {cls} | Batch Size = {BATCH_SIZE} | Evaluation Finished.")

            mdice = np.mean(dice_list)
            mAcc = np.mean(acc_list)
            miou = np.mean(iou_list)

            print("-" * 50)
            print(f"Summary -> Fold: {fold_num} | Sim_Threshold: {threod_sim:.4f}")
            print(f"Mean Dice: {mdice:.4f} | Mean Acc: {mAcc:.4f} | Mean IoU: {miou:.4f}")
            print(f"Loaded Backbone from: {ckpt_res}")
            print("-" * 50)
        
        # 清空显存进行下一折
        torch.cuda.empty_cache()

    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.4f} seconds")