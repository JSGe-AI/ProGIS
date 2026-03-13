import os
import time
import pickle
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import label, find_objects
from skimage.morphology import skeletonize_3d
from skimage.measure import label as label_1
from skimage.measure import regionprops
from PIL import Image

# 假设该模块在你的本地环境中存在
from efficientunet import *

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



# ==============================================================================
# 核心功能与形态学处理 (Core Functions & Morphology)
# ==============================================================================

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

def processMasks(pred_mask_all, GT_mask_all):
    pred_mask_all = (pred_mask_all > 0.5).float()
    batch_size, _, H, W = pred_mask_all.shape
    output = torch.zeros(batch_size, 2, H, W, device=pred_mask_all.device, dtype=torch.float32)
    centers = [] 

    for i in range(batch_size):
        pred_mask = pred_mask_all[i].squeeze(0) 
        GT_mask = GT_mask_all[i].squeeze(0) 

        fg = (GT_mask == 1) & (pred_mask == 0)
        fg = fg.to(torch.float32) 

        bg = (GT_mask == 0) & (pred_mask == 1)
        bg = bg.to(torch.float32) 
        
        if fg.sum() > 0:
            labeled_fg = label_1(fg.cpu().numpy(), connectivity=1)
            regions_fg = regionprops(labeled_fg)
            if regions_fg:
                largest_region_fg = max(regions_fg, key=lambda r: r.area)
                fg_largest = (labeled_fg == largest_region_fg.label)
                fg_largest = torch.from_numpy(fg_largest).to(fg.device, dtype=torch.float32)
                fg_center = largest_region_fg.centroid
                fg_center = (round(fg_center[0]), round(fg_center[1])) 
            else:
                fg_largest = torch.zeros_like(fg)
                fg_center = None
        else:
            fg_largest = torch.zeros_like(fg)
            fg_center = None

        if bg.sum() > 0:
            labeled_bg = label_1(bg.cpu().numpy(), connectivity=1)
            regions_bg = regionprops(labeled_bg)
            if regions_bg:
                largest_region_bg = max(regions_bg, key=lambda r: r.area)
                bg_largest = (labeled_bg == largest_region_bg.label)
                bg_largest = torch.from_numpy(bg_largest).to(bg.device, dtype=torch.float32)
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
            largest_connected = fg_largest
            fg_skeleton = generateGuidingSignal(largest_connected) if largest_connected.sum() > 0 else torch.zeros_like(pred_mask, dtype=torch.float32, device=pred_mask.device) 
            output[i, 0] = fg_skeleton 
            centers.append(fg_center) 
        else:
            largest_connected = bg_largest
            bg_skeleton = generateGuidingSignal(largest_connected) if largest_connected.sum() > 0 else torch.zeros_like(pred_mask, dtype=torch.float32, device=pred_mask.device) 
            output[i, 1] = bg_skeleton 
            centers.append(bg_center) 
         
    return output, centers

def get_largest_connected_component(roi_mask):
    roi_mask_np = roi_mask.squeeze(0).cpu().numpy() 

    structure = np.ones((3, 3), dtype=int) 
    labeled_array, num_features = label(roi_mask_np, structure) 

    if num_features == 0:
        return torch.zeros_like(roi_mask)

    regions = find_objects(labeled_array) 
    region_sizes = []
    for region in regions:
        region_mask = labeled_array[region] == labeled_array[region][0, 0] 
        region_sizes.append(np.sum(region_mask)) 

    max_region_index = np.argmax(region_sizes)
    new_roi_mask_np = np.zeros_like(roi_mask_np) 

    new_roi_mask_np[labeled_array == max_region_index + 1] = 1 
    new_roi_mask = torch.tensor(new_roi_mask_np, dtype=torch.float32).unsqueeze(0) 

    return new_roi_mask

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

            start_y = max(center_y - 128, 0)
            start_x = max(center_x - 128, 0)
            end_y = min(start_y + 256, H)
            end_x = min(start_x + 256, W)

            if end_y - start_y < 256:
                start_y = max(end_y - 256, 0)
            if end_x - start_x < 256:
                start_x = max(end_x - 256, 0)
        else:
            start_y = torch.randint(0, max(H - 256, 1), (1,)).item()
            start_x = torch.randint(0, max(W - 256, 1), (1,)).item()
            end_y = start_y + 256
            end_x = start_x + 256
            
        roi_input = input[b, :, start_y:end_y, start_x:end_x]
        roi_suppixel = superpixel[b, :, start_y:end_y, start_x:end_x]
        roi_mask = mask[b, : ,start_y:end_y, start_x:end_x]
        roi_mask_new = get_largest_connected_component(roi_mask)
    
        mask_box[b, :, start_y:end_y, start_x:end_x] = 1
        
        guidingSignal = generateGuidingSignal_1(roi_mask_new, Config.RANDOMIZE_GUIDING_SIGNAL_TYPE)
        guidingSignal = guidingSignal.unsqueeze(0)

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

# ==============================================================================
# 相似度计算与网络模型 (Models)
# ==============================================================================
class EfficientUNet_proto(nn.Module):
    def __init__(self, freeze=False, pretrained=True):
        super().__init__()
        self.freeze = freeze
        self.EfficientUNet_backbone = get_efficientunet_b0(out_channels=1, concat_input=True, pretrained=False, backbone=True)
        self.segment_part = get_efficientunet_b0(out_channels=1, concat_input=True, pretrained=False, backbone=False)
        
    def forward(self, roi_input, roi_aux_input, roi_suppixel, input, aux_input, superpixels, mask_box, threod):
        pred_mask = torch.zeros_like(roi_suppixel)
        roiseg_input = torch.cat((roi_input, pred_mask, roi_aux_input), dim=1)
        
        seg_orginal = self.segment_part(roiseg_input)
        x = self.EfficientUNet_backbone(input)
        
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
                end_y = start_y + crop_size
                end_x = start_x + crop_size

                start_y = min(start_y, feature_map.shape[1] - crop_size)
                start_x = min(start_x, feature_map.shape[2] - crop_size)
                end_y = start_y + crop_size
                end_x = start_x + crop_size

                cropped = feature_map[:, start_y:end_y, start_x:end_x] 
            else:
                cropped = torch.zeros((x.shape[1], crop_size, crop_size), dtype=x.dtype, device=x.device)
            
            cropped_regions.append(cropped)

        output_tensor = torch.stack(cropped_regions, dim=0) 
        foreground_features = output_tensor * fg_sigmoid_output 

        foreground_pixel_count = fg_sigmoid_output.sum(dim=(2, 3), keepdim=True) 
        foreground_pixel_count = torch.clamp(foreground_pixel_count, min=1) 

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

def ROI_crop(input, aux_input, superpixel, mask):
    # 此函数与 ROI_crop_signal_line 基本一致，出于完整性保留
    batch_size, _, H, W = input.shape
    roi_inputs, roi_aux_inputs, roi_suppixels, roi_masks = [], [], [], []

    for b in range(batch_size):
        indices = (aux_input[b, 0] == 1).nonzero(as_tuple=True)
        if indices[0].numel() > 0:
            center_y = indices[0].float().mean().round().long().item()
            center_x = indices[1].float().mean().round().long().item()

            start_y = max(center_y - 128, 0)
            start_x = max(center_x - 128, 0)
            end_y = min(start_y + 256, H)
            end_x = min(start_x + 256, W)

            if end_y - start_y < 256:
                start_y = max(end_y - 256, 0)
            if end_x - start_x < 256:
                start_x = max(end_x - 256, 0)
        else:
            start_y = torch.randint(0, max(H - 256, 1), (1,)).item()
            start_x = torch.randint(0, max(W - 256, 1), (1,)).item()
            end_y = start_y + 256
            end_x = start_x + 256
            
        roi_inputs.append(input[b, :, start_y:end_y, start_x:end_x])
        roi_aux_inputs.append(aux_input[b, :, start_y:end_y, start_x:end_x])
        roi_suppixels.append(superpixel[b, :, start_y:end_y, start_x:end_x])
        roi_masks.append(mask[b, :, start_y:end_y, start_x:end_x])

    roi_inputs = torch.stack(roi_inputs) if roi_inputs else None
    roi_aux_inputs = torch.stack(roi_aux_inputs) if roi_aux_inputs else None
    roi_suppixels = torch.stack(roi_suppixels) if roi_suppixels else None
    roi_masks = torch.stack(roi_masks) if roi_masks else None
    
    return roi_inputs, roi_aux_inputs, roi_suppixels, roi_masks

# ==============================================================================
# 评价指标 (Metrics)
# ==============================================================================
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
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    iou_foreground = compute_iou(pred, target, 1)
    return iou_foreground

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

def calculate_metrics(output, target):
    batch_size = output.size(0)
    output = torch.sigmoid(output) 
    output = (output > 0.5).float() 

    dice_scores, iou_scores, acc_scores = [], [], []

    for i in range(batch_size):
        pred = output[i, 0] 
        true = target[i, 0] 

        intersection = torch.sum(pred * true)
        union = torch.sum(pred) + torch.sum(true)
        dice = (2 * intersection) / (union + 1e-7) 
        dice_scores.append(dice.item())

        iou = intersection / (union - intersection + 1e-7)
        iou_scores.append(iou.item())

        acc = torch.sum(pred == true).float() / true.numel()
        acc_scores.append(acc.item())

    return sum(dice_scores)/batch_size, sum(acc_scores)/batch_size, sum(iou_scores)/batch_size

def NoI(pre_mask_list, masks, filenames):
    batch_size, _, _, _ = masks.shape
    iou_batch_list, dice_batch_list, acc_batch_list = [], [], []

    for b in range(batch_size):
        iou_single_list, dice_single_list, acc_single_list = [], [], []
        for i, pre_mask in enumerate(pre_mask_list):
            iou_single_list.append(compute_miou_binary(pre_mask[b], masks[b]))
            acc_single_list.append(calculate_binary_segmentation_accuracy(pre_mask[b], masks[b]))
            dice_single_list.append(dice_coeff(pre_mask[b], masks[b]))

        iou_batch_list.append(iou_single_list)
        acc_batch_list.append(acc_single_list)
        dice_batch_list.append(dice_single_list)
                
    return iou_batch_list, acc_batch_list, dice_batch_list

# ==============================================================================
# 数据加载与处理 (Data Loading)
# ==============================================================================
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
        
        image_path = os.path.join(self.images_dir, filename)
        mask_path = os.path.join(self.masks_dir, filename)
        signal_path = os.path.join(self.signal_dir, filename)
        suppixel_path = os.path.join(self.suppixel_dir, filename)

        image = np.load(image_path)
        mask = np.load(mask_path)
        signal = np.load(signal_path)
        suppixel = np.load(suppixel_path)

        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) 
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) 
        signal = torch.tensor(signal.transpose(2, 0, 1), dtype=torch.float32) 
        suppixel = torch.tensor(suppixel.transpose(2, 0, 1), dtype=torch.float32) 

        return image, signal, mask, suppixel, filename

def get_filenames_from_folder(folder_path):
    return [filename for filename in os.listdir(folder_path) if filename.endswith('.npy')]   

# ==============================================================================
# 模型评估循环 (Evaluation Loop)
# ==============================================================================
def inference_model(model, test_loader, epochs=1, threod=0.4, fold=1, cls_num="1", ITERATION_STEPS = 20):
    model.to(Config.DEVICE)

    for epoch in range(epochs):
        model.eval()
        number = 21
        
        iou_NOI_list, acc_NOI_list, dice_NOI_list = [], [], []
        test_dice_score = [0.0] * number
        nuclick_test_dice = [0.0] * number
        test_accuracy = [0.0] * number  
        nuclick_test_acc = [0.0] * number
        mean_IoU = [0.0] * number
        mean_IoU_list = [[] for _ in range(number)]

        with torch.no_grad():
            for images, aux_inputs, masks, superpixels, filenames in test_loader:
                images, aux_inputs, masks, superpixels = images.to(Config.DEVICE), aux_inputs.to(Config.DEVICE), masks.to(Config.DEVICE), superpixels.to(Config.DEVICE)
                
                roi_input, roi_aux_input, roi_suppixel, roi_mask, mask_box, all_aux_inputs, centers_1 = ROI_crop_signal_line(images, aux_inputs, superpixels, masks)
            
                all_perpixel_features, pre_masks, nuclick_out, all_pixel_features, out_mask_normalized, superpixels = model(
                    roi_input, roi_aux_input, roi_suppixel, images, aux_inputs, superpixels, mask_box, threod
                )
                
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
                        end_y = min(start_y + 256, H)
                        end_x = min(start_x + 256, W)

                        if end_y - start_y < 256:
                            start_y = int(max(end_y - 256, 0))
                        if end_x - start_x < 256:
                            start_x = int(max(end_x - 256, 0))
                            
                        mask_box = torch.zeros_like(masks)
                        roi_inputs.append(images[b, :, start_y:end_y, start_x:end_x])
                        roi_aux_inputs.append(union_signal[b, :, start_y:end_y, start_x:end_x])
                        roi_pred_masks.append(out_put[b, :, start_y:end_y, start_x:end_x])

                    roi_inputs = torch.stack(roi_inputs) if roi_inputs else None
                    roi_aux_inputs = torch.stack(roi_aux_inputs) if roi_aux_inputs else None
                    roi_pred_masks = torch.stack(roi_pred_masks) if roi_pred_masks else None
                    
                    roiseg_input = torch.cat((roi_inputs, roi_pred_masks, roi_aux_inputs), dim=1)
                    seg_orginal = model.segment_part(roiseg_input)

                    fg_sigmoid_output = seg_orginal.clone() 
                    mask_greater_than_0_6 = fg_sigmoid_output > 0.5
                    mask_less_equal_0_6 = fg_sigmoid_output <= 0.5

                    fg_sigmoid_output[mask_greater_than_0_6] = 1
                    fg_sigmoid_output[mask_less_equal_0_6] = 0
                    
                    for b in range(batch_size):
                        start_y = max(centers[b][0] - 128, 0)
                        start_x = max(centers[b][1] - 128, 0)
                        end_y = min(start_y + 256, H)
                        end_x = min(start_x + 256, W)

                        if end_y - start_y < 256:
                            start_y = int(max(end_y - 256, 0))
                        if end_x - start_x < 256:
                            start_x = int(max(end_x - 256, 0))
                            
                        pre_masks[b, :, start_y:end_y, start_x:end_x] = fg_sigmoid_output[b]
                        out_mask_normalized[b, :, start_y:end_y, start_x:end_x] = seg_orginal.clone()[b]
                    
                    pre_masks[pre_masks > 0.5] = 1
                    pre_masks[pre_masks <= 0.5] = 0
                    pre_mask_list.append(pre_masks.clone())
                            
                    count += 1 

                iou_batch_list, acc_batch_list, dice_batch_list = NoI(pre_mask_list, masks, filenames)
                iou_NOI_list.extend(iou_batch_list)
                acc_NOI_list.extend(acc_batch_list)
                dice_NOI_list.extend(dice_batch_list)

                for i in range(len(pre_mask_list)):
                    test_dice_score[i] += dice_coeff(pre_mask_list[i], masks).item() * images.size(0)
                    nuclick_test_dice[i] += dice_coeff(nuclick_out, roi_mask).item() * images.size(0)
                    
                    _, batch_accuracy = calculate_binary_segmentation_accuracy(pre_mask_list[i], masks)
                    test_accuracy[i] += batch_accuracy * images.size(0)
                    
                    _, nuclick_batch_accuracy = calculate_binary_segmentation_accuracy(nuclick_out, roi_mask)
                    nuclick_test_acc[i] += nuclick_batch_accuracy * images.size(0)
                    
                    for pred, mask_ in zip(pre_mask_list[i], masks):
                        miou = compute_miou_binary(pred, mask_)
                        if not np.isnan(miou):
                            mean_IoU_list[i].append(miou)
            
            # 保存序列化数据
            fold_dir = Config.SAVE_RESULTS_DIR_TPL.format(fold)
            os.makedirs(fold_dir, exist_ok=True)    
            filename_iou = f"{fold_dir}/iou_{cls_num}_thr{threod}.pkl"
            filename_acc = f"{fold_dir}/acc_{cls_num}_thr{threod}.pkl"
            filename_dice = f"{fold_dir}/dice_{cls_num}_thr{threod}.pkl"
            
            try:
                with open(filename_iou, 'wb') as f:
                    pickle.dump(iou_NOI_list, f)
                with open(filename_acc, 'wb') as f:
                    pickle.dump(acc_NOI_list, f)
                with open(filename_dice, 'wb') as f:
                    pickle.dump(dice_NOI_list, f)
            except Exception as e:
                print(f"Error saving NOI list: {e}")
                    
            dataset_len = len(test_loader.dataset)

            for i in range(len(test_dice_score)):
                test_dice_score[i] /= dataset_len
                nuclick_test_dice[i] /= dataset_len
                test_accuracy[i] /= dataset_len
                nuclick_test_acc[i] /= dataset_len
                mean_IoU[i] = np.mean(mean_IoU_list[i])
        
        for i in range(21):
            print(f'Thresholds:{i+1:.4f} , test Dice: {test_dice_score[i]:.4f}, test_Mean IOU: {mean_IoU[i]:.4f}, test_Acc: {test_accuracy[i]:.4f}, nuclick_test_Dice: {nuclick_test_dice[i]:.4f},  nuclick_test_Acc: {nuclick_test_acc[i]:.4f}')
        
        return test_dice_score[19], mean_IoU[19], test_accuracy[19]
    
    
    
    
# ==============================================================================
# 用户配置区 (User Configuration) - 请在此处修改所有需要自定义的参数
# ==============================================================================
class Config:
    # 1. 基础环境与设备配置
    DEVICE = 'cuda:1'
    MULTI_GPU = False
    
    # 2. 核心参数与循环范围
    FOLD_LIST = [5]
    CHOICES = ['1', '2', '3', '4', '5']
    THREOD_SIM_LIST = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85]
    
    # 3. 数据集路径模板 (使用 {} 预留 fold_num 和 cls 的格式化位置)
    BASE_DATA_DIR = "/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/125WSI"
    TEST_IMG_DIR_TPL = BASE_DATA_DIR + "/fold_{}/test/{}/image_npy"
    TEST_MASK_DIR_TPL = BASE_DATA_DIR + "/fold_{}/test/{}/mask_npy"
    TEST_SIGNAL_DIR_TPL = BASE_DATA_DIR + "/fold_{}/test/{}/signal_max_point_npy"
    TEST_SUPPIXEL_DIR_TPL = BASE_DATA_DIR + "/fold_{}/test/{}/image_SLIC_500"
    
    # 4. 模型权重路径模板
    BACKBONE_CKPT_TPL = BASE_DATA_DIR + "/fold_{}/efficientUnet/efficientUnet_best.pth"
    ROI_CKPT_TPL = BASE_DATA_DIR + "/fold_{}/ROI_ckpt/BCSS_effi-Unet_roi_best_1+1_threod_allmask.pth"
    
    # 5. 结果保存与数据加载参数
    SAVE_RESULTS_DIR_TPL = BASE_DATA_DIR + "/fold_{}/efficientUnet/results"
    TEST_BATCH_SIZE = 16
    NUM_WORKERS = 4
    
    # 6. 算法特定参数
    RANDOMIZE_GUIDING_SIGNAL_TYPE = 'Skeleton'
    
    inter_num = 20  #
    

# ==============================================================================
# 主执行入口 (Main Execution)
# ==============================================================================
if __name__ == "__main__":
    for fold_num in Config.FOLD_LIST:
        model = EfficientUNet_proto()

        ckpt = Config.BACKBONE_CKPT_TPL.format(fold_num)
        ResNetUNet_state_dict = torch.load(ckpt, map_location='cpu')
        model.EfficientUNet_backbone.load_state_dict(ResNetUNet_state_dict, strict=True)
        
        roi_ckpt = Config.ROI_CKPT_TPL.format(fold_num)
        model.segment_part.load_state_dict(torch.load(roi_ckpt, map_location='cpu'))

        # 冻结参数
        for param in model.segment_part.parameters():
            param.requires_grad = False 
        for param in model.EfficientUNet_backbone.parameters():
            param.requires_grad = False 
            
        if Config.MULTI_GPU:
            model = nn.DataParallel(model, device_ids=[0, 1])

        for threod_sim in Config.THREOD_SIM_LIST:
            dice_list, iou_list, acc_list = [], [], []
            
            for cls in Config.CHOICES:
                test_images_dir = Config.TEST_IMG_DIR_TPL.format(fold_num, cls)
                test_masks_dir = Config.TEST_MASK_DIR_TPL.format(fold_num, cls)
                test_signal_dir = Config.TEST_SIGNAL_DIR_TPL.format(fold_num, cls)
                test_superpixel_dir = Config.TEST_SUPPIXEL_DIR_TPL.format(fold_num, cls)

                test_filenames = get_filenames_from_folder(test_images_dir)
                test_dataset = CustomDataset(test_images_dir, test_signal_dir, test_masks_dir, test_superpixel_dir, test_filenames)
                test_loader = DataLoader(test_dataset, batch_size=Config.TEST_BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

                dice, iou, acc = inference_model(model, test_loader, epochs=1, threod=threod_sim, fold=fold_num, cls_num=cls, ITERATION_STEPS = Config.inter_num)
                dice_list.append(dice)
                iou_list.append(iou)
                acc_list.append(acc)

                print(f"{cls}_batch_size = {Config.TEST_BATCH_SIZE},  test")

            mdice = np.mean(dice_list)
            mAcc = np.mean(acc_list)
            miou = np.mean(iou_list)

            print(f"threod_sim:{threod_sim:.4f},mdice:{mdice:.4f}, mAcc:{mAcc:.4f}, miou:{miou:.4f}")
            print(ckpt)
            print("fold", fold_num)
        
        torch.cuda.empty_cache()