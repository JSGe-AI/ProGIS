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
import matplotlib.pyplot as plt

from tqdm import tqdm 

from scipy.ndimage.morphology import distance_transform_edt
import scipy.ndimage as ndi
from skimage.morphology import skeletonize_3d


from skimage.measure import label as label_1
from skimage.measure import regionprops

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


###########################################################################################################
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
    output = torch.zeros(batch_size, 1, H, W, device=pred_mask_all.device, dtype=torch.float32)

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
            output[i,:] = generateGuidingSignal(largest_connected) if largest_connected.sum() > 0 else torch.zeros_like(fg, dtype=torch.float32, device=fg.device)  # 前景骨架信号
            output[i, output[i,:] != 0] = 1
        else:
            largest_connected = bg_largest
            output[i,:] = generateGuidingSignal(largest_connected) if largest_connected.sum() > 0 else torch.zeros_like(bg, dtype=torch.float32, device=bg.device)  # 背景骨架信号
            output[i, output[i,:] != 0] = -1
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

#############################################################################################################

def rearrange(*args, **kwargs):
    """
    Rearrange tensor axes according to a given pattern.

    Args:
        *args: A variable-length argument list of tensors to be rearranged.
        **kwargs: A variable-length keyword argument list of options for the rearrangement.

    Returns:
        The rearranged tensor.
    """
    return _rearrange(*args, **kwargs).contiguous()


class CustomDataset(Dataset):
    def __init__(self, h5_dir, signal_dir, mask_dir):
        self.h5_files = sorted([os.path.join(h5_dir, f) for f in os.listdir(h5_dir) if f.endswith('.h5')])
        self.signal_files = sorted([os.path.join(signal_dir, f) for f in os.listdir(signal_dir) if f.endswith('.npy')])
        self.mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.npy')])
        
        # 检查三个文件夹中的文件数量是否一致
        if len(self.h5_files) != len(self.signal_files) or len(self.h5_files) != len(self.mask_files):
            raise ValueError(f"File count mismatch: {len(self.h5_files)} h5 files, {len(self.signal_files)} signal files, {len(self.mask_files)} mask files. All directories must have the same number of files.")
    

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
        
        # 初始化 patch_embed 的权重
        patch_embed = torch.empty(num_ref_modes, mapped_dim)
        patch_embed = nn.init.xavier_normal_(patch_embed)
        self.patch_embed = nn.Parameter(patch_embed)
        
        # 定义 Q, K, V 的权重
        self.W_q = nn.Linear(mapped_dim, mapped_dim)
        self.W_k = nn.Linear(mapped_dim, mapped_dim)
        self.W_v = nn.Linear(mapped_dim, mapped_dim)
        
        # 分类头：用于二分类 (前景/背景) 的 1x1 卷积
        self.classification_head = nn.Conv2d(mapped_dim, 1, kernel_size=1)

    def forward(self, roi_signal, roi_features, roi_mask):
        
        
        roi_features_A = self.ref_embed(roi_features, roi_signal)
        batch_size, _, ROI_H, ROI_W = roi_mask.shape
        
        # 扁平化ROI区域特征
        roi_features_A_flat = roi_features_A.reshape(self.mapped_dim, -1).permute(1, 0)
        roi_features_O_flat = roi_features.reshape(self.mapped_dim, -1).permute(1, 0)
        
        # 计算Q, K, V
        Q = self.W_q(roi_features_A_flat)
        K = self.W_k(roi_features_O_flat)
        V = self.W_v(roi_features_O_flat)
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.mapped_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # 计算注意力输出
        attention_output = torch.matmul(attention_weights, V)
        
        # 将输出重塑为原始的ROI尺寸并放回到output_tensor中
        attention_output = attention_output.permute(1, 0).view(self.mapped_dim, ROI_H, ROI_W)
        # output_tensor[i, :, x_min:x_max+1, y_min:y_max+1] = attention_output
        attention_output = attention_output.unsqueeze(0)
        # 将 output_tensor 通过分类头获得分割图
        segmentation_map = self.classification_head(attention_output)  # (batch_size, 1, H, W)
        
        segmentation_map = segmentation_map.float()
        # 应用 sigmoid 激活函数以获得二分类结果
        segmentation_map = torch.sigmoid(segmentation_map)  # (batch_size, 1, H, W)

        '''
        similarity_fg = similarity_fg.squeeze(0)
        # similarity_fg[similarity_fg <= 0.8] = 0
        # 假设 signal_file 是一个包含文件路径字符串的元组
        signal_file_str = signal_file[0]  # 从元组中提取出字符串
        filename = signal_file_str.split('/')[-1]
        
        
        
        segmentation_map = segmentation_map.squeeze(0).squeeze(0)
        # 清理绘图
        plt.clf()
        plt.close('all')

        plt.imshow(similarity_fg.detach().cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title('Cosine Similarity Matrix')
        # 保存图片
        plt.savefig(f'/home/gjs/ISF_nuclick/WSI_ISF/keshihua_image/ESD_keshihua_image/patch_256_threod0.98/ROI_pred_{filename}.png')
        plt.show()
        
        # 清理绘图
        plt.clf()
        plt.close('all')
        similarity_fg = similarity_fg.unsqueeze(0)
        segmentation_map = segmentation_map.unsqueeze(0).unsqueeze(0)
        '''
        
        
        '''
        plt.imshow(roi_mask.detach().cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title('roi_mask')
        # 保存图片
        plt.savefig(f'/home/gjs/ISF_nuclick/keshihua_image/ESD_keshihua_image/ROI_mask/roi_mask_{filename}.png')
        plt.show()
        


        
        plt.clf()
        plt.close('all')
        
        plt.imshow(similarity_fg.detach().cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title('Cosine Similarity Matrix')
        # 保存图片
        plt.savefig(f'/home/gjs/ISF_nuclick/keshihua_image/ESD_keshihua_image/threod0.985_0.8_all_train/signal_proto_similarity_{filename}.png')
        plt.show()
        
        # 清理绘图
        plt.clf()
        plt.close('all')
        
        '''
        
    
        
        
        
        '''
        # 堆叠收集的特征以进行注意力计算
        coord_features = torch.stack(coord_features)  # (num_valid_coords, mapped_dim)

        # 初始化信号张量为全零
        signal_tensor = torch.zeros((batch_size, 1, H, W), device=mask.device)
    
        
        # 标记信号张量中的信号位置
        for i in range(batch_size):
            for j in range(num_signal_coords):
                x, y = signal_coords[i][j]
                if 0 <= x < H and 0 <= y < W:
                    signal_tensor[i, 0, x, y] = 1
        
        A_roi= A[i, :, x_min_exp[i]:x_max_exp[i]+1, y_min_exp[i]:y_max_exp[i]+1]
        
        
        # # 从 A 中提取与相同坐标对应的特征
        # coord_features_A = []
        # for i, x, y in coord_indices:
        #     coord_features_A.append(A[i, :, x, y])  # 从 A 中收集特征

        # # 堆叠从 A 收集的特征
        # coord_features_A = torch.stack(coord_features_A)  # (num_valid_coords, mapped_dim)

        # 生成 Q, K, V
        Q = self.W_q(coord_features_A)  # (num_valid_coords, mapped_dim)
        K = self.W_k(coord_features)  # (num_valid_coords, mapped_dim)
        V = self.W_v(coord_features)  # (num_valid_coords, mapped_dim)

        # 对收集的特征进行注意力机制计算
        attention_scores = torch.bmm(Q.unsqueeze(0), K.unsqueeze(0).transpose(1, 2)) / torch.sqrt(torch.tensor(self.mapped_dim, dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.bmm(attention_weights, V.unsqueeze(0)).squeeze(0)  # (num_valid_coords, mapped_dim)

        # 将 attention_output 重新分配回 output_tensor 中的正确位置
        for idx, (i, x, y) in enumerate(coord_indices):
            output_tensor[i, :, x, y] = attention_output[idx]'''
        
        # 将 output_tensor 通过分类头获得分割图
        # segmentation_map = self.classification_head(output_tensor)  # (batch_size, 1, H, W)
        
        # 应用 sigmoid 激活函数以获得二分类结果
        # segmentation_map = torch.sigmoid(segmentation_map)  # (batch_size, 1, H, W)

        return segmentation_map

    def ref_embed(self, x, ref_label):
        """
        :type x: torch.Tensor
        :type ref_label: torch.Tensor
        :rtype: torch.Tensor

        :param x: shape (batch_size, num_channels, down_height, down_width)
        :param ref_label: shape (batch_size, 1, height, width)
        :return: shape (batch_size, num_channels, down_height, down_width)
        """
        # Ensure ref_label is long type
        ref_label = ref_label.long()

        # Create the reference mask with one-hot encoding
        ref_mask = F.one_hot(ref_label, self.num_ref_modes)
        
        # Perform matrix multiplication with the patch embedding
        patch_embed = ref_mask.float() @ self.patch_embed
        
        # Rearrange patch_embed dimensions
        patch_embed = rearrange(patch_embed, 'b () h w c -> b c h w')
        
        # Resize patch_embed to match x if needed
        if patch_embed.size()[-2:] != x.size()[-2:]:
            patch_embed = F.interpolate(
                patch_embed, x.size()[-2:], mode='bilinear', align_corners=False)
        
        # Add the patch embeddings to the input x
        return x + patch_embed

    def min_max_normalize(self, matrix):
        min_val = torch.min(matrix)
        max_val = torch.max(matrix)
        normalized_matrix = (matrix - min_val) / (max_val - min_val + 1e-8)  # 防止除以 0
        return normalized_matrix
    
    def top_30(self, out_mask_clone):
        # 将张量展平为1D
        flattened_mask = out_mask_clone.view(-1)

        # 计算前30%元素的数量
        num_elements = flattened_mask.numel()  # 获取元素总数
        top_30_percent_count = int(0.3 * num_elements)  # 计算前30%的数量

        # 获取排序后的前30%的索引（根据值从大到小排序）
        _, top_indices = torch.topk(flattened_mask, top_30_percent_count)

        # 创建一个与展平后的掩码同样大小的零张量
        binary_mask = torch.zeros_like(flattened_mask)

        # 将前30%的位置设置为1
        binary_mask[top_indices] = 1

        # 将掩码重新调整为原始形状
        out_mask_clone = binary_mask.view_as(out_mask_clone)
        
        return out_mask_clone
    
    
def dice_score(y_true, y_pred, coordinates, a=1., b=1.):
    """
    Calculates the Dice coefficient for specified coordinates.

    Args:
        y_true (torch.Tensor): Ground truth tensor.
        y_pred (torch.Tensor): Predicted tensor.
        coordinates (torch.Tensor): Coordinates of the region of interest. Shape: (1, number_of_points, 2) where the last dimension represents (x, y) coordinates.
        a (float): Smoothing parameter (numerator).
        b (float): Smoothing parameter (denominator).

    Returns:
        float: Dice coefficient.
    """
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
        return float('nan')  # 如果类别在预测和实际中都不存在，忽略此类别
    else:
        return intersection / union

def compute_miou_binary(pred, target):
    # iou_background = compute_iou(pred, target, 0)
    iou_foreground = compute_iou(pred, target, 1)
    # miou = np.nanmean([iou_background, iou_foreground])
    return iou_foreground

def calculate_accuracy(pred, target):
    # coordinates should be of shape (1, number, 2)
    # Extract the (x, y) pairs from coordinates

    # Calculate TP, TN, FP, FN
    TP = ((pred == 1) & (target == 1)).sum().item()
    TN = ((pred == 0) & (target == 0)).sum().item()
    FP = ((pred == 1) & (target == 0)).sum().item()
    FN = ((pred == 0) & (target == 1)).sum().item()
    
    # Calculate ACC
    total_pixels = TP + TN + FP + FN
    acc = (TP + TN) / total_pixels if total_pixels > 0 else 0.1

    return acc

def cut_roi(coordinates, features, signal_coords, mask):
    batch_size, H, W = mask.shape
    num_coords = coordinates.shape[1]
    num_signal_coords = signal_coords.shape[1]
    
    # 初始化特征张量为全零
    output_tensor = torch.zeros((batch_size, 1024, H, W), device=mask.device)
    # 将坐标转换为整数索引
    coordinates = coordinates.long()
    
    flat_indices = (coordinates[..., 0] * W + coordinates[..., 1]).view(-1)
    flat_batch_indices = torch.arange(batch_size, device=mask.device).repeat_interleave(num_coords)
    output_tensor_flat = output_tensor.view(batch_size, 1024, -1)
    output_tensor_flat[flat_batch_indices, :, flat_indices] = features.view(-1, 1024)
    output_tensor = output_tensor_flat.view(batch_size, 1024, H, W)
    
                
    # 初始化信号张量为全零
    signal = torch.zeros((batch_size, 1, H, W), device=mask.device)
    # 标记信号张量中的信号位置
    for i in range(batch_size):
        for j in range(num_signal_coords):
            x, y = signal_coords[i][j]
            if 0 <= x < H and 0 <= y < W:
                signal[i, 0, x, y] = 1            

    
    # 将 signal_coords 转换为整数索引
    signal_coords = signal_coords.long()
    if signal_coords.size(1) > 0:  # 检查第二个维度的大小是否大于 0
        # 获取信号坐标的 ROI 方框
        x_min_1 = signal_coords[:, :, 0].min(dim=1)[0]
        x_max_1 = signal_coords[:, :, 0].max(dim=1)[0]
        y_min_1 = signal_coords[:, :, 1].min(dim=1)[0]
        y_max_1 = signal_coords[:, :, 1].max(dim=1)[0]
    else:
        # 处理 signal_coords 为空的情况
        # 例如，可以跳过当前迭代，或者设置默认值
        x_min_1 = torch.tensor(0, device=signal_coords.device, dtype=signal_coords.dtype) # or any other default value. Important to keep the same device and dtype
        x_max_1 = torch.tensor(0, device=signal_coords.device, dtype=signal_coords.dtype)
        y_min_1 = torch.tensor(0, device=signal_coords.device, dtype=signal_coords.dtype)
        y_max_1 = torch.tensor(0, device=signal_coords.device, dtype=signal_coords.dtype)
    
    # 计算中心坐标
    center_x = (x_min_1 + x_max_1) // 2
    center_y = (y_min_1 + y_max_1) // 2
    
    start_y = torch.tensor(max(center_y - 50, 0), dtype=torch.float)
    start_x = torch.tensor(max(center_x - 50, 0), dtype=torch.float)
    end_y = torch.tensor(start_y + 100, dtype=torch.float)
    end_x = torch.tensor(start_x + 100, dtype=torch.float)
    
    if end_y > W:
        end_y = torch.tensor(W, dtype=torch.float)
        start_y = torch.tensor(max(end_y - 100, 0), dtype=torch.float)  # readjust start_y to maintain 128x128 size
    if end_x > H:
        end_x = torch.tensor(H, dtype=torch.float)
        start_x = torch.tensor(max(end_x - 100, 0), dtype=torch.float) # readjust start_x to maintain 128x128 size
        
    # 将 x_min_exp, x_max_exp, y_min_exp, y_max_exp 转换为整数
    x_min = torch.floor(start_x).to(torch.int)
    x_max = torch.floor(end_x).to(torch.int)
    y_min = torch.floor(start_y).to(torch.int)
    y_max = torch.floor(end_y).to(torch.int)

    
    # print(x_min,x_max,y_min,y_max)
    roi_signal = signal[:, :, x_min:x_max, y_min:y_max]
    roi_features = output_tensor[:, :, x_min:x_max, y_min:y_max]
    roi_mask = mask[:, x_min:x_max, y_min:y_max]
    
    return roi_signal, roi_features, roi_mask

    


def preprocess_and_train(path, fold, train_h5_dir, train_signal_dir, train_mask_dir, val_h5_dir, val_signal_dir, val_mask_dir, num_epochs=30, learning_rate=1e-4, batch_size=1, device='cuda:1'):
    best_roi_dice = 0
    
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # 数据集和数据加载器
    train_dataset = CustomDataset(train_h5_dir, train_signal_dir, train_mask_dir)
    val_dataset = CustomDataset(val_h5_dir, val_signal_dir, val_mask_dir)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 8, pin_memory = True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers = 8, pin_memory = True)

    # _, features, _, _, _ = train_dataset[0]
    feature_dim = 1024

    # 模型、损失函数、优化器
    model = SimilarityModel(feature_dim).to(device)
    
    # pretrained_weights = '/home/gjs/ISF_nuclick/WSI_ISF/weight_pth/ESD_model_epoch_20.pth'
    # pretrained_weights = None
    # 如果提供了预训练权重，则加载
    # if pretrained_weights:
    #     model.load_state_dict(torch.load(pretrained_weights))
    #     print(f'Loaded pretrained weights from {pretrained_weights}')
    
    #criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 初始化GradScaler
    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_dice_scores_roi = []
        train_iou_scores_roi = []
        train_acc_scores_roi = []
        
        train_loader_tqdm = tqdm(train_dataloader, total=len(train_dataloader), desc=f'Epoch {epoch+1}/{num_epochs} [Training]', unit='batch')

        for coordinates, features, signal_coords, mask, signal_file in train_loader_tqdm:
            # 数据移动到GPU
            coordinates, features, signal_coords, mask = coordinates.to(device, non_blocking=True), features.to(device, non_blocking=True), signal_coords.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            coordinates = coordinates[:, :, [1, 0]]  #调转（H,W）
            roi_signal, roi_features, roi_mask = cut_roi(coordinates, features, signal_coords, mask)
            roi_signal[roi_signal == 1] = 2  
            roi_signal[roi_signal == 0] = 4 
            
            
            # unique_values = torch.unique(roi_signal)
            # num_categories = len(unique_values)
            # print(f"roi_signal 中有 {num_categories} 种类别.")
            roi_mask = roi_mask.unsqueeze(1).float()            
            optimizer.zero_grad()

            with autocast():  # 使用autocast进行混合精度训练
            # /tab
                roi_segmap_1 = model(roi_signal, roi_features, roi_mask)
                roi_seg_1 =  (roi_segmap_1 >= 0.5).float()

                
                signal_2 = processMasks(roi_seg_1, roi_mask)
                roi_seg_1[signal_2 == 1] = 2
                roi_seg_1[signal_2 == -1] = 3
                
                roi_seg_1[roi_signal == 2] = 2   # 添加上一步信号
                
                roi_segmap_2 = model(roi_seg_1, roi_features, roi_mask)
                roi_seg_2 =  (roi_segmap_2 >= 0.5).float()
                
                
                signal_3 = processMasks(roi_seg_2, roi_mask)
                roi_seg_2[signal_3 == 1] = 2
                roi_seg_2[signal_3 == -1] = 3
                
                roi_seg_2[signal_2 == 1] = 2   # 添加上一步信号
                roi_seg_2[signal_2 == -1] = 3  # 添加上一步信号
                roi_seg_2[roi_signal == 2] = 2  # 添加上一步信号
                
                # 条件 1: roi_seg_2 中值为 0，但 roi_seg_1 中对应位置值为 1
                condition1 = (roi_seg_2 == 0) & (roi_seg_1 == 1)
                roi_seg_2[condition1] = 4

                # 条件 2: roi_seg_2 中值为 1，但 roi_seg_1 中对应位置值为 0
                condition2 = (roi_seg_2 == 1) & (roi_seg_1 == 0)
                roi_seg_2[condition2] = 4
                
                roi_segmap_3 = model(roi_seg_2, roi_features, roi_mask)
                roi_seg_3 =  (roi_segmap_3 >= 0.5).float()
                
                
                loss = dice_loss(roi_segmap_1, roi_mask) + dice_loss(roi_segmap_2, roi_mask) + dice_loss(roi_segmap_3, roi_mask)
            # loss = dice_loss(roi_segmap_3, roi_mask)
            # /tab

            scaler.scale(loss).backward()  # 使用scaler.scale()进行反向传播
            scaler.step(optimizer)  # 更新模型参数
            scaler.update()  # 更新scaler
            
            # loss.backward()  # 使用标准的反向传播
            # optimizer.step()  # 更新模型参数
            
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
                # 数据移动到GPU
                coordinates, features, signal_coords, mask = coordinates.to(device, non_blocking=True), features.to(device, non_blocking=True), signal_coords.to(device, non_blocking=True), mask.to(device, non_blocking=True)
                coordinates = coordinates[:, :, [1, 0]]  #调转（H,W）
                roi_signal, roi_features, roi_mask = cut_roi(coordinates, features, signal_coords, mask)
                roi_signal[roi_signal == 1] = 2
                roi_signal[roi_signal == 0] = 4
                roi_mask = roi_mask.unsqueeze(1).float()
                with autocast():  # 使用autocast进行混合精度训练
                # /tab
                    roi_segmap_1 = model(roi_signal, roi_features, roi_mask)
                    roi_seg_1 =  (roi_segmap_1 >= 0.5).float()

                    
                    signal_2 = processMasks(roi_seg_1, roi_mask)
                    roi_seg_1[signal_2 == 1] = 2
                    roi_seg_1[signal_2 == -1] = 3
                    
                    roi_seg_1[roi_signal == 2] = 2   # 添加上一步信号
                    
                    roi_segmap_2 = model(roi_seg_1, roi_features, roi_mask)
                    roi_seg_2 =  (roi_segmap_2 >= 0.5).float()
                    
                    
                    signal_3 = processMasks(roi_seg_2, roi_mask)
                    roi_seg_2[signal_3 == 1] = 2
                    roi_seg_2[signal_3 == -1] = 3
                    
                    roi_seg_2[signal_2 == 1] = 2  # 添加上一步信号
                    roi_seg_2[signal_2 == -1] = 3 # 添加上一步信号
                    roi_seg_2[roi_signal == 2] = 2 # 添加上一步信号
                    
                    # 条件 1: roi_seg_2 中值为 0，但 roi_seg_1 中对应位置值为 1
                    condition1 = (roi_seg_2 == 0) & (roi_seg_1 == 1)
                    roi_seg_2[condition1] = 4

                    # 条件 2: roi_seg_2 中值为 1，但 roi_seg_1 中对应位置值为 0
                    condition2 = (roi_seg_2 == 1) & (roi_seg_1 == 0)
                    roi_seg_2[condition2] = 4
                    
                    roi_segmap_3 = model(roi_seg_2, roi_features, roi_mask)
                    roi_seg_3 =  (roi_segmap_3 >= 0.5).float()
                    
                    
                    loss = dice_loss(roi_segmap_1, roi_mask) + dice_loss(roi_segmap_2, roi_mask) + dice_loss(roi_segmap_3, roi_mask)
                # loss = dice_loss(roi_segmap_3, roi_mask)
                # /tab

                # scaler.scale(loss).backward()  # 使用scaler.scale()进行反向传播
                # scaler.step(optimizer)  # 更新模型参数
                # scaler.update()  # 更新scaler
            
                
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
            torch.save(model.state_dict(), f'{path}/Lung_best_model_fold_{fold}.pth')
            print(f'Model checkpoint saved at epoch {epoch + 1}')
            
            
        print(f'Epoch [{epoch + 1}/{num_epochs}], train_ROI_Dice: {train_mean_dice_roi:.4f} ,train_ROI Loss: {train_loss / len(train_dataloader):.4f}, train all Dice_roi: {train_mean_dice_roi:.4f}, train all Acc_roi: {train_mean_acc_roi:.4f}, train all mIOU_roi: {train_mean_iou_roi:.4f}')
        print(f'Epoch [{epoch + 1}/{num_epochs}], val_ROI_Dice: {val_mean_dice_roi:.4f} ,val_ROI Loss: {val_loss / len(val_dataloader):.4f}, val all Dice_roi: {val_mean_dice_roi:.4f}, val all Acc_roi: {val_mean_acc_roi:.4f}, val all mIOU_roi: {val_mean_iou_roi:.4f}')

    

# 示例调用
# h5_dir = '/data_nas2/gjs/ESD_ISF/patch_256/ESD_patch256_step128_features/h5_files_+64_down128'
# signal_dir = '/data_nas2/gjs/ESD_ISF/ESD_signals/class_0_0_255_max_XY'
# mask_dir = '/data_nas2/gjs/ESD_ISF/ESD_mask/class_0_0_255_down128_npy'

# train_h5_dir = '/data_nas2/gjs/Camelyon16/WSI_ISF/C16_tumor_features/h5_files_down256'
# train_signal_dir = '/data_nas2/gjs/Camelyon16/WSI_ISF/signals_max_XY'
# train_mask_dir = '/data_nas2/gjs/Camelyon16/WSI_ISF/C16_tumor_mask_npy'

fold_list = [4,5]
# path = "/data_nas2/gjs/Lung/UNI_5_fold"

path = "/data_993/gjs/UNI_5_fold"


for fold in fold_list:
    
    train_h5_dir = f'{path}/fold_{fold}/train/feature_h5'
    train_signal_dir = f'{path}/fold_{fold}/train/signals_max_XY'
    train_mask_dir = f'{path}/fold_{fold}/train/mask'

    val_h5_dir = f'{path}/fold_{fold}/test/feature_h5'
    val_signal_dir = f'{path}/fold_{fold}/test/signals_max_XY'
    val_mask_dir = f'{path}/fold_{fold}/test/mask'


    device = 'cuda:3'  # 使用第一块GPU

    print("fold",fold)
    preprocess_and_train(path, fold, train_h5_dir, train_signal_dir, train_mask_dir, val_h5_dir, val_signal_dir, val_mask_dir, device=device)