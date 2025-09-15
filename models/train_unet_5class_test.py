import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, jaccard_score
import numpy as np

from efficientunet import *
from UNet import *
import os


import torch
import torch.nn as nn

from torch.utils.data import DataLoader


import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy.ma as ma


class DynamicClassWeights:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.error_counts = torch.zeros(num_classes)  # 每个类别的错误计数
        self.sample_counts = torch.zeros(num_classes)  # 每个类别的样本总数

    def update(self, preds, targets):
        """
        更新错误计数和样本计数
        :param preds: 模型预测值，形状为 (batch_size, num_classes, H, W)
        :param targets: 标签值，形状为 (batch_size, H, W)
        """
        targets = targets-1
        preds = preds.argmax(dim=1)  # 获取每个像素的预测类别
        for cls in range(self.num_classes):
            mask = (targets == cls)
            self.sample_counts[cls] += mask.sum().item()
            self.error_counts[cls] += ((preds != targets) & mask).sum().item()

    def compute_weights(self):
        """
        根据错误率计算类别权重
        :return: 权重张量，形状为 (num_classes,)
        """
        weights = self.error_counts / (self.sample_counts + 1e-6)  # 避免除零
        return (weights / weights.sum()).to(torch.float32)  # 归一化


'''
# 定义多类分割的损失函数（结合交叉熵和Dice Loss）
class MultiClassSegmentationLoss(nn.Module):
    def __init__(self, weight=None, smooth=1e-5):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight)
        self.smooth = smooth

    def dice_loss(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target_one_hot = nn.functional.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
        union = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def forward(self, pred, target):
        target = target - 1
        loss_ce = self.cross_entropy(pred, target)
        loss_dice = self.dice_loss(pred, target)
        return 0.3*loss_ce + 0.7*loss_dice

'''

# 定义多类分割的损失函数（结合交叉熵, Dice Loss 和 Focal Loss）
class MultiClassSegmentationLoss(nn.Module):
    def __init__(self, weight=None, smooth=1e-5, gamma=4.0, alpha=0.75):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight)  # 可传入类别权重
        self.smooth = smooth
        self.gamma = gamma
        self.alpha = alpha

    def dice_loss(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
        union = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        num_classes = pred.shape[1]
        weights = torch.zeros(num_classes, device=pred.device, dtype=torch.float)

        for c in range(num_classes):
            class_pixels = (target == c).sum().float()
            weights[c] = 1.0 / (class_pixels + 1e-7)  # 避免除零

        # normalize weights
        weights /= weights.sum()

        weighted_dice = (dice * weights.unsqueeze(0)).mean()

        return 1 - weighted_dice
        # return 1 - dice.mean()
    '''
    def focal_loss(self, pred, target):
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        pt = torch.where(target_one_hot == 1, torch.softmax(pred, dim=1), 1 - torch.softmax(pred, dim=1))
        focal_weight = (1 - pt)**self.gamma
        loss = -focal_weight * F.log_softmax(pred, dim=1) * self.alpha # 加上alpha
        return loss.mean()
    '''
    
    def focal_loss(self, pred, target, alpha=None, gamma=4):
        """
        Focal Loss with per-class alpha and gamma.

        Args:
            pred: (B, C, H, W)  Predicted logits.
            target: (B, H, W) Ground truth labels.
            alpha: (Optional, list or tensor of length C) Per-class weighting factor.
            gamma: (Optional, float or list/tensor of length C) Focusing parameter.

        Returns:
            The calculated focal loss.
        """
        if alpha is None:
            # Calculate alpha based on class frequencies if not provided
            # class_pixels = [100804847, 91805721, 32732635, 16480868, 13766329] #你的类别像素数量
            class_pixels = [182791584, 167463064, 56947320, 30751392, 26303664] #你的类别像素数量
            alpha = [1/count for count in class_pixels] #计算倒数作为alpha
            alpha = torch.tensor(alpha).to(pred.device).float() #转为tensor并在GPU上计算
            alpha = alpha / alpha.sum() #归一化alpha
        else:
            if isinstance(alpha, list): #如果输入是list则转为tensor
                alpha = torch.tensor(alpha).to(pred.device).float()
            if not torch.is_tensor(alpha):
                raise TypeError("alpha must be a list or tensor")
            alpha = alpha / alpha.sum() #确保alpha归一化

        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # Use log_softmax and nll_loss for numerical stability
        logpt = F.log_softmax(pred, dim=1)  # numerically more stable than softmax followed by log
        pt = torch.exp(logpt) #softmax

        focal_weight = (1 - pt) ** gamma #计算focal weight

        # Apply alpha and calculate loss
        loss = -focal_weight * target_one_hot * logpt * alpha.view(1,-1,1,1)

        return loss.sum() / target_one_hot.sum() # normalize by the number of positive pixels

    def forward(self, pred, target):
        target = target - 1  # 确保目标类别从0开始
        loss_ce = self.cross_entropy(pred, target)
        loss_dice = self.dice_loss(pred, target)
        loss_focal = self.focal_loss(pred, target)
        return 0.3* loss_ce + 0.3 * loss_dice + 0.4 * loss_focal # 调整权重



# 计算评价指标：Dice、IoU、Acc

'''  # 第一种
def calculate_metrics(pred, target, num_classes):
    pred = torch.argmax(torch.softmax(pred, dim=1), dim=1).cpu().numpy()
    target = target.cpu().numpy()
    target = target - 1

    dice_list, iou_list, acc_list = [], [], []
    for cls in range(num_classes):
        pred_cls = (pred == cls).astype(np.int32)
        target_cls = (target == cls).astype(np.int32)

        intersection = np.sum(pred_cls * target_cls)
        union = np.sum(pred_cls) + np.sum(target_cls)
        dice = (2.0 * intersection) / (union + 1e-5)

        iou = intersection / (np.sum(pred_cls + target_cls) - intersection + 1e-5)
        acc = accuracy_score(target_cls.flatten(), pred_cls.flatten())

        dice_list.append(dice)
        iou_list.append(iou)
        acc_list.append(acc)

    return dice_list, iou_list, acc_list
'''

def calculate_metrics(pred, target, num_classes):
    pred = torch.argmax(torch.softmax(pred, dim=1), dim=1).cpu().numpy()
    target = target.cpu().numpy()
    target = target - 1

    dice_scores = []
    iou_scores = []
    acc_scores = []

    for i in range(pred.shape[0]):  # 遍历每张图片
        present_classes = np.unique(target[i]) # 获取当前图片中存在的类别
        for cls in range(num_classes):
            if cls in present_classes:
                pred_cls = (pred[i] == cls).astype(np.int32)
                target_cls = (target[i] == cls).astype(np.int32)
                intersection = np.sum(pred_cls * target_cls)
                union = np.sum(pred_cls) + np.sum(target_cls)
                dice = (2.0 * intersection) / (union + 1e-5)

                iou = intersection / (np.sum(pred_cls + target_cls) - intersection + 1e-5)
                acc = accuracy_score(target_cls.flatten(), pred_cls.flatten())
                
            else:
                dice, iou, acc = 0, 0, 0
                
            dice_scores.append(dice)
            iou_scores.append(iou)
            acc_scores.append(acc)

    return dice_scores, iou_scores, acc_scores

'''
# 第二种
def calculate_metrics(pred, target, num_classes):
    """
    Calculates Dice, IoU, and Accuracy for each class in a single image.

    Args:
        pred (torch.Tensor): Model predictions for a single image (num_classes, height, width).
        target (torch.Tensor): Ground truth labels for a single image (height, width).
        num_classes (int): Number of classes.

    Returns:
        tuple: Dice, IoU, and Accuracy for each class in the image.
    """
    pred = torch.argmax(torch.softmax(pred, dim=0), dim=0).cpu().numpy() # 注意这里改成dim=0
    target = target.cpu().numpy()
    target = target - 1

    image_dice, image_iou, image_acc = [], [], []
    present_classes = np.unique(target)  # Find present classes in the image

    for cls in range(num_classes):
        if cls in present_classes:
            pred_cls = (pred == cls).astype(np.int32)
            target_cls = (target == cls).astype(np.int32)

            intersection = np.sum(pred_cls * target_cls)
            union = np.sum(pred_cls) + np.sum(target_cls)
            dice = (2.0 * intersection) / (union + 1e-5)

            iou = intersection / (np.sum(pred_cls + target_cls) - intersection + 1e-5)
            acc = accuracy_score(target_cls.flatten(), pred_cls.flatten())
        else:
            dice, iou, acc = 0, 0, 0

        image_dice.append(dice)
        image_iou.append(iou)
        image_acc.append(acc)

    return image_dice, image_iou, image_acc
'''

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir,  filenames):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        # self.signal_dir = signal_dir
        # self.suppixel_dir = suppixel_dir
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # 根据索引获取当前文件名
        filename = self.filenames[idx]
        
        # 加载图像、掩模和超像素文件
        image_path = os.path.join(self.images_dir, filename)
        mask_path = os.path.join(self.masks_dir, filename)
        # signal_path = os.path.join(self.signal_dir, filename)
        # suppixel_path = os.path.join(self.suppixel_dir, filename)

        image = np.load(image_path)
        mask = np.load(mask_path)
        # signal = np.load(signal_path)
        # suppixel = np.load(suppixel_path)

        # 转换为 PyTorch 张量
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)  # (channels, height, width)
        mask = torch.tensor(mask, dtype=torch.float32)  #.unsqueeze(0)  # (1, height, width)
        # signal = torch.tensor(signal.transpose(2, 0, 1), dtype=torch.float32)  # (channels, height, width)
        # suppixel = torch.tensor(suppixel.transpose(2, 0, 1), dtype=torch.float32)  # (1, height, width)

        return image, mask
    
# 获取文件夹中的文件名
def get_filenames_from_folder(folder_path):
    return [filename for filename in os.listdir(folder_path) if filename.endswith('.npy')]    
    


i = 1
cls = 'Contrast_learning'
train_images_dir = f"/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/150/step256_patch512_filling/fold_{i}/train/{cls}/image_npy"
train_masks_dir = f"/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/150/step256_patch512_filling/fold_{i}/train/{cls}/mask_npy"
# train_signal_dir = f'/data_nas/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/150/step256_patch512_filling/fold_{i}/train/{cls}/signal_max_point_npy'


val_images_dir = f"/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/150/step256_patch512_filling/fold_{i}/val/{cls}/image_npy"
val_masks_dir = f"/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/150/step256_patch512_filling/fold_{i}/val/{cls}/mask_npy"
# val_signal_dir = f'/data_nas/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/150/step256_patch512_filling/fold_{i}/val/{cls}/signal_max_point_npy'

# 获取训练集和验证集的文件名
train_filenames = get_filenames_from_folder(train_images_dir)
val_filenames = get_filenames_from_folder(val_images_dir)

# 创建自定义数据集类的实例
train_dataset = CustomDataset(train_images_dir, train_masks_dir,  train_filenames)
val_dataset = CustomDataset(val_images_dir, val_masks_dir, val_filenames)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True ,  num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=1 ,shuffle=False,  num_workers=4)


# 定义模型、优化器和损失函数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=3, n_classes=5).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

class_pixels = np.array([182791584, 167463064, 56947320, 30751392, 26303664])  # 示例：四个类别的像素数量
total_pixels = np.sum(class_pixels)
# 计算类别权重：类别权重 = 总像素数 / (类别像素数 * 类别数)
weights = torch.tensor(total_pixels / (class_pixels * len(class_pixels)), dtype=torch.float32).to(device)
weights = weights / weights.sum()  # 归一化权重

# 手动调整权重：对小类别的权重进行放大
# adjustment_factor = torch.tensor([1.0, 1.0, 4.0, 8.0, 10.0], dtype=torch.float32).to(device)  # 根据需要调整
# weights *= adjustment_factor  # 增强小类别权重
# weights = weights / weights.sum()  # 再次归一化

loss_fn = MultiClassSegmentationLoss(weight=weights)


# 初始化动态权重计算器
# dynamic_weights = DynamicClassWeights(num_classes=5)

# 训练和验证循环
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.long().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    # # 更新损失函数权重
    # weights = dynamic_weights.compute_weights(outputs, masks)
    # loss_fn = MultiClassSegmentationLoss(weight=weights.to(device))

    return epoch_loss / len(loader)


def validate_one_epoch(model, loader, loss_fn, device, num_classes):
    model.eval()
    epoch_loss = 0
    all_dice, all_iou, all_acc = [], [], []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.long().to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)
            epoch_loss += loss.item()

            # 计算评价指标
            dice, iou, acc = calculate_metrics(outputs, masks, num_classes)
            all_dice.append(dice)
            all_iou.append(iou)
            all_acc.append(acc)

    # 计算每个类别的平均指标                   第一种
    # mean_dice = np.mean(all_dice, axis=0)
    # mean_iou = np.mean(all_iou, axis=0)
    # mean_acc = np.mean(all_acc, axis=0)
    
    # 计算每个类别的平均指标                   第二种
    all_images_dice = np.array(all_dice)
    all_images_iou = np.array(all_iou)
    all_images_acc = np.array(all_acc)

    masked_dice = ma.masked_equal(all_images_dice, 0)
    masked_iou = ma.masked_equal(all_images_iou, 0)
    masked_acc = ma.masked_equal(all_images_acc, 0)


    mean_dice = masked_dice.mean(axis=0)
    mean_iou = masked_iou.mean(axis=0)
    mean_acc = masked_acc.mean(axis=0)

    return epoch_loss / len(loader), mean_dice, mean_iou, mean_acc


# 训练主循环
num_epochs = 100
num_classes = 5
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
    val_loss, val_dice, val_iou, val_acc = validate_one_epoch(model, val_loader, loss_fn, device, num_classes)

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    for cls in range(num_classes):
        print(f"Class {cls}: Dice: {val_dice[cls]:.4f}, IoU: {val_iou[cls]:.4f}, Acc: {val_acc[cls]:.4f}")
