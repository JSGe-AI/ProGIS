def dice_loss(pred, target, smooth=1.):
    """
    Dice loss calculation for semantic segmentation.
    
    Args:
        pred (torch.Tensor): Model's output, shape (batch_size, num_classes, height, width).
        target (torch.Tensor): Ground truth labels, shape (batch_size, num_classes, height, width).
        smooth (float): Smoothing factor to avoid division by zero, default 1.
    
    Returns:
        torch.Tensor: Dice loss value.
    """
    num_classes = pred.size(1)
    
    dice = 0.
    for i in range(num_classes):
        pred_i = pred[:, i, :, :]  # (batch_size, height, width)
        target_i = target[:, i, :, :]  # (batch_size, height, width)
        intersection = (pred_i * target_i).sum(dim=(1, 2))
        union = pred_i.sum(dim=(1, 2)) + target_i.sum(dim=(1, 2))
        dice_i = (2. * intersection + smooth) / (union + smooth)
        dice += dice_i.mean()
    
    return 1. - dice / num_classes

"""
这个 dice_loss 函数的实现步骤如下:

获取输出 pred 的 shape,确定有多少个类别 num_classes。
遍历每个类别,对于每个类别:
从 pred 和 target 中提取该类别的 2D 张量。
计算该类别的交集和并集。
根据 Dice 系数公式计算该类别的 Dice 系数,并累加到 dice 变量中。
最后返回 1 - dice / num_classes 作为 Dice 损失。
这样设计的 Dice 损失函数可以很好地适用于这个分割模型的训练。通过最小化 Dice 损失,模型可以学习到更准确的分割结果。
"""