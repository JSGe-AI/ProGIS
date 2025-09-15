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

