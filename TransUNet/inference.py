#分割模型和相似度模型计算分成俩模型
#不对特征上采样，最后对预测的mask进行上采样
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F1
from torchvision import models
#from resnet import resnet50

#from config import config
# from models.losses import getLoss, dice_coef

import torch.optim as optim
from sklearn.model_selection import train_test_split

from scipy.ndimage.morphology import distance_transform_edt
import scipy.ndimage as ndi
from skimage.morphology import skeletonize_3d
from scipy.ndimage import label

import cv2
from skimage.segmentation import slic

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from skimage.draw import disk

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import argparse



# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device_id =  1# Specify which GPU to use, e.g., GPU 0
torch.cuda.set_device(device_id)

# Set the main device to GPU 0
# device_ids = 0   #[0, 1]
# device = torch.device('cuda')

multiGPU = False
learningRate = 4e-4
img_chls = 3
weight_decay = 5e-5
RandomizeGuidingSignalType='Skeleton'

##################################################   transunet   ###########################################


# 创建模型实例

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()





config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
config_vit.n_classes = 1
config_vit.n_skip = 3
if args.vit_name.find('R50') != -1:
    config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
transunet_model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
# transunet_model.load_from(weights=np.load(config_vit.pretrained_path))
############################################################################################################



###

###  backbone (可替换）

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
    
class ResNetUNet(nn.Module):
    def __init__(self, freeze=False, pretrained=True):
        super().__init__()
        self.freeze = freeze
    
        base_model = models.resnet18(pretrained=pretrained)
        self.base_layers = list(base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
#         self.layer0_1x1 = convbnrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
#         self.layer1_1x1 = convbnrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
#         self.layer2_1x1 = convbnrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
#         self.layer3_1x1 = convbnrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
#         self.layer4_1x1 = convbnrelu(512, 512, 1, 0)

#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
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
        
        # self.conv_original_size = convbnrelu(64, 64, 3, 1)
        
        # projection head
        self.conv_proh1 = nn.Conv2d(64, 64, 1)
        self.conv_proh2 = nn.Conv2d(64, 32, 1)
        self.conv_proh3 = nn.Conv2d(32, 32, 1)

        
    def forward(self, input ):
        

        
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

#         layer4 = self.layer4_1x1(layer4)
        x = self.upsample3(layer4)
#         layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up31(x)
        x = self.conv_up32(x)

        x = self.upsample2(x)
#         layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up21(x)
        x = self.conv_up22(x)

        x = self.upsample1(x)
#         layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up11(x)
        x = self.conv_up12(x)

        x = self.upsample0(x)
#         layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up01(x)
        x = self.conv_up02(x)

        x = self.upsample(x)
        # x = self.conv_original_size(x)  #x.size = (batch_size , channels 64, H , W)
        x = self.conv_proh1(x)
        x = self.conv_proh2(x)
        
        x = self.conv_proh3(x)
        

        return  x
    
###
class ResNetUNetHead(nn.Module):
    def __init__(self, freeze=True, pretrained=False):
        super().__init__()
        self.freeze = freeze

        base_model = models.resnet18(pretrained=pretrained)
        base_layers = list(base_model.children())

        self.layer0 = nn.Sequential(*base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
#         self.layer0_1x1 = convbnrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
#         self.layer1_1x1 = convbnrelu(64, 64, 1, 0)
        self.layer2 = base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
#         self.layer2_1x1 = convbnrelu(128, 128, 1, 0)
        self.layer3 = base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
#         self.layer3_1x1 = convbnrelu(256, 256, 1, 0)
        self.layer4 = base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
#         self.layer4_1x1 = convbnrelu(512, 512, 1, 0)

#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
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
        
        if self.freeze:
            with torch.no_grad():
                layer0 = self.layer0(input)
                layer1 = self.layer1(layer0)
                layer2 = self.layer2(layer1)
                layer3 = self.layer3(layer2)
                layer4 = self.layer4(layer3)
        else:
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


###  ROI区域分割
def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, actv='relu', use_bias=False, use_regularizer=False, do_batch_norm=True):
    layers = []
    padding = (kernel_size - 1) // 2 * dilation
    if use_regularizer:
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=use_bias, padding_mode='zeros'))
    else:
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=use_bias))
    if do_batch_norm:
        layers.append(BatchNorm2d(out_channels))
    if actv != 'None':
        if actv == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif actv == 'selu':
            layers.append(nn.SELU(inplace=True))
    return nn.Sequential(*layers)

class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sizes, dilatation_rates, is_dense=True):
        super(MultiScaleConvBlock, self).__init__()
        self.is_dense = is_dense
        if is_dense:
            self.conv0 = conv_bn_relu(in_channels, 4*out_channels, kernel_size=1)
        else:
            self.conv0 = nn.Identity()
        
        self.conv1 = conv_bn_relu(4*out_channels if is_dense else in_channels, out_channels, kernel_size=sizes[0], dilation=dilatation_rates[0])
        self.conv2 = conv_bn_relu(4*out_channels if is_dense else in_channels, out_channels, kernel_size=sizes[1], dilation=dilatation_rates[1])
        self.conv3 = conv_bn_relu(4*out_channels if is_dense else in_channels, out_channels, kernel_size=sizes[2], dilation=dilatation_rates[2])
        self.conv4 = conv_bn_relu(4*out_channels if is_dense else in_channels, out_channels, kernel_size=sizes[3], dilation=dilatation_rates[3])
        
        if is_dense:
            self.conv_out = conv_bn_relu(4*out_channels, out_channels, kernel_size=3)
    
    def forward(self, x):
        if self.is_dense:
            x = self.conv0(x)
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        conv3_out = self.conv3(x)
        conv4_out = self.conv4(x)
        output_map = torch.cat([conv1_out, conv2_out, conv3_out, conv4_out], dim=1)
        if self.is_dense:
            output_map = self.conv_out(output_map)
            output_map = torch.cat([x, output_map], dim=1)
        return output_map

class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, actv='relu', use_bias=False, use_regularizer=False, dilation=1):
        super(ResidualConv, self).__init__()
        self.actv = actv
        self.conv1 = conv_bn_relu(in_channels, out_channels, kernel_size, dilation=dilation, actv='None', use_bias=use_bias, use_regularizer=use_regularizer, do_batch_norm=True)
        self.conv2 = conv_bn_relu(out_channels, out_channels, kernel_size, dilation=dilation, actv='None', use_bias=use_bias, use_regularizer=use_regularizer, do_batch_norm=True)

        
    def forward(self, x):
        out_1 = self.conv1(x)
        out_2 = self.conv2(out_1)
        out = out_1 + out_2
        if self.actv == 'relu':
            out = F.relu(out)
        elif self.actv == 'selu':
            out = F.selu(out)
        return out

class MultiScaleResUnet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MultiScaleResUnet, self).__init__()
        self.conv1_1 = conv_bn_relu(in_channels, 64, kernel_size=7)
        self.conv1_2 = conv_bn_relu(64, 32, kernel_size=5)
        self.conv1_3 = conv_bn_relu(32, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.res2_1 = ResidualConv(32, 64)
        self.res2_2 = ResidualConv(64, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.res3_1 = ResidualConv(64, 128)
        self.msconv3 = MultiScaleConvBlock(128, 32, sizes=[3, 3, 5, 5], dilatation_rates=[1, 3, 3, 6], is_dense=False)
        self.res3_2 = ResidualConv(128, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.res4_1 = ResidualConv(128, 256)
        self.res4_2 = ResidualConv(256, 256)
        self.res4_3 = ResidualConv(256, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.res5_1 = ResidualConv(256, 512)
        self.res5_2 = ResidualConv(512, 512)
        self.res5_3 = ResidualConv(512, 512)
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        
        self.res51_1 = ResidualConv(512, 1024)
        self.res51_2 = ResidualConv(1024, 1024)
        
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.res6_1 = ResidualConv(1024, 512)
        self.res6_2 = ResidualConv(512, 256)
        
        self.up7 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.res7_1 = ResidualConv(512, 256)
        self.msconv7 = MultiScaleConvBlock(256, 64, sizes=[3, 3, 5, 5], dilatation_rates=[1, 3, 2, 3], is_dense=False)
        self.res7_2 = ResidualConv(256, 256)
        
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.res8_1 = ResidualConv(256, 128)
        self.res8_2 = ResidualConv(128, 128)
        
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.res9_1 = ResidualConv(128, 64)
        self.msconv9 = MultiScaleConvBlock(64, 16, sizes=[3, 3, 5, 7], dilatation_rates=[1, 3, 3, 6], is_dense=False)
        self.res9_2 = ResidualConv(64, 64)
        
        self.up10 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv10_1 = conv_bn_relu(64, 64)
        self.conv10_2 = conv_bn_relu(64, 32)
        self.conv10_3 = conv_bn_relu(32, 32)
        
        self.conv11 = nn.Conv2d(32, num_classes, kernel_size=1)
    
    def forward(self, x, aux_input):
        inputs = torch.cat([x, aux_input], dim=1)
        
        conv1 = self.conv1_1(inputs)
        conv1 = self.conv1_2(conv1)
        conv1 = self.conv1_3(conv1)
        pool1 = self.pool1(conv1)
        
        conv2 = self.res2_1(pool1)
        conv2 = self.res2_2(conv2)
        pool2 = self.pool2(conv2)
        
        conv3 = self.res3_1(pool2)
        conv3 = self.msconv3(conv3)
        conv3 = self.res3_2(conv3)
        pool3 = self.pool3(conv3)
        
        conv4 = self.res4_1(pool3)
        conv4 = self.res4_2(conv4)
        conv4 = self.res4_3(conv4)
        pool4 = self.pool4(conv4)
        
        conv5 = self.res5_1(pool4)
        conv5 = self.res5_2(conv5)
        conv5 = self.res5_3(conv5)
        pool5 = self.pool5(conv5)
        
        conv51 = self.res51_1(pool5)
        conv51 = self.res51_2(conv51)
        
        up6 = self.up6(conv51)
        up6 = torch.cat([up6, conv5], dim=1)
        conv6 = self.res6_1(up6)
        conv6 = self.res6_2(conv6)
        
        up7 = self.up7(conv6)
        up7 = torch.cat([up7, conv4], dim=1)
        conv7 = self.res7_1(up7)
        conv7 = self.msconv7(conv7)
        conv7 = self.res7_2(conv7)
        
        up8 = self.up8(conv7)
        up8 = torch.cat([up8, conv3], dim=1)
        conv8 = self.res8_1(up8)
        conv8 = self.res8_2(conv8)
        
        up9 = self.up9(conv8)
        up9 = torch.cat([up9, conv2], dim=1)
        conv9 = self.res9_1(up9)
        conv9 = self.msconv9(conv9)
        conv9 = self.res9_2(conv9)
        
        up10 = self.up10(conv9)
        up10 = torch.cat([up10, conv1], dim=1)
        conv10 = self.conv10_1(up10)
        conv10 = self.conv10_2(conv10)
        conv10 = self.conv10_3(conv10)
        
        conv11 = self.conv11(conv10)
        output = F.sigmoid(conv11)
        
        # min_val = torch.amin(output, dim=(2, 3), keepdim=True)[0]  # 找到每个通道的最小值
        # max_val = torch.amax(output, dim=(2, 3), keepdim=True)[0]  # 找到每个通道的最大值
        # normalized_output = (output - min_val) / (max_val - min_val + 1e-8)  # 防止除以0
        
        # 使用SLIC进行超像素分割
        # segments_slic = slic(x, n_segments = 100, sigma = 5)
        
        return output
        

### 相似度计算
    
class ResNetUNet_proto(nn.Module):
    def __init__(self, freeze=False, pretrained=True):
        super().__init__()
        self.freeze = freeze
        self.ResNetUNet = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)
        self.segment_part = MultiScaleResUnet(in_channels=5, num_classes=1)


        
    def forward(self, roi_input , roi_aux_input , roi_suppixel, input , aux_input, superpixels , masks):
        
        seg_orginal = self.segment_part(roi_input , roi_aux_input)
        
        x = self.ResNetUNet(input)
        
        
        # 对sigmoid_output进行阈值处理
        # bg_sigmoid_output = seg_orginal.clone()  # 复制一份，防止原始数据被修改
        fg_sigmoid_output = seg_orginal.clone()  # 复制一份，防止原始数据被修改
        
        # 将大于 0.2 的元素置零
        # 创建布尔掩码
        mask_greater_than_0_6 = fg_sigmoid_output > 0.5
        mask_less_equal_0_6 = fg_sigmoid_output <= 0.5

        # 使用掩码设置值
        fg_sigmoid_output[mask_greater_than_0_6] = 1
        fg_sigmoid_output[mask_less_equal_0_6] = 0
        
             
        ### 通过avgpooling,获得所有超像素块的特征
        
        # 初始化一个张量来存储每个超像素块的特征表示
        batch_size, channels, H, W = x.shape
        all_suppixel_labels = torch.unique(superpixels) 
        num_all_labels = torch.unique(superpixels).numel()
        # 获取 roi 区域中存在的超像素标签值，并将其转换为列表
        roi_suppixel_labels = torch.unique(roi_suppixel)

        all_superpixel_features = torch.zeros((batch_size, channels, num_all_labels), device=x.device)
        all_superpixel_counts = torch.zeros((batch_size, num_all_labels), device=x.device)
        
        for b in range(batch_size):
            for sp in all_suppixel_labels:
                sp = int(sp.item())  # 将张量转换为整数，便于后续处理
                mask = (superpixels[b] == sp).float()  # 将mask转换为float类型，用于乘法运算
                
                # 对应的x中的特征加到superpixel_features中
                selected_features = x[b] * mask  # x[b] 形状为 (channels, 512, 512)
                all_superpixel_features[b, :, sp] += selected_features.sum(dim=(1, 2))  # 对 (H, W) 维度求和
                all_superpixel_counts[b, sp] += mask.sum()  # 统计当前超像素块中的像素个数


        # 对每个超像素块进行平均池化
        all_superpixel_features /= (all_superpixel_counts.unsqueeze(1) + 1e-6)  # 防止除零错误
        
        ###

                
        # 初始化一个列表来存储所有batch的similarity_mask
        all_similarity_masks = []
        fg_proto_spp_features = []
        
        # 2. 找出在 fg_sigmoid_output 中前景占比超过 50% 的超像素块的标签，并计算这些超像素块的平均特征
        for b in range(batch_size):
            foreground_superpixels = []
            foreground_superpixel_features_sum = torch.zeros(channels, device=x.device)  # 初始化和
            foreground_superpixel_count = 0  # 记录符合条件的超像素块数量
            
            '''
            ###以点作指导信号
            # 第一步：提取 aux_input 第 i 个 batch 中第一个通道值为 1 的位置
            mask = aux_input[b, 0, :, :] == 1  # shape: (H, W)

            # 第二步：使用 mask 在 superpixels 中获取相应位置的标签
            sp = superpixels[b, 0, :, :][mask]  # shape: (num_selected_pixels,)
            foreground_superpixels.append(sp)
            
            for i in foreground_superpixels[0]:
                if i.numel() == 1:  # 确保 i 是单元素张量
                    i = int(i.item())  # 使用 .item() 提取张量中的标量值
                else:
                    print(f"i contains more than one element: {i}")
                foreground_superpixel_features_sum += all_superpixel_features[b, :, i]  # 累加特征
                foreground_superpixel_count += 1  # 增加符合条件的超像素块数量
            ###
            '''
            
            ###以线作为指导信号
            # 遍历所有存在的超像素标签值
            for sp in roi_suppixel_labels:
            # for sp in all_suppixel_labels:
                sp = int(sp.item())  # 将张量转换为整数，便于后续处理
                mask = (roi_suppixel[b] == sp).float()
                # mask = (superpixels[b] == sp).float()
                mask = mask.squeeze(0)  # 去掉多余的维度，使其形状为 (512, 512)
                foreground_ratio = fg_sigmoid_output[b, 0][mask.bool()].sum().float() / mask.sum().float()
                
                if foreground_ratio > 0.5:
                    foreground_superpixels.append(sp)
                    foreground_superpixel_features_sum += all_superpixel_features[b, :, sp]  # 累加特征
                    foreground_superpixel_count += 1  # 增加符合条件的超像素块数量
            ###
            
            ################ 计算原型相似度#############
            
            if foreground_superpixel_count > 0:
                # print("superpixels number: ",foreground_superpixel_count)    
                average_foreground_feature = foreground_superpixel_features_sum / foreground_superpixel_count  # 计算平均特征
                # 3. 计算 average_foreground_feature 与所有超像素块的余弦相似度
                cosine_similarities = torch.zeros(num_all_labels, device=x.device)
                for sp in all_suppixel_labels:
                    sp = int(sp.item())  # 将张量转换为整数，便于后续处理
                    cosine_similarities[sp] = F.cosine_similarity(all_superpixel_features[b, :, sp], average_foreground_feature, dim=0)
                 # 创建一个与输入图像相同尺寸的 mask
                similarity_mask = torch.zeros((H, W), device=x.device)

                # 将每个超像素块的相似度值赋给对应超像素块的每个像素
                for sp in all_suppixel_labels:
                    sp = int(sp.item())  # 将张量转换为整数，便于后续处理
                    current_mask = (superpixels[b, 0] == sp)  # 创建布尔掩码，形状为 (512, 512)
                    similarity_mask[current_mask] = cosine_similarities[sp]

                # 将生成的 similarity_mask 存储在列表中
                all_similarity_masks.append(similarity_mask.unsqueeze(0))  # 在第0维增加一个维度，方便之后的拼接
                fg_proto_spp_features.append(average_foreground_feature.unsqueeze(0))

            else:
                print(f"第 {b} 个 batch 中没有前景占比超过 50% 的超像素块")
                average_foreground_feature = torch.zeros((channels), device=x.device) # 计算平均特征
                similarity_mask = torch.zeros((H, W), device=x.device)
                all_similarity_masks.append(similarity_mask.unsqueeze(0))
                fg_proto_spp_features.append(average_foreground_feature.unsqueeze(0))    
                
              
        
        # 使用 torch.cat 将所有 batch 的 similarity_mask 拼接在一起
        out_mask = torch.cat(all_similarity_masks, dim=0)
        fg_proto_feature = torch.cat(fg_proto_spp_features, dim=0)
        
        # 计算out_mask的最小值和最大值
        min_val = out_mask.min()
        max_val = out_mask.max()

        # 最大最小归一化： (out_mask - min) / (max - min)
        out_mask_normalized = (out_mask - min_val) / (max_val - min_val + 1e-6)  # 避免除以0
        
        # 定义阈值列表，例如：[0.1, 0.3, 0.5, 0.7, 0.9]
        thresholds = [0.2,0.25,0.3,0.35,0.4,0.45,0.5]
        out_masks_list = []
        nuclick_out_list = []

        # 遍历每个阈值，生成相应的 out_mask 并保存到列表中
        for thresh in thresholds:
            out_mask_thresh = out_mask_normalized.clone()  # 创建一个 out_mask 的副本
            out_mask_thresh[out_mask_thresh <= thresh] = 0
            out_mask_thresh[out_mask_thresh > thresh] = 1
            
            out_mask_thresh = out_mask_thresh.unsqueeze(1)
            out_masks_list.append(out_mask_thresh)
            
            nuclick_out = seg_orginal.clone()
            nuclick_out[nuclick_out <=0.5] = 0
            nuclick_out[nuclick_out > 0.5] = 1
            nuclick_out_list.append(nuclick_out)

        return all_superpixel_features, out_masks_list, nuclick_out_list, fg_proto_feature

        # return  x , out_mask , seg_orginal , fg_proto_feature 
    
    
def ROI_crop(input, aux_input, superpixel, mask):
    # 假设 input 和 aux_input 的形状分别为 (batch_size, 3, H, W) 和 (batch_size, 2, H, W)
    batch_size, _, H, W = input.shape
    
    # 创建空列表以存储裁剪后的 ROI
    roi_inputs = []
    roi_aux_inputs = []
    roi_suppixels = []
    roi_masks = []

    # 遍历 batch 中的每个样本
    for b in range(batch_size):
        # 找到 aux_input 中像素值为 1 的所有位置
        indices = (aux_input[b, 0] == 1).nonzero(as_tuple=True)

        # 如果找到指导信号点
        if indices[0].numel() > 0:
            # 计算中心点（x 和 y 的平均值）
            center_y = indices[0].float().mean().round().long().item()
            center_x = indices[1].float().mean().round().long().item()

            # 计算裁剪的边界
            start_y = max(center_y - 128, 0)
            start_x = max(center_x - 128, 0)
            end_y = min(start_y + 256, H)
            end_x = min(start_x + 256, W)

            # 调整起始位置，以保持裁剪区域的大小为 256x256
            if end_y - start_y < 256:
                start_y = max(end_y - 256, 0)
            if end_x - start_x < 256:
                start_x = max(end_x - 256, 0)
        else:
            # print("signal 为空！")
            # 随机生成裁剪的起始点，确保不会超出边界
            start_y = torch.randint(0, max(H - 256, 1), (1,)).item()
            start_x = torch.randint(0, max(W - 256, 1), (1,)).item()
            end_y = start_y + 256
            end_x = start_x + 256
            
        # 裁剪 input 和 aux_input
        roi_input = input[b, :, start_y:end_y, start_x:end_x]
        roi_aux_input = aux_input[b, :, start_y:end_y, start_x:end_x]
        roi_suppixel = superpixel[b, :, start_y:end_y, start_x:end_x]
        roi_mask = mask[b, :, start_y:end_y, start_x:end_x]

        roi_inputs.append(roi_input)
        roi_aux_inputs.append(roi_aux_input)
        roi_suppixels.append(roi_suppixel)
        roi_masks.append(roi_mask)
        

    # 将裁剪后的列表转换为张量
    roi_inputs = torch.stack(roi_inputs) if roi_inputs else None
    roi_aux_inputs = torch.stack(roi_aux_inputs) if roi_aux_inputs else None
    roi_suppixels = torch.stack(roi_suppixels) if roi_suppixels else None
    roi_masks = torch.stack(roi_masks) if roi_masks else None
    
    
    return roi_inputs, roi_aux_inputs, roi_suppixels, roi_masks







### 评价指标

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
        return float('nan')  # 如果类别在预测和实际中都不存在，忽略此类别
    else:
        return intersection / union

def compute_miou_binary(pred, target):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    # 计算背景类别（0）的IOU
    iou_background = compute_iou(pred, target, 0)
    # 计算前景类别（1）的IOU
    iou_foreground = compute_iou(pred, target, 1)
    
    # 计算mIOU，忽略nan值
    miou = np.nanmean([iou_background, iou_foreground])
    return miou

def calculate_binary_segmentation_accuracy(preds, labels):
    """
    计算前景背景分割的像素级准确率
    :param preds: 模型的预测值，形状为 [batch_size, height, width] 或 [batch_size, 1, height, width]
    :param labels: 真实标签，形状为 [batch_size, 1, height, width]
    :return: 每个样本的准确率和平均准确率
    """
    if preds.dim() == 4:
        preds = preds.squeeze(1)  # 去掉频道维度
    
    if labels.dim() == 4:
        labels = labels.squeeze(1)  # 去掉频道维度
    
    assert preds.shape == labels.shape, "预测值和标签的形状必须一致"

    preds = (preds > 0.5).float()  # 阈值 0.5，用于二分类
    
    correct = (preds == labels).float().sum(dim=[1, 2])  # 每个样本的正确预测像素数
    total_pixels_per_sample = labels.size(1) * labels.size(2)
    
    accuracy_per_sample = correct / total_pixels_per_sample  # 每个样本的准确率
    mean_accuracy = accuracy_per_sample.mean().item()  # 批次中的平均准确率
    
    return accuracy_per_sample, mean_accuracy








'''
# Define the dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, aux_inputs, masks, superpixels , filenames):
        self.images = images
        self.aux_inputs = aux_inputs
        self.masks = masks
        self.superpixels = superpixels
        self.filenames = filenames
        
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].transpose(2, 0, 1)  # 转换成 (channels, height, width)
        aux_input = self.aux_inputs[idx].transpose(2, 0, 1)  # 转换成 (channels, height, width)
        mask = torch.tensor(self.masks[idx], dtype=torch.float32)
        superpixels = self.superpixels[idx].transpose(2, 0, 1)  # 转换成 (channels, height, width)
        filename = self.filenames[idx]
        
        
        # Ensure mask has an additional dimension at the end
        mask = mask.unsqueeze(-1)
        # Transpose mask to (height, width, 1) and then to (1, height, width)
        mask = mask.permute(0, 1, 2).permute(2, 0, 1)
        
        image = torch.tensor(image, dtype=torch.float32)
        aux_input = torch.tensor(aux_input, dtype=torch.float32)
        superpixels = torch.tensor(superpixels, dtype=torch.float32)
        
        return image, aux_input, mask, superpixels , filename

# Load your data here
# images, aux_inputs, masks should be numpy arrays or lists containing your dataset
# images: (N, 3, H, W), aux_inputs: (N, 3, H, W), masks: (N, H, W)

# 从文件夹中加载数据
def load_data_from_folder(folder_path, filenames_list):
    data = []
    filenames = []
    for filename in filenames_list:
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            if os.path.exists(file_path):
                array = np.load(file_path)
                data.append(array)
                filenames.append(filename)
            else:
                continue
                # print(f"File not found and skipped: {file_path}")
    return data, filenames



# 从文本文件中读取文件名
def load_filenames_from_txt(file_path):
    with open(file_path, 'r') as f:
        filenames = f.read().splitlines()
    return filenames


# 文件夹路径
images_dir = "/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/TEST_1_no_tianchong/tumor_1/image_npy"
masks_dir = "/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/TEST_1_no_tianchong/tumor_1/mask_npy"
aux_inputs_dir = "/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/TEST_1_no_tianchong/tumor_1/signal_max_point_npy"
superpixel_dir = '/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/TEST_1_no_tianchong/tumor_1/image_SLIC_1000'

# 训练和验证文件名列表
train_filename_txt = "/home/gjs/ISF_nuclick/Z_data_sets/new/BCSS_X10_reinhard_allclass_notianchong/train_filenames_1.txt"
val_filename_txt = "/home/gjs/ISF_nuclick/Z_data_sets/new/BCSS_X10_reinhard_allclass_notianchong/val_filenames_1.txt"

train_filenames = load_filenames_from_txt(train_filename_txt)
val_filenames = load_filenames_from_txt(val_filename_txt)

# 加载训练集数据
train_images, train_imgname = load_data_from_folder(images_dir, train_filenames)
train_aux_inputs, _ = load_data_from_folder(aux_inputs_dir, train_filenames)
train_masks, _ = load_data_from_folder(masks_dir, train_filenames)
train_superpixels, _ = load_data_from_folder(superpixel_dir, train_filenames)

# 加载验证集数据
val_images, val_imgname = load_data_from_folder(images_dir, val_filenames)
val_aux_inputs, _ = load_data_from_folder(aux_inputs_dir, val_filenames)
val_masks, _ = load_data_from_folder(masks_dir, val_filenames)
val_superpixels, _ = load_data_from_folder(superpixel_dir, val_filenames)

# 转换为 numpy 数组
train_images = np.array(train_images)
train_aux_inputs = np.array(train_aux_inputs)
train_masks = np.array(train_masks)
train_superpixels = np.array(train_superpixels)

val_images = np.array(val_images)
val_aux_inputs = np.array(val_aux_inputs)
val_masks = np.array(val_masks)
val_superpixels = np.array(val_superpixels)

# 创建 dataloaders
train_dataset = CustomDataset(train_images, train_aux_inputs, train_masks, train_superpixels, train_imgname)
val_dataset = CustomDataset(val_images, val_aux_inputs, val_masks, val_superpixels, val_imgname)
'''

###################################################

# Define the dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, signal_dir, masks_dir, suppixel_dir,  filenames):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.signal_dir = signal_dir
        self.suppixel_dir = suppixel_dir
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # 根据索引获取当前文件名
        filename = self.filenames[idx]
        
        # 加载图像、掩模和超像素文件
        image_path = os.path.join(self.images_dir, filename)
        mask_path = os.path.join(self.masks_dir, filename)
        signal_path = os.path.join(self.signal_dir, filename)
        suppixel_path = os.path.join(self.suppixel_dir, filename)

        image = np.load(image_path)
        mask = np.load(mask_path)
        signal = np.load(signal_path)
        suppixel = np.load(suppixel_path)

        # 转换为 PyTorch 张量
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)  # (channels, height, width)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # (1, height, width)
        
        signal = torch.tensor(signal.transpose(2, 0, 1), dtype=torch.float32)  # (channels, height, width)
        # suppixel = torch.tensor(suppixel, dtype=torch.float32).unsqueeze(0)
        suppixel = torch.tensor(suppixel.transpose(2, 0, 1), dtype=torch.float32)  # (1, height, width)

        return image, signal, mask, suppixel, filename
    
# 获取文件夹中的文件名
def get_filenames_from_folder(folder_path):
    return [filename for filename in os.listdir(folder_path) if filename.endswith('.npy')]   
 

# 设置训练集和验证集的文件夹路径
train_images_dir = "/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/Train_val_step256_no_filling/train/tumor/image_npy"
train_masks_dir = "/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/Train_val_step256_no_filling/train/tumor/mask_npy"
train_signal_dir = '/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/Train_val_step256_no_filling/train/tumor/signal_max_point_npy'
train_superpixel_dir = '/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/Train_val_step256_no_filling/train/tumor/image_SLIC_600'

val_images_dir = "/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/Train_val_step256_no_filling/val/tumor/image_npy"
val_masks_dir = "/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/Train_val_step256_no_filling/val/tumor/mask_npy"
val_signal_dir = "/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/Train_val_step256_no_filling/val/tumor/signal_max_point_npy"
val_superpixel_dir = '/data_nas2/gjs/ISF_pixel_level_data/BCSS_x10_reinhard_cut/Train_val_step256_no_filling/val/tumor/image_SLIC_600'

# 获取训练集和验证集的文件名
train_filenames = get_filenames_from_folder(train_images_dir)
val_filenames = get_filenames_from_folder(val_images_dir)

# 创建自定义数据集类的实例
train_dataset = CustomDataset(train_images_dir, train_signal_dir, train_masks_dir, train_superpixel_dir, train_filenames)
val_dataset = CustomDataset(val_images_dir, val_signal_dir, val_masks_dir, val_superpixel_dir, val_filenames)



#####################################################






train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)



# 创建第二个模型实例
model = ResNetUNet_proto()


# ResNetUNet_state_dict = torch.load('/home/gjs/ISF_nuclick/checkpoints/ResUNet_cocoval_bcss_x10_reinhard/resunet_cocoval_x10_reinhard_best.pth')
ResNetUNet_state_dict = torch.load('/home/gjs/ISF_nuclick/checkpoints_new/TransUNet/transunet_nofilling_best_2.pth')
# ResNetUNet_state_dict = torch.load('/home/gjs/ISF_nuclick/checkpoints_new/ResUNet_BCSS_X10_reinhard_notianchong_withbg/resunet_allclass_x10_reinhard_best.pth', map_location=torch.device('cpu'))



# 去除 'module.' 前缀
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in ResNetUNet_state_dict.items():
    # 去掉 `module.` 前缀
    if k.startswith('module.'):
        k = k[7:]
    new_state_dict[k] = v
    

# 加载修改后的权重到模型
model.ResNetUNet.load_state_dict(new_state_dict)
# model.load_state_dict(new_state_dict)

model.segment_part.load_state_dict(torch.load('/home/gjs/ISF_nuclick/checkpoints_new/Nuclick_256_nofilling/nuclick_256_nofilling_tumor_1_best.pth'))

# model.load_state_dict(torch.load('/home/gjs/ISF_nuclick/checkpoints/resunet_suppixel_best_model_test.pth'))

# 冻结第一个模型部分的参数
for param in model.segment_part.parameters():
    param.requires_grad = False 

for param in model.ResNetUNet.parameters():
    param.requires_grad = False 


if multiGPU:
    model = nn.DataParallel(model, device_ids=[0, 1])

# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=4e-4)


# 使用 BCEWithLogitsLoss 作为损失函数
criterion = nn.BCEWithLogitsLoss()

# Training function
def train_model(model, train_loader, val_loader, epochs=50):
    best_dice = 0.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    for epoch in range(epochs):
        model.train()
        # train_loss = 0.0
        train_dice_score = [0.0 for _ in range(8)]
        nuclick_train_dice = [0.0 for _ in range(8)]
        train_accuracy = [0.0 for _ in range(8)]  
        nuclick_train_acc = [0.0 for _ in range(8)]
        
        '''
        for item,(images, aux_inputs, masks, superpixels, filenames) in enumerate(train_loader):
            
            count = 0  #统计迭代次数
            
            images, aux_inputs, masks, superpixels = images.to(device), aux_inputs.to(device), masks.to(device), superpixels.to(device)
            ##optimizer.zero_grad()

            roi_input , roi_aux_input , roi_suppixel , roi_mask = ROI_crop(images , aux_inputs, superpixels, masks)
            


            # outputs, all_superpixel_features = model(images, aux_inputs, superpixels)
            all_superpixel_features , out_masks_list ,nuclick_out , fg_proto_features = model(roi_input , roi_aux_input , roi_suppixel, images, aux_inputs, superpixels, masks)
            
            
            ###  可视化特征分布
            
            out_mask = out_masks_list[3]

            # 假设 all_superpixel_features 的 shape 是 (channels, num_superpixels)
            batch_size, channels, num_superpixels = all_superpixel_features.shape
            for b in range(batch_size): 
                
                target_sps = superpixels[b,0,...] # w, h
        
                sps_labels_list = torch.unique(target_sps)
                
                # 将 masks 和 superpixels 展平
                mask_flat = masks[b].view(-1)  # shape: (H*W,)
                out_mask_flat = out_mask[b].view(-1)
                
                superpixels_flat = superpixels[b].view(-1)  # shape: (H*W,)

                fg_sps_labels_list = []
                fg_sps_labels_list_pre = []

                for label in sps_labels_list:
                    # 获取该超像素块中所有像素的掩码值
                    mask_for_superpixel = mask_flat[superpixels_flat == label]

                    # 计算前景和背景像素的数量
                    num_fg_pixels = torch.sum(mask_for_superpixel)
                    num_bg_pixels = mask_for_superpixel.numel() - num_fg_pixels

                    if num_fg_pixels > num_bg_pixels:
                        fg_sps_labels_list.append(label)
                    else:
                        continue
                    
                for label in sps_labels_list:
                    # 获取该超像素块中所有像素的掩码值
                    out_mask_for_superpixel = out_mask_flat[superpixels_flat == label]

                    # 计算前景和背景像素的数量
                    num_fg_pixels = torch.sum(out_mask_for_superpixel)
                    num_bg_pixels = out_mask_for_superpixel.numel() - num_fg_pixels

                    if num_fg_pixels > num_bg_pixels:
                        fg_sps_labels_list_pre.append(label)
                    else:
                        continue
                
                
                filename = filenames[b]
                # 转置特征矩阵，使其变成 (num_superpixels, channels)
                features_transposed = all_superpixel_features[b].T  # shape: (num_superpixels, channels)
                features_transposed = features_transposed.detach().cpu().numpy()

                # 获取第 b 张图片的原型特征
                prototype_feature = fg_proto_features[b].reshape(1, -1)  # shape: (1, channels)
                prototype_feature = prototype_feature.detach().cpu().numpy()

                # 将原型特征与超像素特征拼接，进行统一的降维
                all_features = np.vstack([features_transposed, prototype_feature])  # shape: (num_superpixels + 1, channels)

                # 选择降维方法 (PCA 或 t-SNE)
                tsne = TSNE(n_components=2, random_state=42)
                reduced_features_tsne = tsne.fit_transform(all_features)

                # 创建颜色列表，默认全为蓝色
                colors = ['blue'] * num_superpixels
                # 将前景超像素编号对应的颜色设置为绿色
                # 处理 fg_sps_labels_list 和 fg_sps_labels_list_pre 的情况
                for sp_idx in sps_labels_list:
                    if sp_idx in fg_sps_labels_list and sp_idx in fg_sps_labels_list_pre:
                        colors[sp_idx] = 'brown'  # 同时在两个列表中，置为棕色
                    elif sp_idx in fg_sps_labels_list:
                        colors[sp_idx] = 'green'  # 只在 fg_sps_labels_list 中，置为绿色
                    elif sp_idx in fg_sps_labels_list_pre:
                        colors[sp_idx] = 'black'  # 只在 fg_sps_labels_list_pre 中，置为黑色


                # 添加原型特征的颜色（红色）
                colors.append('red')  # 原型特征的位置在最后
                
                # 绘制 t-SNE 可视化
                plt.figure(figsize=(8, 6))
                plt.scatter(reduced_features_tsne[:-1, 0], reduced_features_tsne[:-1, 1], c=colors[:-1], alpha=0.7, s=5, label='Superpixels')
                plt.scatter(reduced_features_tsne[-1, 0], reduced_features_tsne[-1, 1], c='red', alpha=1.0, s=50, label='Prototype Feature')
                plt.title('t-SNE of Superpixel and Prototype Features')
                plt.xlabel('Dimension 1')
                plt.ylabel('Dimension 2')
                plt.legend(loc='best')
                plt.savefig(f'/home/gjs/ISF_nuclick/X_ISF/keshihua_image/t-SNE_197/tumor_train_predraw/{filename}.png')
                plt.clf()
                plt.close('all')
                
            
            
            
            # first_seg = model(images, aux_inputs, superpixels)
            for i in range(len(out_masks_list)):

            
                train_dice_score[i] += dice_coeff(out_masks_list[i], masks).item() * images.size(0)
                nuclick_train_dice[i] += dice_coeff(nuclick_out[i], roi_mask).item() * images.size(0)
                
                # 计算准确率
                _, batch_accuracy = calculate_binary_segmentation_accuracy(out_masks_list[i], masks)
                train_accuracy[i] += batch_accuracy * images.size(0)  # 将每个批次的准确率加权累积
                
                _, nuclick_batch_accuracy = calculate_binary_segmentation_accuracy(nuclick_out[i], roi_mask)
                nuclick_train_acc[i] += nuclick_batch_accuracy * images.size(0)  # 将每个批次的准确率加权累积
            
            
            
            
        # 对每个列表中的每个元素除以 len(train_loader.dataset)
        dataset_len = len(train_loader.dataset)

        for i in range(len(train_dice_score)):
            train_dice_score[i] /= dataset_len
            nuclick_train_dice[i] /= dataset_len
            train_accuracy[i] /= dataset_len
            nuclick_train_acc[i] /= dataset_len
        '''
        
        ###
        model.eval()
        
        val_dice_score = [0.0 for _ in range(7)]
        nuclick_val_dice = [0.0 for _ in range(7)]
        val_accuracy = [0.0 for _ in range(7)]  
        nuclick_val_acc = [0.0 for _ in range(7)]
        mean_IoU = [0.0 for _ in range(7)]
        mean_IoU_list = [[],[],[],[],[],[],[]]
        
        

        with torch.no_grad():
            
            for images, aux_inputs, masks, superpixels, filenames in val_loader:
                images, aux_inputs, masks, superpixels= images.to(device), aux_inputs.to(device), masks.to(device), superpixels.to(device)
                # outputs, all_superpixel_features = model(images, aux_inputs, superpixels)
                roi_input , roi_aux_input , roi_suppixel , roi_mask = ROI_crop(images , aux_inputs, superpixels, masks)
            

                
                # outputs, all_superpixel_features = model(images, aux_inputs, superpixels)
                all_superpixel_features , out_masks_list ,nuclick_out , fg_proto_features = model(roi_input , roi_aux_input , roi_suppixel, images, aux_inputs, superpixels, masks)
                
                #############################################  迭代  ############################################
                
                
                count = 10
                
                while count < 1 :
                    out_put = out_masks_list[3]
                

                    # 计算准确率
                    _, batch_accuracy = calculate_binary_segmentation_accuracy(out_put, masks)
                    # val_accuracy += batch_accuracy * images.size(0)  # 将每个批次的准确率加权累积
                    # print("矫正前的准确率：",batch_accuracy)
                    # print("文件名：", filenames)
                    
                    if batch_accuracy < 0.8 :
                        
                        
                        # print("矫正前的准确率：",batch_accuracy)
                        
                        
                        # 假设 outputs 和 masks 的形状都是 (batch_size, 1, H, W)
                        batch_size, _, H, W = out_put.shape
                        
                        out_mask_thresh = out_put  # 创建一个 out_mask 的副本



                        for b in range(batch_size):
                            
                            roi_inputs = []
                            roi_aux_inputs = []
                            roi_suppixels = []
                            coord_list = []  # 存储坐标的列表
                            
                            # 将 outputs 和 masks 转为 NumPy 数组进行连通域分析
                            output_np = out_mask_thresh[b, 0].detach().cpu().numpy()  # 取出单个 batch 的 output，转为 NumPy 格式
                            mask_np = masks[b, 0].detach().cpu().numpy()      # 同样取出对应的 mask

                            # 找出 masks 为 0 且 outputs 为 1 的区域
                            region = (output_np == 0) & (mask_np == 1)
                            region_bg = (output_np == 1) & (mask_np == 0)

                            # 对这些区域执行连通域标记
                            labeled_region, num_features = ndi.label(region)
                            labeled_region_bg, num_features_bg = ndi.label(region_bg)

                            if num_features > 0:
                                # 找到每个连通域的面积（像素数）
                                region_sizes = ndi.sum(region, labeled_region, range(1, num_features + 1))
                                
                                # 找到最大连通域的标签
                                max_region_label = np.argmax(region_sizes) + 1
                                
                                # 找到最大连通域的坐标
                                max_region_coords = np.column_stack(np.where(labeled_region == max_region_label))
                                
                                # 计算最大连通域的中心坐标
                                center_coord = max_region_coords.mean(axis=0).astype(int)
                                
                                # 生成与 masks 形状相同的全零张量
                                Iteration_Signal = torch.zeros_like(masks)
                                # 将这个中心坐标在 zero_tensor 中置为 1
                                Iteration_Signal[b, 0, center_coord[0], center_coord[1]] = 1
                                mask_signal = Iteration_Signal[b, 0, :, :] == 1  # shape: (H, W)
                                sp = superpixels[b, 0, :, :][mask_signal]  # shape: (num_selected_pixels,)
                                sp = torch.tensor(sp, dtype=torch.long)
                                
                                
                                # 获取中心点坐标所在的超像素块编号
                                center_sp_id = sp.item()
                                
                                # 获取该超像素块的特征作为原型特征
                                fg_proto_features = all_superpixel_features[b, :, center_sp_id]
    
                                
                                if batch_accuracy < 0.5 :
                                    
                                    # 计算裁剪的边界，中心坐标为正方形框的中心
                                    start_y = max(center_coord[0] - 128, 0)
                                    start_x = max(center_coord[1] - 128, 0)
                                    end_y = min(start_y + 256, H)
                                    end_x = min(start_x + 256, W)

                                    # 调整起始位置，以保持裁剪区域的大小为 100x100
                                    if end_y - start_y < 256:
                                        start_y = max(end_y - 256, 0)
                                    if end_x - start_x < 256:
                                        start_x = max(end_x - 256, 0)
                        
                                    # 裁剪 superpixels 区域
                                    roi_suppixel_1 = superpixels[b, :, start_y:end_y, start_x:end_x]
                                
                                    
                                    # 裁剪 input 和 aux_input
                                    roi_input_1 = images[b, :, start_y:end_y, start_x:end_x]
                                    # 生成 roi_signal
                                    roi_signal = np.zeros((2, H, W), dtype=np.float32)
                                    # 在第一个通道生成一个半径为2的圆盘
                                    rr, cc = disk(center_coord, 2, shape=(H, W))
                                    roi_signal[0, rr, cc] = 1  # 第一个通道设置为圆盘
                                    # 第二个通道保持为全0 (已经默认是0)
                                    # 转换为 torch tensor
                                    roi_signal_tensor = torch.from_numpy(roi_signal).float().to(images.device)
                                    roi_aux_input_1 = roi_signal_tensor[ :, start_y:end_y, start_x:end_x]
                                    
                                    roi_inputs.append(roi_input_1)
                                    roi_aux_inputs.append(roi_aux_input_1)
                                    roi_suppixels.append(roi_suppixel_1)
                                    
                                    # 将裁剪后的列表转换为张量
                                    roi_inputs = torch.stack(roi_inputs) if roi_inputs else None
                                    roi_aux_inputs = torch.stack(roi_aux_inputs) if roi_aux_inputs else None
                                    roi_suppixels = torch.stack(roi_suppixels) if roi_suppixels else None
                                    
                                    
                                    all_superpixel_features , out_masks_list ,nuclick_out , fg_proto_features = model(roi_inputs , roi_aux_inputs , roi_suppixels, images, aux_inputs, superpixels, masks)
                                    
                                    
                                    out_put = out_masks_list[1]
                                    out_put[out_put > 0.5] = 1
                                    out_put[out_put <= 0.5] = 0
                                    
                                    '''
                                    #### 可视化
                                    out_put1 = out_put.squeeze(1)
                
                                    batch_size, _, _ = all_superpixel_features.shape
                                    for b in range(batch_size):
                                        signal_file_str = filenames[b]  # 从元组中提取出字符串
                                        filename = signal_file_str.split('/')[-1]
                                        
                                        out_put_1 = out_put1[b]
                                        # 清理绘图
                                        plt.clf()
                                        plt.close('all')

                                        plt.imshow(out_put_1.detach().cpu().numpy(), cmap='viridis')
                                        plt.colorbar()
                                        plt.title('Cosine Similarity Matrix')
                                        
                                        # 计算最大连通域的中心坐标
                                        center_coord = max_region_coords.mean(axis=0).astype(int)
                                        
                                        # 在图上显示红点
                                        plt.scatter(center_coord[1], center_coord[0], color='red', s=50, label='Center')
                                        plt.legend()
                                        
                                        # 保存图片
                                        plt.savefig(f'/home/gjs/ISF_nuclick/X_ISF/Test_New_data/keshihua_test/pre_{count}/{filename}_all.png')
                                        plt.show()
                                        
                                        # 清理绘图
                                        plt.clf()
                                        plt.close('all')
                                    '''
                                
                                else :
                                    
                                    # 计算裁剪的边界，中心坐标为正方形框的中心
                                    start_y = max(center_coord[0] - 40, 0)
                                    start_x = max(center_coord[1] - 40, 0)
                                    end_y = min(start_y + 80, H)
                                    end_x = min(start_x + 80, W)

                                    # 调整起始位置，以保持裁剪区域的大小为 100x100
                                    if end_y - start_y < 80:
                                        start_y = max(end_y - 80, 0)
                                    if end_x - start_x < 80:
                                        start_x = max(end_x - 80, 0)
                        
                                    # 裁剪 superpixels 区域
                                    roi_suppixel_1 = superpixels[b, :, start_y:end_y, start_x:end_x]
                                    
                                    # 获取ROI区域内的超像素编号
                                    roi_superpixels = roi_suppixel_1.flatten().unique().long()
                                    
                                    # 取出ROI区域内超像素块的特征
                                    roi_superpixel_features = all_superpixel_features[b, :, roi_superpixels]
                                
                                    # 计算原型特征与ROI内超像素块的余弦相似度
                                    similarity = F.cosine_similarity(fg_proto_features.unsqueeze(1), roi_superpixel_features, dim=0)
                                    
                                    # 将相似度值映射回每个像素点
                                    similarity_map = torch.zeros((end_y - start_y, end_x - start_x), device=images.device)
                                    for i, sp_id in enumerate(roi_superpixels):
                                        similarity_map[roi_suppixel_1.squeeze(0) == sp_id] = similarity[i]
                                    
                                    # 归一化 similarity_map (最小值归 0，最大值归 1)
                                    min_val = similarity_map.min()
                                    max_val = similarity_map.max()
                                    normalized_similarity_map = (similarity_map - min_val) / (max_val - min_val + 1e-6)

                                    # 获取 out_put 对应区域
                                    out_patch = out_put[b, 0, start_y:end_y, start_x:end_x]

                                    # 比较 similarity_map 和 out_put 对应位置的值，取较大的
                                    out_put[b, 0, start_y:end_y, start_x:end_x] = torch.max(out_patch, normalized_similarity_map)
                                    
                                    out_put[out_put > 0.5] = 1
                                    out_put[out_put <= 0.5] = 0
                                    '''
                                    #### 可视化
                                    out_put1 = out_put.squeeze(1)
                
                                    batch_size, _, _ = all_superpixel_features.shape
                                    for b in range(batch_size):
                                        signal_file_str = filenames[b]  # 从元组中提取出字符串
                                        filename = signal_file_str.split('/')[-1]
                                        
                                        out_put_1 = out_put1[b]
                                        # 清理绘图
                                        plt.clf()
                                        plt.close('all')

                                        plt.imshow(out_put_1.detach().cpu().numpy(), cmap='viridis')
                                        plt.colorbar()
                                        plt.title('Cosine Similarity Matrix')
                                        
                                        # 计算最大连通域的中心坐标
                                        center_coord = max_region_coords.mean(axis=0).astype(int)
                                        
                                        # 在图上显示红点
                                        plt.scatter(center_coord[1], center_coord[0], color='red', s=50, label='Center')
                                        plt.legend()
                                        
                                        # 保存图片
                                        plt.savefig(f'/home/gjs/ISF_nuclick/X_ISF/Test_New_data/keshihua_test/pre_{count}/{filename}.png')
                                        plt.show()
                                        
                                        # 清理绘图
                                        plt.clf()
                                        plt.close('all')     
                                    '''
                                out_masks_list[3] = out_put
                                
                                # _, batch_accuracy = calculate_binary_segmentation_accuracy(out_put, masks)            
                                # print("矫正后的准确率：",batch_accuracy)
                            else:
                                continue
                            
                    count +=1        
                            
                
                        

                    
                    
                            
                            
                
                # 计算准确率
                # _, batch_accuracy = calculate_binary_segmentation_accuracy(out_put, masks)
                # # val_accuracy += batch_accuracy * images.size(0)  # 将每个批次的准确率加权累积
                # print("矫正后的准确率：",batch_accuracy)
                # print("文件名：", filenames)
                

                ###  可视化特征分布
                '''
                out_put = out_masks_list[3]
                
                out_put = out_put.squeeze(1)
                
                batch_size, _, _ = all_superpixel_features.shape
                for b in range(batch_size):
                    signal_file_str = filenames[b]  # 从元组中提取出字符串
                    filename = signal_file_str.split('/')[-1]
                    
                    out_put_1 = out_put[b]
                    # 清理绘图
                    plt.clf()
                    plt.close('all')

                    plt.imshow(out_put_1.detach().cpu().numpy(), cmap='viridis')
                    plt.colorbar()
                    plt.title('Cosine Similarity Matrix')
                    # 保存图片
                    plt.savefig(f'/home/gjs/ISF_nuclick/X_ISF/Test_New_data/keshihua/pre_mask_val_0.8/{filename}.png')
                    plt.show()
                    
                    # 清理绘图
                    plt.clf()
                    plt.close('all')
                '''
                
                
                
                
                '''
                # 假设 all_superpixel_features 的 shape 是 (channels, num_superpixels)
                batch_size, channels, num_superpixels = all_superpixel_features.shape
                for b in range(batch_size): 
                    
                    target_sps = superpixels[b,0,...] # w, h
            
                    sps_labels_list = torch.unique(target_sps)
                    
                    # 将 masks 和 superpixels 展平
                    mask_flat = masks[b].view(-1)  # shape: (H*W,)
                    out_mask_flat = out_mask[b].view(-1)
                    
                    superpixels_flat = superpixels[b].view(-1)  # shape: (H*W,)

                    fg_sps_labels_list = []
                    fg_sps_labels_list_pre = []

                    for label in sps_labels_list:
                        # 获取该超像素块中所有像素的掩码值
                        mask_for_superpixel = mask_flat[superpixels_flat == label]

                        # 计算前景和背景像素的数量
                        num_fg_pixels = torch.sum(mask_for_superpixel)
                        num_bg_pixels = mask_for_superpixel.numel() - num_fg_pixels

                        if num_fg_pixels > num_bg_pixels:
                            fg_sps_labels_list.append(label)
                        else:
                            continue
                        
                    for label in sps_labels_list:
                        # 获取该超像素块中所有像素的掩码值
                        out_mask_for_superpixel = out_mask_flat[superpixels_flat == label]

                        # 计算前景和背景像素的数量
                        num_fg_pixels = torch.sum(out_mask_for_superpixel)
                        num_bg_pixels = out_mask_for_superpixel.numel() - num_fg_pixels

                        if num_fg_pixels > num_bg_pixels:
                            fg_sps_labels_list_pre.append(label)
                        else:
                            continue
                    
                    
                    filename = filenames[b]
                    # 转置特征矩阵，使其变成 (num_superpixels, channels)
                    features_transposed = all_superpixel_features[b].T  # shape: (num_superpixels, channels)
                    features_transposed = features_transposed.detach().cpu().numpy()

                    # 获取第 b 张图片的原型特征
                    prototype_feature = fg_proto_features[b].reshape(1, -1)  # shape: (1, channels)
                    prototype_feature = prototype_feature.detach().cpu().numpy()

                    # 将原型特征与超像素特征拼接，进行统一的降维
                    all_features = np.vstack([features_transposed, prototype_feature])  # shape: (num_superpixels + 1, channels)

                    # 选择降维方法 (PCA 或 t-SNE)
                    tsne = TSNE(n_components=2, random_state=42)
                    reduced_features_tsne = tsne.fit_transform(all_features)

                    # 创建颜色列表，默认全为蓝色
                    colors = ['blue'] * num_superpixels
                    
                    # 将前景超像素编号对应的颜色设置为绿色
                    # 处理 fg_sps_labels_list 和 fg_sps_labels_list_pre 的情况
                    for sp_idx in sps_labels_list:
                        sp_idx = int(sp_idx)  # 确保 sp_idx 是整数
                        if sp_idx in fg_sps_labels_list and sp_idx in fg_sps_labels_list_pre:
                            colors[sp_idx] = 'brown'  # 同时在两个列表中，置为棕色
                        elif sp_idx in fg_sps_labels_list:
                            colors[sp_idx] = 'green'  # 只在 fg_sps_labels_list 中，置为绿色
                        elif sp_idx in fg_sps_labels_list_pre:
                            colors[sp_idx] = 'black'  # 只在 fg_sps_labels_list_pre 中，置为黑色


                    # 添加原型特征的颜色（红色）
                    colors.append('red')  # 原型特征的位置在最后
                    
                    # 绘制 t-SNE 可视化
                    plt.figure(figsize=(8, 6))
                    plt.scatter(reduced_features_tsne[:-1, 0], reduced_features_tsne[:-1, 1], c=colors[:-1], alpha=0.7, s=5, label='Superpixels')
                    plt.scatter(reduced_features_tsne[-1, 0], reduced_features_tsne[-1, 1], c='red', alpha=1.0, s=50, label='Prototype Feature')
                    plt.title('t-SNE of Superpixel and Prototype Features')
                    plt.xlabel('Dimension 1')
                    plt.ylabel('Dimension 2')
                    plt.legend(loc='best')
                    plt.savefig(f'/home/gjs/ISF_nuclick/X_ISF/Test_New_data/T-sne/val/{filename}.png')
                    plt.clf()
                    plt.close('all')
                   '''
                    
                    
                for i in range(len(out_masks_list)):
                    iou_scores = []
                    val_dice_score[i] += dice_coeff(out_masks_list[i], masks).item() * images.size(0)
                    nuclick_val_dice[i] += dice_coeff(nuclick_out[i], roi_mask).item() * images.size(0)
                    
                    # 计算准确率
                    _, batch_accuracy = calculate_binary_segmentation_accuracy(out_masks_list[i], masks)
                    val_accuracy[i] += batch_accuracy * images.size(0)  # 将每个批次的准确率加权累积
                    
                    _, nuclick_batch_accuracy = calculate_binary_segmentation_accuracy(nuclick_out[i], roi_mask)
                    nuclick_val_acc[i] += nuclick_batch_accuracy * images.size(0)  # 将每个批次的准确率加权累积
                    
                    for pred, mask in zip(out_masks_list[i], masks):
                        miou = compute_miou_binary(pred, mask)
                        if not np.isnan(miou):
                            mean_IoU_list[i].append(miou)
                    
                    
                    
            
            # 对每个列表中的每个元素除以 len(train_loader.dataset)
            dataset_len = len(val_loader.dataset)

            for i in range(len(val_dice_score)):
                val_dice_score[i] /= dataset_len
                nuclick_val_dice[i] /= dataset_len
                val_accuracy[i] /= dataset_len
                nuclick_val_acc[i] /= dataset_len
    
                mean_IoU[i] = np.mean(mean_IoU_list[i])
                
        thresholds = [0.2,0.25,0.3,0.35,0.4,0.45,0.5]
        
        for i in range(len(thresholds)):
            print(f'Thresholds:{thresholds[i]:.4f} , Train Dice: {train_dice_score[i]:.4f}, Train Acc: {train_accuracy[i]:.4f} , nuclick_Train_Dice: {nuclick_train_dice[i]:.4f}, nuclick_Train_Acc: {nuclick_train_acc[i]:.4f}, Val Dice: {val_dice_score[i]:.4f}, Val_Mean IOU: {mean_IoU[i]:.4f}, Val_Acc: {val_accuracy[i]:.4f}, nuclick_Val_Dice: {nuclick_val_dice[i]:.4f},  nuclick_Val_Acc: {nuclick_val_acc[i]:.4f}')




train_model(model, train_loader, val_loader, epochs=1)
