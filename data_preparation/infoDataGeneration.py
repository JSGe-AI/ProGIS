import os
import cv2
import numpy as np
from scipy import ndimage
import scipy.io

def maskRelabeling(inMask, sizeLimit):
    # A function to take an input label mask and relabel it with objects
    # starting with index 1
    
    outMask = np.zeros_like(inMask, dtype=np.uint8)#创建一个与输入掩码相同形状的全零数组，并指定数据类型为np.uint8：
    uniqueLabels = np.unique(inMask)
    uniqueLabels = uniqueLabels[uniqueLabels != 0] #获取输入掩码中的唯一标签值，并将值为0的标签排除在外
    i = 1

    for label in uniqueLabels:  #对于每个唯一标签值，执行以下操作
        thisMask = inMask == label   #创建一个布尔掩码，用于提取当前标签的区域
        labeledMask, numObjects = ndimage.label(thisMask)   #使用ndimage.label()函数对当前标签的区域进行连通组件标记，并返回标记后的掩码和对象数量
        for objLabel in range(1, numObjects + 1):           #遍历每个对象的标签，并计算对象的大小
            objSize = np.sum(labeledMask == objLabel)
            if objSize >= sizeLimit:                        #如果对象的大小大于等于sizeLimit，则将该对象所对应的像素在输出掩码中赋值为当前的索引i，并更新索引值
                outMask[labeledMask == objLabel] = i
                i += 1

    return outMask

def clickMapGenerator(mask):
    # Accepts an instance segmentation map and creates a clickMap,
    # which contains positive pixels at each nuclei (real) centroid and negative elsewhere.

    # Simple and fast approach: using centroid option of cv2.connectedComponentsWithStats function.
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)#使用cv2.connectedComponentsWithStats()函数对实例分割掩码进行连通组件分析，返回连通组件的数量、状态统计信息、质心坐标等
    cx = np.round(centroids[1:, 0]).astype(int)  #提取质心坐标的x和y坐标，并进行四舍五入取整
    cy = np.round(centroids[1:, 1]).astype(int)
    Indxs = np.ravel_multi_index((cy, cx), mask.shape)  #将质心坐标转换为在掩码数组中的索引值
    clickMap = np.zeros_like(mask, dtype=bool)   #创建一个与掩码数组形状相同的全零数组，并指定数据类型为布尔类型
    clickMap.flat[Indxs] = True                  #将点击地图中对应索引位置的像素值设为True，表示正值像素

    return clickMap, cx, cy

# NuClick: Semi-automatic Nuclei instance segmentation

# Generating info files to be fed into data.py function.
# This function accepts image and their relative masks and output a MAT
# file that encompasses all need information about that images.

# Training/Validation data generation
# set application
application = 'Gland'  # either 'Cell', 'Gland', 'Nucleus'

if application == 'Gland':
    sizeLimit = 500
    m = 512
    n = 512
elif application == 'Cell':
    sizeLimit = 300
    bb = 256
elif application == 'Nucleus':
    sizeLimit = 100
    bb = 128

# image reading & definitions
# set the paths for image reading and info saving
set = 'testB'  # either: 'train', 'testA', or 'testB'
imgPath = './Data/GlandSegmentation/data/' + set + '/image/'
maskPath = './Data/GlandSegmentation/data/' + set + '/_mask/'
infosSavePath = './Data/' + set + '/infos/'

imgExt = '.jpg'
maskExt = '.jpg'

# making the folders
if not os.path.exists(infosSavePath):
    os.makedirs(infosSavePath)

files = os.listdir(imgPath)
total = 1   #初始化一个变量total，用于记录处理的总数

for i in files:
    print('Working on', i)
    if i.endswith(imgExt):
        # 完整文件路径
        img_path = os.path.join(imgPath, i)
        mask_path = os.path.join(maskPath, i)
    img = cv2.imread(img_path)                         #使用cv2.imread()函数读取图像文件和掩码文件，并分别存储在变量img和mask中
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  

    if application == 'Gland':            #如果应用程序类型为'Gland'，则将图像img调整为指定尺寸(m, n)，并将其赋值给变量thisImg，同时将掩码mask调整为指定尺寸(m, n)
        img = cv2.resize(img, (m, n))
        thisImg = img
        mask = cv2.resize(mask, (m, n))
    else:
        m, n, _ = mask.shape              #否则，获取掩码的形状(m, n, _)


    mask = maskRelabeling(mask, sizeLimit)   #调用maskRelabeling()函数对掩码进行重新标记

    if application == 'Gland':  # for Gland, the process include image resizing  
        for j in range(1, np.max(mask) + 1):  # do for all objects in the mask          #对于掩码中的每个对象（标签从1到最大标签值）
            thisObject = mask == j
            thisPoint, _, _ = clickMapGenerator(mask)
            otherObjects = np.uint8(~thisObject) * mask
            otherObjects = maskRelabeling(otherObjects, sizeLimit)
            otherPoints, _, _ = clickMapGenerator(otherObjects)        #调用clickMapGenerator()函数生成点击地图，并将结果分别赋值给变量thisPoint、otherObjects和otherPoints

            thisBack = ~(mask > 0)             #创建一个背景掩码thisBack，用于表示非对象的区域
            if np.sum(thisObject) > 0:         #计算权重thisWeight
                w0 = (np.sum(otherObjects) + np.finfo(float).eps) / (np.sum(thisObject) + np.finfo(float).eps)
            else:
                w0 = 1
            thisWeight = np.double(thisBack) + 2 * np.double(otherObjects > 0) + (2 + w0) * np.double(thisObject)

            #cv2.imshow('thisWeightBCE', thisWeight)
            #cv2.waitKey(1)

            # 保存合成图像的信息
            save_name = '{}_{}_{}_{}'.format(application, set, i, j)
            save_path = infosSavePath + save_name + '_info.mat'

            # 创建一个字典，存储需要保存的变量
            save_data = {
                'thisImg': thisImg,
                'thisObject': thisObject,
                'thisPoint': thisPoint,
                'otherObjects': otherObjects,
                'otherPoints': otherPoints,
                'thisWeight': thisWeight
            }

            # 使用scipy.io.savemat保存数据到MAT文件
            scipy.io.savemat(save_path, save_data)
            total += 1

    else:  # for processing nucleus and cells, we crop a patch of [bb x bb]    #使用裁剪的方式处理图像和掩码
        clickMap, cx, cy = clickMapGenerator(mask)

        for j in range(len(cx)):  # do for all bounding boxes    #对于每个边界框，执行以下操作
            thisCx = cx[j]                              #获取当前边界框的中心坐标thisCx和thisCy，并计算裁剪的起始坐标(xStart, yStart)和结束坐标(xEnd, yEnd)
            thisCy = cy[j]
            xStart = max(thisCx - bb / 2, 1)
            yStart = max(thisCy - bb / 2, 1)
            xEnd = xStart + bb - 1
            yEnd = yStart + bb - 1
            if xEnd > n:                  #如果结束坐标超过图像的尺寸，则调整起始坐标以保证裁剪框在图像内部
                xEnd = n
                xStart = n - bb + 1
            if yEnd > m:
                yEnd = m
                yStart = m - bb + 1                         

            # Cropping the image & mask based on the bounding box
            maskVal = mask[thisCy, thisCx]       #获取当前边界框的掩码值maskVal，如果为0，则跳过当前边界框
            if maskVal == 0:
                continue
            thisObject = mask == maskVal         #创建当前对象的布尔掩码thisObject和其他对象的掩码otherObjects
            otherObjects = ((mask > 0) - thisObject)

            # saving the information for the synthesized image
            saveName = '{}_{}_{}_{:03d}_{:03d}'.format(application, set, i, j)
            np.savez_compressed(infosSavePath + saveName + '_info.npz', thisImg=thisImg, thisObject=thisObject,
                                thisPoint=thisPoint, otherObjects=otherObjects, thisWeight=np.zeros_like(thisObject))
            total += 1