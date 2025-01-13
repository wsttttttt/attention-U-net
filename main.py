import os,time,cv2

import numpy as np
import torch.utils.data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T

# 用于记录日志
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim

from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.utils import make_grid

# Data manipulations
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt

# helpers
import glob
import os
import copy
import time
import csv
from tqdm import tqdm  # 导入 tqdm
#from google.colab import drive
from model import AttentionUNet
import os
from torch.utils.data import Dataset
ReadColorImage = 0
class trainDataloader(Dataset):
    def __init__(self,mode, target_size=(256, 256),transform=None):#None,
        #self.image_dir = image_dir
        #self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size  # 目标大小
        self.mode =mode
        self.dir = os.path.join('./data/train/image/')  # 修正 s.path 为 os.path
        self.gt_dir = os.path.join('./data/train/mask/')
      
       
        self.image_list = sorted([f for f in os.listdir(self.dir) if f.endswith('.png')])
        self.gt_list = sorted([f for f in os.listdir(self.gt_dir) if f.endswith('.png')])

        # 确保文件一一对应
        self.image_list = [f for f in self.image_list if f in self.gt_list]
        self.gt_list = [f for f in self.gt_list if f in self.image_list]

        # 创建文件映射
        self.file_mapping = [(img, gt) for img, gt in zip(self.image_list, self.gt_list)]
        
        # 检查文件映射是否为空
        assert len(self.file_mapping) > 0, "No files found in the dataset."
        self.transform = transform
        print(f"Number of image-mask pairs: {len(self.file_mapping)}")
        train_files, val_files = train_test_split(self.file_mapping, train_size=0.8, random_state=42)
        if self.mode == 'training':
            self.file_mapping = train_files
            print(f"Training set size: {len(self.file_mapping)}")
        elif self.mode == 'test':
            self.file_mapping = val_files
        
            
    def __len__(self):
         return len(self.file_mapping)

    def __getitem__(self, idx):
        # 加载图像和 mask
        if idx >= len(self.file_mapping):
                raise IndexError(f"Index {idx} is out of range for dataset with size {len(self.file_mapping)}")
        img_name, gt_name = self.file_mapping[idx]

        img_path = os.path.join(self.dir, img_name)
        mask_path = os.path.join(self.gt_dir, gt_name)
        #img_path = os.path.join(self.image_dir, self.image_names[idx])
        #mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        image = image.resize(self.target_size)  # 调整图像大小
        mask = mask.resize(self.target_size)    # 调整掩码大小
        image_array = np.array(image).astype(np.float32) / 255.0  # 归一化到 [0, 1]，并确保类型为 float32
        mask_array = np.array(mask).astype(np.float32) 
        #mask = mask.astype(np.uint8)
        if self.transform:
            # 将图像和掩码打包成字典传入 transform
            sample = {'image': image, 'mask': mask}
            sample = self.transform(sample)
            image, mask = sample['image'], sample['mask']

        return  {'image': image, 'mask': mask}

class TestSRISTDataset:
    def __init__(self, image_dir, mask_dir, target_size=(256, 256),transform=None):#None,
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size  # 目标大小
        self.image_names = [
            file_name for file_name in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, file_name)) and not file_name.startswith('.')
        ]
        self.mask_names = [
            file_name for file_name in os.listdir(mask_dir)
            if os.path.isfile(os.path.join(mask_dir, file_name)) and not file_name.startswith('.')
        ]
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        
        self.image_mask_pairs = self.match_files(self.image_files, self.mask_files)
        
    def match_files(self, image_files, mask_files):
        """
        根据公共部分匹配图像和掩码文件
        :param image_files: 图像文件列表
        :param mask_files: 掩码文件列表
        :return: [(image_file, mask_file), ...] 匹配好的文件对
        """
        pairs = []
        for image_file in image_files:
            # 提取图像文件的公共部分（如 "Misc_1"）
            common_part = image_file.split('.')[0]  # 假设以 '.' 分隔，取第一部分
            
            # 使用正则表达式确保精确匹配掩码文件（公共部分后加上特定后缀）
            matching_masks = [m for m in mask_files if m.startswith(common_part + "_pixels0")]
            #matching_masks = [m for m in mask_files if m.startswith(common_part)]
            if matching_masks:
                # 如果找到多个匹配，仅保留第一个匹配项
                pairs.append((image_file, matching_masks[0]))
            else:
                print(f"警告：未找到图像 {image_file} 对应的掩码文件！")
        
        return pairs
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # 加载图像和 mask
        #img_path = os.path.join(self.image_dir, self.image_names[idx])
        #mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
        image_file, mask_file = self.image_mask_pairs[idx]
        img_path = os.path.join(self.image_dir, image_file)
        mask_path = os.path.join(self.mask_dir, mask_file)
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        image = image.resize(self.target_size)  # 调整图像大小
        mask = mask.resize(self.target_size)    # 调整掩码大小
        image_array = np.array(image).astype(np.float32) / 255.0  # 归一化到 [0, 1]，并确保类型为 float32
        mask_array = np.array(mask).astype(np.float32) 
        #mask = mask.astype(np.uint8)
        if self.transform:
            # 将图像和掩码打包成字典传入 transform
            sample = {'image': image, 'mask': mask}
            sample = self.transform(sample)
            image, mask = sample['image'], sample['mask']
        
        # 统一调整图像大小
        
        
        # 统一调整大小
                  # 掩码转为 float32

        # 转为 PyTorch 张量
        #image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        #mask_tensor = torch.from_numpy(mask_array).unsqueeze(0) 
        
        
        return {'image': image, 'mask': mask}
    
class TestMDvsFADataloader(Dataset):
    def __init__(self, mode, target_size=(256, 256), transform=None):
        """
        自定义数据加载器
        :param mode: 'train' 或 'test'
        :param target_size: 图像和 mask 的目标大小 (H, W)
        :param transform: 数据增强或变换操作
        """
        self.mode = mode
        self.target_size = target_size
        self.transform = transform

        if self.mode == 'train':
            self.dir = os.path.join('./data/train/image/')
            self.gt_dir = os.path.join('./data/train/mask/')
        elif self.mode == 'test':
            self.dir = os.path.join('./data/test/MDvsFA/image/')
            self.gt_dir = os.path.join('./data/test/MDvsFA/mask/')
        else:
            raise NotImplementedError("Mode should be 'train' or 'test'.")

        # 检查目录是否存在
        assert os.path.exists(self.dir), f"Image directory does not exist: {self.dir}"
        assert os.path.exists(self.gt_dir), f"Mask directory does not exist: {self.gt_dir}"

        # 加载图像和 mask 列表
        self.image_list = sorted([f for f in os.listdir(self.dir) if f.endswith('.png')])
        self.gt_list = sorted([f for f in os.listdir(self.gt_dir) if f.endswith('.png')])

        # 确保文件一一对应
        self.image_list = [f for f in self.image_list if f in self.gt_list]
        self.gt_list = [f for f in self.gt_list if f in self.image_list]

        # 创建文件映射
        self.file_mapping = [(img, gt) for img, gt in zip(self.image_list, self.gt_list)]
        assert len(self.file_mapping) > 0, "No image-mask pairs found in the dataset."

        print(f"Mode: {self.mode}, Number of image-mask pairs: {len(self.file_mapping)}")

    def __len__(self):
        return len(self.file_mapping)

    def __getitem__(self, idx):
        """
        加载和预处理图像-mask 对
        :param idx: 索引
        :return: 包含 Tensor 的字典 {'image': Tensor, 'mask': Tensor}
        """
        if idx >= len(self.file_mapping):
            raise IndexError(f"Index {idx} is out of range for dataset with size {len(self.file_mapping)}")

        img_name, gt_name = self.file_mapping[idx]
        img_path = os.path.join(self.dir, img_name)
        mask_path = os.path.join(self.gt_dir, gt_name)

        # 加载图像和 mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # 调整大小
        image = image.resize(self.target_size)
        mask = mask.resize(self.target_size)

        # 转换为 NumPy 数组
        image_array = np.array(image).astype(np.float32) / 255.0  # 归一化到 [0, 1]
        mask_array = np.array(mask).astype(np.float32)

        # 转换为 PyTorch Tensor
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()  # 转换为 [C, H, W]
        mask_tensor = torch.from_numpy(mask_array).long()  # mask 保持整数值，类型为 long

        # 应用数据增强（如果有）
        if self.transform:
            sample = {'image': image_tensor, 'mask': mask_tensor}
            sample = self.transform(sample)
            image_tensor, mask_tensor = sample['image'], sample['mask']

        return {'image': image_tensor, 'mask': mask_tensor}

class Resize(object):
    """Resize image and/or masks."""

    def __init__(self, image_resize, mask_resize):
        self.image_resize = image_resize
        self.mask_resize = mask_resize

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if len(image.shape) == 3:
            image = image.transpose(1, 2, 0)
        if len(mask.shape) == 3:
            mask = mask.transpose(1, 2, 0)
        mask = cv2.resize(mask, self.mask_resize, cv2.INTER_AREA)
        image = cv2.resize(image, self.image_resize, cv2.INTER_AREA)
        if len(image.shape) == 3:
            image = image.transpose(2, 0, 1)
        if len(mask.shape) == 3:
            mask = mask.transpose(2, 0, 1)

        return {'image': image,
                'mask': mask}

class ToTensor:
    def __call__(self, sample, mask_resize=None, image_resize=None):
        image, mask = sample['image'], sample['mask']

        # 处理 image
        if isinstance(image, np.ndarray):  # 如果是 NumPy 数组
            image = image.transpose(2, 0, 1)  # 转换维度 (H, W, C) -> (C, H, W)
        elif isinstance(image, torch.Tensor):  # 如果已经是张量
            image = image.permute(2, 0, 1)  # 转换维度 (H, W, C) -> (C, H, W)
        else:  # 如果是 PIL.Image，则转换为 NumPy 数组
            image = np.array(image).transpose(2, 0, 1)

        # 处理 mask
        if isinstance(mask, Image.Image):  # 如果 mask 是 PIL.Image 类型
            mask = np.array(mask)  # 转换为 NumPy 数组
        
        if len(mask.shape) == 2:  # 如果 mask 是单通道
            mask = mask.reshape((1,) + mask.shape)

        return {'image': torch.tensor(image, dtype=torch.float32),
                'mask': torch.tensor(mask, dtype=torch.float32)}

class Normalize(object):
   

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        return {'image': image.type(torch.FloatTensor) / 255,
                'mask': mask.type(torch.FloatTensor) / 255}


class HorizontalFlip(object):
    def __init__(self, prob=.5):
        self.prob = prob

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        if np.random.random() < self.prob:
            
            image = np.array(image)
            mask = np.array(mask)
            mask = mask.astype(np.uint8)
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        return {'image': image,
                'mask': mask}


class ApplyClaheColor(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = np.array(image)
        mask = np.array(mask)
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0)
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

        return {'image': img_output,
                'mask': mask}

class Denoise(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = np.array(image)
        mask = np.array(mask)
        image = cv2.bilateralFilter(image, d=5, sigmaColor=9, sigmaSpace=9)
        return {'image': image,
                'mask': mask}


def get_data_loaders(batch_size=4):
    data_transforms = {
        # Resize((592, 576), (592, 576)),
        'training': transforms.Compose([ HorizontalFlip(), ApplyClaheColor(), Denoise(),ToTensor(),Normalize()]),#ToTensor(),HorizontalFlip(), ApplyClaheColor(), Denoise(),
        
        'test': transforms.Compose([HorizontalFlip(), ApplyClaheColor(), Denoise(),ToTensor(),Normalize()]),#ToTensor(),HorizontalFlip(), ApplyClaheColor(), Denoise(),  ,Normalize()
    }

    image_datasets = {x: trainDataloader(mode=x, transform=data_transforms[x])
                      for x in ['training', 'test']}

    data_loaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0)
                    for x in ['training', 'test']}

    return data_loaders

def plot_batch_from_dataloader(dataloaders, batch_size):
    # Get a batch of data
    batch = next(iter(dataloaders['training']))

    for i in range(batch_size):
        # Extract image and mask
        np_img = batch['image'][i].numpy()  # Convert tensor to numpy array
        np_mask = batch['mask'][i].numpy()

        # Handle dimensions for image
        if np_img.shape[0] == 3:  # If shape is (C, H, W), transpose to (H, W, C)
            np_img = np.transpose(np_img, (1, 2, 0))
        elif np_img.shape[-1] == 3:  # If shape is already (H, W, C), no need to transpose
            pass
        else:
            raise ValueError(f"Unexpected image shape: {np_img.shape}")

        # Normalize image data to [0, 1] for imshow
        if np_img.min() < 0 or np_img.max() > 1:
            np_img = (np_img + 1.0) / 2.0  # Map [-1, 1] to [0, 1]

        # Handle dimensions for mask
        if np_mask.ndim == 3:  # If mask has shape (1, H, W), squeeze it
            np_mask = np.squeeze(np_mask)
        elif np_mask.ndim == 2:  # If mask has shape (H, W), no need to adjust
            pass
        else:
            raise ValueError(f"Unexpected mask shape: {np_mask.shape}")

        # Create subplots
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(np_img)
        ax[0].set_title("Image")
        ax[0].axis('off')

        ax[1].imshow(np_mask, cmap='gray')
        ax[1].set_title("Mask")
        ax[1].axis('off')

        plt.show()



#损失与指标
def dice_coeff(prediction, target):

    mask = np.zeros_like(prediction)
    mask[prediction >= 0.5] = 1

    inter = np.sum(mask * target)
    union = np.sum(mask) + np.sum(target)
    epsilon=1e-6
    result = np.mean(2 * inter / (union + epsilon))
    return result


class FocalLoss(nn.modules.loss._WeightedLoss):

    def __init__(self, gamma=0, size_average=None, ignore_index=-100,
                 reduce=None, balance_param=1.0):
        super(FocalLoss, self).__init__(size_average)
        self.gamma = gamma
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):
        
        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)
           
        # compute the negative likelyhood
        logpt = - F.binary_cross_entropy_with_logits(input, target)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -( (1-pt)**self.gamma ) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss
def f1_score(output_image,gt_image,thre):
    output_image = np.squeeze(output_image)
    gt_image = np.squeeze(gt_image)
    out_bin = output_image>thre
    gt_bin = gt_image>thre
    recall = np.sum(gt_bin*out_bin)/np.maximum(1,np.sum(gt_bin))
    prec   = np.sum(gt_bin*out_bin)/np.maximum(1,np.sum(out_bin))
    F1 = 2*recall*prec/np.maximum(0.001,recall+prec)
    return F1



def train_and_test(model, dataloaders, optimizer, criterion, num_epochs=100, show_images=False):
    since = time.time()
    best_dice_coeff = 0  # 初始化为最低值
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    fieldnames = ['epoch', 'training_loss', 'test_loss', 'training_dice_coeff', 'test_dice_coeff', "training_f1", "test_f1"]
    train_epoch_losses = []
    test_epoch_losses = []

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        batchsummaryf1 = {a: [0] for a in fieldnames}
        batch_train_loss = 0.0
        batch_test_loss = 0.0

        for phase in ['training', 'test']:
            if phase == 'training':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # 自定义进度条：设置 total 为数据加载器的长度，实时显示损失和 F1
            phase_dataloader = dataloaders[phase]
            progress_bar = tqdm(phase_dataloader, desc=f"{phase.capitalize()} Phase", leave=True)

            for sample in progress_bar:
                if show_images:
                    grid_img = make_grid(sample['image'])
                    grid_img = grid_img.permute(1, 2, 0)
                    plt.imshow(grid_img)
                    plt.show()

                inputs = sample['image'].to(device)
                masks = sample['mask'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # track history only in training phase
                with torch.set_grad_enabled(phase == 'training'):
                    outputs = model(inputs)
                    loss = criterion(outputs, masks)

                    y_pred = outputs.data.cpu().numpy().ravel()
                    y_true = masks.data.cpu().numpy().ravel()

                    # 计算 F1 score
                    f1 = f1_score(y_pred, y_true, 0.5)
                    batchsummaryf1[f'{phase}_f1'].append(f1)

                    # back-propagation
                    if phase == 'training':
                        loss.backward()
                        optimizer.step()
                        batch_train_loss += loss.item() * sample['image'].size(0)
                    else:
                        batch_test_loss += loss.item() * sample['image'].size(0)

                    # 更新进度条显示
                    progress_bar.set_postfix({
                        "Loss": f"{loss.item():.4f}",
                        "F1": f"{f1:.4f}"
                    })

            # 保存每个 epoch 的损失
            if phase == 'training':
                epoch_train_loss = batch_train_loss / len(dataloaders['training'].dataset)
                train_epoch_losses.append(epoch_train_loss)
            else:
                epoch_test_loss = batch_test_loss / len(dataloaders['test'].dataset)
                test_epoch_losses.append(epoch_test_loss)

            # 保存 epoch 相关统计信息
            batchsummaryf1['epoch'] = epoch

        # 计算测试集上的平均 F1
        avg_test_f1 = np.mean(batchsummaryf1['test_f1'])

        # 保存最佳模型
        if avg_test_f1 > best_dice_coeff:
            best_dice_coeff = avg_test_f1
            torch.save(model.state_dict(), 'best_model.pth')  # 保存最优模型
            print(f"Best model saved with F1: {best_dice_coeff:.4f}")
        
        # 输出训练和测试的 F1
        for field in fieldnames[3:]:
            batchsummaryf1[field] = np.mean(batchsummaryf1[field])
        print(f'\t\t\t train_f1: {batchsummaryf1["training_f1"]:.4f}, test_f1: {batchsummaryf1["test_f1"]:.4f}')

    # 总结
    print(f'Best F1 score: {best_dice_coeff:.4f}')
    print(f'Last F1 score: {avg_test_f1:.4f}')
    torch.save(model.state_dict(), 'last_model.pth')
    return model, train_epoch_losses, test_epoch_losses

image_dir = "./data/test/SIRST/image"
mask_dir = "./data/test/SIRST/mask"
img_size = (224, 224)  # 模型输入大小

# 创建 Dataset 和 DataLoader
#test_dataset = TestDataset(image_dir, mask_dir, img_size)
def get_testSRISTdata_loaderst(mode, batch_size=4):

    data_transforms = {
        # Resize((592, 576), (592, 576)),
        'training': transforms.Compose([HorizontalFlip(), ApplyClaheColor(), Denoise(),ToTensor(), Normalize()]),#transforms.Resize((256, 256))ToTensor(),HorizontalFlip(), ApplyClaheColor(), Denoise(),Normali()
        'test': transforms.Compose([HorizontalFlip(), ApplyClaheColor(), Denoise(), ToTensor(),Normalize()]),#HorizontalFlip(), ApplyClaheColor(), Denoise(),  ,Normalize()
    }

    image_datasets = {x: TestSRISTDataset(image_dir, mask_dir, img_size,transform=data_transforms[x])#transform=data_transforms[x]
                      for x in ['training', 'test']}

    data_loaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0)
                    for x in ['training', 'test']}

    return data_loaders

def visualize_predictions(model, dataloader, threshold=0.5, device='cuda:0', num_images=5):
    """
    可视化测试集中模型的预测结果
    :param model: 训练好的模型
    :param dataloader: 测试数据集的 DataLoader
    :param threshold: 二值化阈值，用于生成预测 mask
    :param device: 设备 (cpu 或 cuda)
    :param num_images: 显示的图像数量
    """
    model.eval()  # 设置为评估模式
    model.to(device)

    images_shown = 0

    with torch.no_grad():
        for sample in dataloader:
            #print(type(sample))  # 查看 sample 的类型
            #print(sample)        # 打印 sample 的内容
            inputs = sample['image'].to(device)
            masks = sample['mask'].to(device)

            # 模型预测
            outputs = model(inputs)
            preds = (outputs > threshold).float()  # 二值化预测结果

            # 转为 CPU 格式，便于可视化
            inputs = inputs.cpu()
            masks = masks.cpu()
            preds = preds.cpu()

            # 显示 num_images 张图片
            for i in range(inputs.size(0)):  # 遍历 batch 中的每张图片
                if images_shown >= num_images:  # 如果已经显示的图像数达到限制，结束循环
                    break  # 使用 break 更符合逻辑，替代 return
            
                plt.figure(figsize=(12, 4))  # 设置画布大小
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                plt.subplots_adjust(wspace=0, hspace=0)
                
                # 调整子图之间的间距
                #fig.tight_layout(pad=2)
                # 显示原图
                plt.subplot(1, 3, 1)
                image_to_display1 = torch.clip(inputs[i], 0, 1)  # 将值限制在 [0, 1] 范围内
                plt.imshow(image_to_display1.permute(1, 2, 0).cpu().numpy())  # 调整通道顺序并转为 NumPy
                plt.title("Input Image")
                plt.axis('off')
                # 显示真实 Mask
                plt.subplot(1, 3, 2)
                image_to_display2 = torch.clip(masks[i], 0, 1)  # 确保值在 [0, 1] 范围内
                plt.imshow(image_to_display2.squeeze().cpu().numpy(), cmap='gray')  # 转为 NumPy 并显示灰度图
                plt.title("Ground Truth Mask")
                plt.axis('off')
                # 显示预测 Mask
                
                plt.subplot(1, 3, 3)
                image_to_display3 = torch.clip(preds[i], 0, 1)  # 同样限制值范围
                plt.imshow(image_to_display3.squeeze().cpu().numpy(), cmap='gray')  # 转为 NumPy 并显示灰度图
                plt.title("Predicted Mask")
                plt.axis('off')
                plt.show()  # 显示图片

                images_shown += 1  # 增加已显示图像计数
                


from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
def testSRIST_model(model, dataloader, criterion, threshold=0.5, device='cuda:0'):
    """
    测试模型性能并计算 F1 指标，同时可视化预测结果
    :param model: 训练好的模型
    :param dataloader: 测试数据集的 DataLoader
    :param criterion: 损失函数，用于计算测试损失
    :param threshold: 二值化阈值，用于生成预测 mask
    :param device: 设备 (cpu 或 cuda)
    """
    model.eval()  # 设置为评估模式
    model.to(device)

    test_loss = 0.0
    f1_scores = []  # 用于记录每批次的 F1 指标
    dice_coefff = []
    batch_indices = []
    batch_idx=1
    # 用 tqdm 显示测试进度条
    with torch.no_grad():
        for sample in tqdm(dataloader, desc="Testing"):
            #print(type(sample))  # 查看 sample 的类型
            #print(sample)        # 打印 sample 的内容
            inputs = sample['image'].to(device)
            masks = sample['mask'].to(device)

            # 模型预测
            outputs = model(inputs)
            loss = criterion(outputs, masks)  # 计算测试损失
            test_loss += loss.item() * inputs.size(0)

            # 转为 numpy 格式，便于计算 F1 指标
            y_pred = outputs.data.cpu().numpy().ravel()
            y_true = masks.data.cpu().numpy().ravel()

            # 计算 F1 指标
            f1 = f1_score(y_pred, y_true,0.5)
            f1_scores.append(f1)
            dice = dice_coeff(y_pred, y_true)
            dice_coefff.append(dice)
            batch_indices.append(batch_idx + 1)
            batch_idx=batch_idx+1
    # 计算平均测试损失和 F1 指标
    avg_test_loss = test_loss / len(dataloader.dataset)
    avg_f1 = np.mean(f1_scores)
    avg_dice_coefff = np.mean(dice_coefff)
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test F1 Score: {avg_f1:.4f}")
    #print(f"Test dice Score: {avg_dice_coefff:.4f}")
    '''
    plt.figure(figsize=(10, 5))
    plt.plot(batch_indices, f1_scores, label='F1 Score', marker='o')
    #plt.plot(batch_indices, dice_coefff, label='Dice Coefficient', marker='x', linestyle='--')
    plt.xlabel('Batch Index')
    plt.ylabel('Score')
    plt.title('F1 Scores')
    plt.legend()
    plt.grid(True)
    plt.show()
    '''
    plt.figure(figsize=(12, 6))
    
    # 柱状图
    plt.bar(batch_indices, f1_scores, color='skyblue', alpha=0.7, label='F1 Score')
    
    ## 平滑折线图
    #smoothed_f1 = gaussian_filter1d(f1_scores, sigma=2)
    #plt.plot(batch_indices, smoothed_f1, color='red', label='Smoothed F1 Score', linewidth=2)

    plt.xlabel('Batch Index')
    plt.ylabel('Score')
    plt.title('F1 Scores')
    plt.legend()
    plt.grid(True)
    plt.show()
    return avg_test_loss, avg_f1,avg_dice_coefff

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
def get_testMDvsFAdata_loaderst(mode, batch_size=4):
    
    data_transforms = {
        # Resize((592, 576), (592, 576)),
        'training': transforms.Compose([ToTensor()]),#transforms.Resize((256, 256)),HorizontalFlip(), ApplyClaheColor(), Denoise(),Normali()
        'test': transforms.Compose([ToTensor()]),#HorizontalFlip(), ApplyClaheColor(), Denoise(),  ,Normalize()
    }

    image_datasets = {x: TestMDvsFADataloader(mode = mode)#transform=data_transforms[x]
                      for x in ['training', 'test']}

    data_loaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0)
                    for x in ['training', 'test']}

    return data_loaders

def testMDvsFA_model(model, dataloader, criterion, threshold=0.5, device='cuda:0'):
    model.eval()
    test_loss = 0.0
    test_f1 = 0.0
    test_dice = 0.0
    f1_scores = []  # 用于记录每批次的 F1 指标
    dice_coefff = []
    batch_indices = []
    batch_idx=1
    with torch.no_grad():
        for batch in dataloader:
            inputs, masks = batch['image'].to(device), batch['mask'].to(device)

            # 模型预测
            outputs = model(inputs)

            # 打印形状调试
            #print(f"Model output shape: {outputs.shape}")
            #print(f"Target (masks) shape: {masks.shape}")

            # 如果是二分类任务，移除多余的通道维度
            #if outputs.shape[1] == 1:
            #    outputs = outputs.squeeze(1)  # [batch_size, height, width]
            #    outputs = torch.sigmoid(outputs)  # 转换为概率分布

            # 确保目标是整数类型，并移除多余通道维度
            masks = masks.long()
            if len(masks.shape) == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)
            if len(masks.shape) < len(outputs.shape):
                masks = masks.unsqueeze(1).float()  # [batch_size, 1, height, width]

            # 计算损失
            loss = criterion(outputs, masks)
            test_loss += loss.item() * inputs.size(0)

            # 计算 F1 和 Dice（伪代码，需根据任务调整）
            # outputs_binary = (outputs > threshold).float()
            # f1_score = compute_f1_score(outputs_binary, masks)
            # dice_score = compute_dice_score(outputs_binary, masks)
            # test_f1 += f1_score * inputs.size(0)
            # test_dice += dice_score * inputs.size(0)
            y_pred = outputs.data.cpu().numpy().ravel()
            y_true = masks.data.cpu().numpy().ravel()

            # 计算 F1 指标
            f1 = f1_score(y_pred, y_true,0.5)
            f1_scores.append(f1)
            dice = dice_coeff(y_pred, y_true)
            dice_coefff.append(dice)
            batch_indices.append(batch_idx + 1)
            batch_idx=batch_idx+1
    avg_test_loss = test_loss / len(dataloader.dataset)
    avg_f1 = np.mean(f1_scores)
    avg_dice_coefff = np.mean(dice_coefff)
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test F1 Score: {avg_f1:.4f}")
    #print(f"Test dice Score: {avg_dice_coefff:.4f}")
    '''
    plt.figure(figsize=(10, 5))
    plt.plot(batch_indices, f1_scores, label='F1 Score', marker='o')
    #plt.plot(batch_indices, dice_coefff, label='Dice Coefficient', marker='x', linestyle='--')
    plt.xlabel('Batch Index')
    plt.ylabel('Score')
    plt.title('F1 Scores')
    plt.legend()
    plt.grid(True)
    plt.show()
    '''
    plt.figure(figsize=(12, 6))
    
    # 柱状图
    plt.bar(batch_indices, f1_scores, color='skyblue', alpha=0.7, label='F1 Score')
    
    ## 平滑折线图
    #smoothed_f1 = gaussian_filter1d(f1_scores, sigma=2)
    #plt.plot(batch_indices, smoothed_f1, color='red', label='Smoothed F1 Score', linewidth=2)

    plt.xlabel('Batch Index')
    plt.ylabel('Score')
    plt.title('F1 Scores')
    plt.legend()
    plt.grid(True)
    plt.show()
    return avg_test_loss, avg_f1,avg_dice_coefff
    return test_loss / len(dataloader.dataset), test_f1 / len(dataloader.dataset), test_dice / len(dataloader.dataset)