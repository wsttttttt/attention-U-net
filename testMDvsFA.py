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


from main import get_data_loaders,train_and_test,FocalLoss,get_testMDvsFAdata_loaderst,testMDvsFA_model,visualize_predictions
from model import AttentionUNet

batch_size=4
bpath = '.'
test2dataloader = get_testMDvsFAdata_loaderst(mode = 'test', batch_size=batch_size)
test2_loader = test2dataloader['test']

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = model.to(device)  # 将模型移动到 GPU 或 CPU


import torch

# 加载模型
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 自动选择 GPU 或 CPU
model = AttentionUNet()
model.load_state_dict(torch.load('last_model.pth', map_location=device, weights_only=True))
#model.load_state_dict(torch.load('best_model.pth', map_location=device))  # 加载权重，并确保权重在正确设备上
model2 = model.to(device)  # 将模型移动到 GPU 或 CPU

# 定义损失函数
criterion = FocalLoss()  # 根据任务选择适当的损失函数

# 测试模型并计算 F1 指标
test2_loss, test2_f1, test2_dice = testMDvsFA_model(model2, test2_loader, criterion, threshold=0.5, device=device)

# 可视化预测结果
visualize_predictions(model2, test2_loader, threshold=0.5, device=device, num_images=5)

# 打印测试集的 Dice 指标
print(f"Test F1: {test2_dice}")