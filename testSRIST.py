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


from main import get_data_loaders,train_and_test,FocalLoss,get_testMDvsFAdata_loaderst,testMDvsFA_model,visualize_predictions,get_testSRISTdata_loaderst,testSRIST_model
from model import AttentionUNet

batch_size=4
bpath = '.'
test1dataloader = get_testSRISTdata_loaderst(mode = 'test', batch_size=batch_size)
test_loader = test1dataloader['test']

# 假设测试数据集的 DataLoader 为 test_loader
# 加载保存的模型权重
model = AttentionUNet()
model.load_state_dict(torch.load('last_model.pth'))  # 加载保存的模型权重
#model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))

# 定义损失函数
criterion = FocalLoss()  # 根据任务选择适当的损失函数

# 测试模型并计算 F1 指标
test_loss, test_f1 ,test_dice = testSRIST_model(model, test_loader, criterion, threshold=0.5, device='cuda:0')

# 可视化预测结果
visualize_predictions(model, test_loader, threshold=0.5, device='cuda:0', num_images=5)
#print(test_dice)
