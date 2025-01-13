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


from main import get_data_loaders,train_and_test,FocalLoss
from model import AttentionUNet

batch_size = 4
epochs = 50
bpath = '.'


dataloaderss = get_data_loaders( batch_size=4)
print(f"Training dataset size: {len(dataloaderss['training'].dataset)}")
print(f"Test dataset size: {len(dataloaderss['test'].dataset)}")

epochs = 120

def train():
    model = AttentionUNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = FocalLoss(gamma=2)

    trained_model, train_epoch_losses, test_epoch_losses = train_and_test(model, dataloaderss, optimizer, criterion, num_epochs=epochs)

    return trained_model, train_epoch_losses, test_epoch_losses


trained_model, train_epoch_losses, test_epoch_losses = train()
     