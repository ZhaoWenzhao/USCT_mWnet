#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function, division

try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse
    
import os
import math
import numpy as np 
import pandas as pd 
#from skimage.io import imread
import matplotlib.pyplot as plt
#from skimage.segmentation import mark_boundaries

import cv2
import random
from datetime import datetime
import json
import gc

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn
import torch.backends.cudnn
from torch.autograd import Variable
from torch.nn import functional as F
#from torchvision.transforms import ToTensor, Normalize, Compose
#from torchvision import models
import torch.utils.model_zoo as model_zoo
#from sklearn.model_selection import train_test_split

#from skimage.morphology import binary_opening, disk

from tqdm import tqdm
from pathlib import Path

#from skimage.morphology import label
#from skimage.transform import resize

from os.path import dirname, join as pjoin
import scipy.io as sio


torch.manual_seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(0)

# Parameters
BATCH_SIZE = 16
LR = 1.0e-4
NUM_epochs = 50
# Dataset path (all the *.mat files)
train_image_dir = './data110yall/data110/'
test_image_dir = './data110yall/data110/valid/'
valid_image_dir = './data110yall/data110/valid/'

# List of *.mat files
train_df = pd.read_csv("./data110yall/data110/img110yall.csv",  usecols=[0])
valid_df = pd.read_csv("./data110yall/data110/valid/img110yvalid.csv",  usecols=[0])


# Statistics for input data normalization
filepath='./data110yall/usct_stat110.mat'#
    
matfile = sio.loadmat(filepath)
maxII=matfile['maxII'][0,0]
minII=matfile['minII'][0,0]
meanII=matfile['meanII'][0,0]
meanminII=matfile['meanminII'][0,0]
meanmaxII=matfile['meanmaxII'][0,0]
dII=(meanmaxII-meanminII)*1.2

maxIR=matfile['maxIR'][0,0]
minIR=matfile['minIR'][0,0]
meanIR=matfile['meanIR'][0,0]
meanminIR=matfile['meanminIR'][0,0]
meanmaxIR=matfile['meanmaxIR'][0,0]
dIR=(meanmaxIR-meanminIR)*1.2

maxOI=matfile['maxOI'][0,0]
minOI=matfile['minOI'][0,0]
meanOI=matfile['meanOI'][0,0]
meanminOI=matfile['meanminOI'][0,0]
meanmaxOI=matfile['meanmaxOI'][0,0]
dOI=(meanmaxOI-meanminOI)*1.2

maxOR=matfile['maxOR'][0,0]
minOR=matfile['minOR'][0,0]
meanOR=matfile['meanOR'][0,0]
meanminOR=matfile['meanminOR'][0,0]
meanmaxOR=matfile['meanmaxOR'][0,0]
dOR=(meanmaxOR-meanminOR)*1.2

print(dII)
print(dIR)
print(dOI)
print(dOR)

print(meanII)
print(meanIR)
print(meanOI)
print(meanOR)

maxIn=np.maximum(maxII,maxIR)
maxOt=np.maximum(maxOI,maxOR)

# Read *.mat data files and data normalization
def read_mat(filepath):
    matfile = sio.loadmat(filepath)
    inpI = (matfile['inputI']-meanminII)/dII
    inpR = (matfile['inputR']-meanminIR)/dIR 
    oupI = (matfile['outputI']-meanminOI)/dOI
    oupR = (matfile['outputR']-meanminOR)/dOR
    
    shape0 = matfile['inputI'].shape
    shape1 = matfile['outputI'].shape

    inp = np.squeeze(np.reshape(np.concatenate([np.reshape(inpI,[1,110,128]),np.reshape(inpR,[1,110,128])],0),[-1,2,110,128]))
    out = np.squeeze(np.reshape(np.concatenate([np.reshape(oupI,[1,110,86]),np.reshape(oupR,[1,110,86])],0),[-1,2,110,86]))
    
    inp=np.array(inp,dtype='float32')
    out=np.array(out,dtype='float32')
    inp=inp[:,20:84,20:84]
    out=out[:,20:84,20:84]
    return inp, out

Esp2=3.340347928039855e+02
Esp_pre = Esp2/dOR/dOR
SNR_dB = 100
snr = 10**(SNR_dB/10)
nlevel = np.sqrt(Esp_pre/snr) 
#print(np.sqrt(Esp2/snr)/dOR)
#print(np.sqrt(Esp_pre/snr))


# ## Dataset class for PyThorch dataloader
class USCTDataset(Dataset):
    def __init__(self, in_df, transform=None, mode='train'):
        
        self.image_ids =  in_df
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.image_ids)
               
    def __getitem__(self, idx):
        input_file_name = self.image_ids.loc[idx][0]
        if self.mode == 'train':
            file_path = os.path.join(train_image_dir, input_file_name)
        elif self.mode == 'valid':
            file_path = os.path.join(train_image_dir, 'valid', input_file_name)
        else:
            if input_file_name[-4:]=='.mat':
                file_path = os.path.join(test_image_dir, input_file_name)
            else:
                file_path = os.path.join(test_image_dir, input_file_name+'.mat')
        data = read_mat(file_path)#
        input = data[0]
        res = data[1]

        if self.transform is not None:
            input, res = self.transform(input, res)

        if res is not None:           
            if res.ndim == 2:
                res = np.expand_dims(res, axis=2)            
        if input is not None:           
            if input.ndim == 2:
                input = np.expand_dims(input, axis=2)                   
                
        return input, res


# # Augment Data

def clip(x, dtype, maxval):
    return np.clip(x, 0, maxval).astype(dtype)

class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, yy=None):
        for t in self.transforms:
            x, yy = t(x, yy)
            if yy is not None:           
                if yy.ndim == 2:
                    yy = np.expand_dims(yy, axis=2)
        return x, yy
    
class OneOf:
    def __init__(self, transforms, prob=0.5):
        self.transforms = transforms
        self.prob = prob

    def __call__(self, x, yy=None):
        if random.random() < self.prob:
            t = random.choice(self.transforms)
            t.prob = 1.
            x, yy = t(x, yy)

            if yy is not None:           
                if yy.ndim == 2:
                    yy = np.expand_dims(yy, axis=2)
        return x, yy

class OneOrOther:
    def __init__(self, first, second, prob=0.5):
        self.first = first
        first.prob = 1.
        self.second = second
        second.prob = 1.
        self.prob = prob

    def __call__(self, x, yy=None):
        if random.random() < self.prob:
            x, yy = self.first(x, yy)
        else:
            x, yy = self.second(x, yy)
        if yy is not None:           
            if yy.ndim == 2:
                yy = np.expand_dims(yy, axis=2)
        return x, yy

class RandomNoise:
    def __init__(self, limit=0.1, prob=0.7):
        self.limit = limit
        self.prob = prob

    def __call__(self, x, yy=None):
        h, w, c = x.shape
        dtype = x.dtype
        #print(dtype)
        if random.random() < self.prob:
            SNR_dB = 112 + random.random()*30 
            snr = 10**(SNR_dB/10)
            nlevel = np.sqrt(Esp_pre/snr)
            gauss = np.random.normal(0,nlevel,(h,w,c))
            gauss = gauss.reshape(h,w,c)
            x = x + gauss         

        return x.astype('float32') ,yy

train_transform = DualCompose([
    RandomNoise()
])

val_transform = DualCompose([
    RandomNoise()
      ])


# # Data loader
def make_loader(in_df, batch_size, shuffle=False, transform=None, mode = 'train'):
        return DataLoader(
            dataset=USCTDataset(in_df, transform=transform, mode = mode),
            shuffle=shuffle,
            num_workers =0,
            batch_size = batch_size,
            pin_memory=torch.cuda.is_available()
        )

train_loader = make_loader(train_df, batch_size =  BATCH_SIZE, shuffle=True, transform=train_transform)
valid_loader = make_loader(valid_df, batch_size = BATCH_SIZE//2 , transform=None, mode='valid')


# # Network model
class _DUS_Block(nn.Module):
    def __init__(self):
        super(_DUS_Block, self).__init__()

        #res1
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.PReLU()
        #res1
        #concat1

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu6 = nn.PReLU()

        #res2
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu8 = nn.PReLU()
        #res2
        #concat2

        self.conv9 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu10 = nn.PReLU()

        #res3
        self.conv11 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu12 = nn.PReLU()
        #res3

        self.conv13 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.up14 = nn.PixelShuffle(2)

        #concat2
        self.conv15 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)
        #res4
        self.conv16 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu17 = nn.PReLU()
        #res4

        self.conv18 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.up19 = nn.PixelShuffle(2)

        #concat1
        self.conv20 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        #res5
        self.conv21 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu22 = nn.PReLU()
        self.conv23 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu24 = nn.PReLU()
        #res5

        self.conv25 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        res1 = x
        out = self.relu4(self.conv3(self.relu2(self.conv1(x))))
        out = torch.add(res1, out)
        cat1 = out

        out = self.relu6(self.conv5(out))
        res2 = out
        out = self.relu8(self.conv7(out))
        out = torch.add(res2, out)
        cat2 = out

        out = self.relu10(self.conv9(out))
        res3 = out

        out = self.relu12(self.conv11(out))
        out = torch.add(res3, out)

        out = self.up14(self.conv13(out))

        out = torch.cat([out, cat2], 1)
        out = self.conv15(out)
        res4 = out
        out = self.relu17(self.conv16(out))
        out = torch.add(res4, out)

        out = self.up19(self.conv18(out))

        out = torch.cat([out, cat1], 1)
        out = self.conv20(out)
        res5 = out
        out = self.relu24(self.conv23(self.relu22(self.conv21(out))))
        out = torch.add(res5, out)

        out = self.conv25(out)
        out = torch.add(out, res1)

        return out
    
class Res_Block16(nn.Module):
    def __init__(self):
        super(Res_Block16, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.PReLU()

        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu6= nn.PReLU()
        self.conv7 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu8 = nn.PReLU()

        self.conv9 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu10 = nn.PReLU()
        self.conv11 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu12 = nn.PReLU()

        self.conv13 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu14 = nn.PReLU()
        self.conv15 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu16 = nn.PReLU()

        self.conv17 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        res1 = x
        output = self.relu4(self.conv3(self.relu2(self.conv1(x))))
        output = torch.add(output, res1)

        res2 = output
        output = self.relu8(self.conv7(self.relu6(self.conv5(output))))
        output = torch.add(output, res2)

        res3 = output
        output = self.relu12(self.conv11(self.relu10(self.conv9(output))))
        output = torch.add(output, res3)

        res4 = output
        output = self.relu16(self.conv15(self.relu14(self.conv13(output))))
        output = torch.add(output, res4)

        output = self.conv17(output)
        output = torch.add(output, res1)

        return output  
    
class Res_Block64(nn.Module):
    def __init__(self):
        super(Res_Block64, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.PReLU()

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu6= nn.PReLU()
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu8 = nn.PReLU()

        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu10 = nn.PReLU()
        self.conv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu12 = nn.PReLU()

        self.conv13 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu14 = nn.PReLU()
        self.conv15 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu16 = nn.PReLU()

        self.conv17 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        res1 = x
        output = self.relu4(self.conv3(self.relu2(self.conv1(x))))
        output = torch.add(output, res1)

        res2 = output
        output = self.relu8(self.conv7(self.relu6(self.conv5(output))))
        output = torch.add(output, res2)

        res3 = output
        output = self.relu12(self.conv11(self.relu10(self.conv9(output))))
        output = torch.add(output, res3)

        res4 = output
        output = self.relu16(self.conv15(self.relu14(self.conv13(output))))
        output = torch.add(output, res4)

        output = self.conv17(output)
        output = torch.add(output, res1)

        return output  
class Res_Block128(nn.Module):
    def __init__(self):
        super(Res_Block128, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.PReLU()

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu6= nn.PReLU()
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu8 = nn.PReLU()

        self.conv9 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu10 = nn.PReLU()
        self.conv11 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu12 = nn.PReLU()

        self.conv13 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu14 = nn.PReLU()
        self.conv15 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu16 = nn.PReLU()

        self.conv17 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        res1 = x
        output = self.relu4(self.conv3(self.relu2(self.conv1(x))))
        output = torch.add(output, res1)

        res2 = output
        output = self.relu8(self.conv7(self.relu6(self.conv5(output))))
        output = torch.add(output, res2)

        res3 = output
        output = self.relu12(self.conv11(self.relu10(self.conv9(output))))
        output = torch.add(output, res3)

        res4 = output
        output = self.relu16(self.conv15(self.relu14(self.conv13(output))))
        output = torch.add(output, res4)

        output = self.conv17(output)
        output = torch.add(output, res1)

        return output    
    
class Res_Block(nn.Module):
    def __init__(self):
        super(Res_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.PReLU()

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu6= nn.PReLU()
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu8 = nn.PReLU()

        self.conv9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu10 = nn.PReLU()
        self.conv11 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu12 = nn.PReLU()

        self.conv13 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu14 = nn.PReLU()
        self.conv15 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu16 = nn.PReLU()

        self.conv17 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        res1 = x
        output = self.relu4(self.conv3(self.relu2(self.conv1(x))))
        output = torch.add(output, res1)

        res2 = output
        output = self.relu8(self.conv7(self.relu6(self.conv5(output))))
        output = torch.add(output, res2)

        res3 = output
        output = self.relu12(self.conv11(self.relu10(self.conv9(output))))
        output = torch.add(output, res3)

        res4 = output
        output = self.relu16(self.conv15(self.relu14(self.conv13(output))))
        output = torch.add(output, res4)

        output = self.conv17(output)
        output = torch.add(output, res1)

        return output

########################################################################
class mWnet(nn.Module):
    def __init__(self):
        super(mWnet, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1i = nn.PReLU()
        self.conv_down = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu2i = nn.PReLU()

        self.conv_down1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu2 = nn.PReLU()

        self.u_1 = _DUS_Block()
        self.u_2 = _DUS_Block()
        self.u_3 = _DUS_Block()
        self.u_4 = _DUS_Block()
                
        self.res64 = Res_Block64()
        self.res128 = Res_Block128()
        self.res = Res_Block()
        #concat
        self.conv_mid = nn.Conv2d(in_channels=1280, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)####
        self.relu3 = nn.PReLU()
        self.conv256512 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu4 = nn.PReLU()

        self.subpixel1 = nn.PixelShuffle(2)
        self.conv_output1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.convone128256 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.subpixel = nn.PixelShuffle(2)
        self.convone12864 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_output = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv00 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv01 = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv02 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        
        
        self.relu00 = nn.PReLU()
        self.relu01 = nn.PReLU()
        self.relu02 = nn.PReLU()
        
        
    def forward(self, x):
        x = x.float()
        x = nn.functional.pad(x,[0,0,9,9],"constant",value=0)
        out = self.relu1i(self.conv_input(x))
        out = self.res64(out)
        cat1 = out

        out = self.relu2i(self.conv_down(out))
      
        out = self.res128(out)
        cat2 = out

        out = self.relu2(self.conv_down1(out))
        
        out1 = self.u_1(out)
        out11 = torch.cat([out1, out], 1)
        out11 = self.relu00(self.conv00(out11))
        
        out2 = self.u_2(out11)
        out22 = torch.cat([out2, out1, out], 1)
        out22 = self.relu01(self.conv01(out22))
        
        out3 = self.u_3(out22)
        out33 = torch.cat([out3, out2, out1, out], 1)
        out33 = self.relu02(self.conv02(out33))

        out4 = self.u_4(out33)

        out = torch.cat([out, out1, out2, out3, out4], 1)

        out = self.relu3(self.conv_mid(out))
        
        out = self.res(out)

        out = self.conv256512(out)
        out= self.subpixel1(out)
        out = torch.cat([cat2,out],1)
        out = self.conv_output1(out)
        
        out = self.res128(out)
        out = self.convone128256(out)

        out= self.subpixel(out)

        out = torch.cat([cat1, out],1)
        out = self.convone12864(out)
        out = self.res64(out)
        out = self.conv_output(out)
        
        out = out[:,:,9:119,22:108]
        out0 = out[:,0, :,:]
        out1 = out[:,1, :,:]

        return out0, out1

model = mWnet() #

params = list(model.parameters())
k = 0
for i in params:
    l = 1
    print("The structure in this layer：" + str(list(i.size())))
    for j in i.size():
        l *= j
    print("Parameter number of this layer：" + str(l))
    k = k + l
print("The total parameter number：" + str(k))

# ## Validation routine

def validation(model: nn.Module, criterion, valid_loader):
    print("Validation on hold-out....")
    model.eval()
    losses = []
    jaccard = []
    ious = []
    for inputs, targets in valid_loader:
        inputs = variable(inputs, volatile=True)
        targets = variable(targets)
        outputs, o2 = model(inputs)
        loss = criterion(outputs,o2, targets)
        losses.append(loss.data)
    
    valid_loss = np.mean([x.item() for x in losses])#np.mean(losses)  # type: float

    print('Valid loss: {:.5f}'.format(valid_loss))
    metrics = {'valid_loss': valid_loss}
    return metrics


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

# some helper functions
def variable(x, volatile=False):
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(Variable(x, volatile=volatile))

def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x

def write_event(log, lr, step: int, **data):
    data['lr'] = lr
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True, cls=MyEncoder))
    log.write('\n')
    log.flush()

# ## LOSS
def criterionL(output_1, output_2, truth_pixel, is_average=True):

    bat,dim,m,n=truth_pixel.shape

    loss1 = nn.L1Loss()(output_1,truth_pixel[:,0,:,:])
 
    loss2 = nn.L1Loss()(output_2,truth_pixel[:,1,:,:])

    weight_1, weight_2 = 0.1, 0.9  

    return weight_1*loss1+ weight_2*loss2


# # ****Trainer Function****
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]

    assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr

def save_checkpoint(state, is_best, filename='model_1.pt'):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print ("=> Saving a new best")
        torch.save(state, filename) # save checkpoint
    else:
        print ("=> Validation Accuracy did not improve")


def train2ac(lr, model,train_loader, valid_loader,criterion, validation, init_optimizer, n_epochs=1, fold=1, accumulation_steps=4):
    
    optimizer = init_optimizer(lr)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5,patience=20)################

    if torch.cuda.is_available():
        model.cuda()
    isrestore = False   
    startepoch = 0
    model_path = Path('model_{fold}.pt'.format(fold=fold))
    model_path_trained = Path('model_1.pt')
    if model_path_trained.exists():
        state = torch.load(str(model_path_trained))

        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
        isrestore = True
        startepoch = epoch
        
        valid_metrics = validation(model, criterion, valid_loader)
        valid_loss = valid_metrics['valid_loss']

        best_accuracy=valid_loss
    else:
        epoch = 0
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))
    

    report_each = 50
    log = open('train_{fold}.log'.format(fold=fold),'at', encoding='utf8')
    valid_losses = []
    for epoch in range(epoch+1, epoch+n_epochs + 1):
        model.train()

        tq = tqdm(total=len(train_loader) *  BATCH_SIZE )
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):

                inputs, targets = variable(inputs), variable(targets)

                outputs,o2 = model(inputs)
                loss = criterion(outputs, o2, targets)
                loss.cuda()

                batch_size = inputs.size(0)

                loss.backward()
                
                if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
                    optimizer.step()
                    model.zero_grad() 
                    step += 1
                    tq.update(batch_size*accumulation_steps)############19
                    losses.append(loss.data)

                    mean_loss = np.mean([x.item() for x in losses[-report_each:]],dtype=np.float32)

                    tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                    if i and i % report_each == 0:
                        write_event(log, lr, step, loss=mean_loss)
                        
            write_event(log, lr, step, loss=mean_loss)########################lr
            tq.close()

            valid_metrics = validation(model, criterion, valid_loader)
            write_event(log, lr, step, **valid_metrics)#############################
            valid_loss = valid_metrics['valid_loss']

            acc= valid_loss#
            
            if epoch==1:
                best_accuracy=valid_loss

                is_best=True
                if is_best:
                    print ("=> Saving a new best")
                    save(epoch) # save checkpoint
                else:
                    print ("=> Validation Accuracy did not improve")
            else:
                #if isrestore == False:
                is_best = bool(acc < best_accuracy)
                best_accuracy = min(acc, best_accuracy)
                if is_best:
                    print ("=> Saving a new best")
                    save(epoch) # save checkpoint
                else:
                    print ("=> Validation Accuracy did not improve")
                #else:
                      
            valid_losses.append(valid_loss)
        except KeyboardInterrupt:
            tq.close()
            #print('Ctrl+C, saving snapshot')
            #save(epoch)
            print('done.')
            return


# # Training......

torch.cuda.is_available()

torch.cuda.device_count() 

#torch.cuda.set_device(1)

torch.cuda.current_device() 

torch.cuda.get_device_name(0)

torch.cuda.device(0)

if torch.cuda.is_available():
    model = model.cuda() 

train2ac(init_optimizer=lambda lr:torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False),
        lr = LR,
        n_epochs = NUM_epochs, 
        model=model,
        criterion=criterionL,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=validation,
        accumulation_steps=1,
     )


# ## Log visualization

log_file = 'train_1.log'
logs = pd.read_json(log_file, lines=True)

plt.figure(figsize=(26,6))
plt.subplot(1, 2, 1)
plt.plot(logs.step[logs.loss.notnull()],
            logs.loss[logs.loss.notnull()],
            label="on training set")
 
plt.plot(logs.step[logs.valid_loss.notnull()],
            logs.valid_loss[logs.valid_loss.notnull()],
            label = "on validation set")
         
plt.xlabel('step')
plt.legend(loc='center left')
plt.tight_layout()
plt.show();


# # Prediction

model = mWnet()#
model_path ='model_1.pt'
state = torch.load(str(model_path))
state = {key.replace('module.', ''): value for key, value in state['model'].items()}
model.load_state_dict(state)
if torch.cuda.is_available():
    model.cuda()

model.eval()

loader = DataLoader(
        dataset=USCTDataset(valid_df, transform=None, mode='valid'),###########################################################
        shuffle=False,
        batch_size=2,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    ) 

y_valid_ori = []
y_valid_oriR = []

out_pred_rows = []
out_pred_rowsR = []
for batch_num, (inputs, targets) in enumerate(tqdm(loader, desc='valid')):
    
    inputs = variable(inputs, volatile=True)
    
    outputs,outputsR = model(inputs)
    
    for i, tagts0 in enumerate(targets):
        mask = np.squeeze(outputs.detach().cpu().numpy()[i] )
        out_pred_rows.extend([mask.tolist()])
        
        maskR = np.squeeze(outputsR.detach().cpu().numpy()[i] )
        out_pred_rowsR.extend([maskR.tolist()])
        
        tagts = ([tagts0[0].detach().cpu().numpy()])
        y_valid_ori.extend(tagts)
        tagtsR = ([tagts0[1].detach().cpu().numpy()])
        y_valid_oriR.extend(tagtsR)


A=y_valid_ori[1999]*dOI+meanminOI
B=np.squeeze(preds_valid[1999])*dOI+meanminOI
ax=None
mse = (np.square(A - B)).mean(axis=ax)
print(mse)


A=y_valid_oriR[1999]*dOR+meanminOR
B=np.squeeze(preds_validR[1999])*dOR+meanminOR
ax=None
mse = (np.square(A - B)).mean(axis=ax)
print(mse)

