# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 17:22:41 2021

@author: USER
"""

import torch
import torch.utils.data as data
from torchvision.transforms import transforms
from PIL import Image
import os
import numpy as np

data_transform = transforms.Compose([transforms.ToTensor()])

img_H = 224
img_W = 224

class DS(data.Dataset):
    def __init__(self,mode,dir):
        self.mode = mode
        self.list_img = []
        self.list_label = []
        self.data_size = 0
        self.transform = data_transform
        
        if self.mode == 'train':
            for file in os.listdir(dir):
                self.list_img.append(dir + file)
                self.data_size += 1
                name = file.split(sep='.')
                if name[0] == 'cracked':
                    self.list_label.append(0)
                else:
                    self.list_label.append(1)
        elif self.mode == 'test':
            for file in os.listdir(dir):
                self.list_img.append(dir + file)
                self.data_size += 1
                self.list_label.append(2)
        else:
            print('Undefined')
            
    def __getitem__(self,item):
        if self.mode == 'train':
            img = Image.open(self.list_img[item])
            img = img.resize((img_H,img_W))
            img = np.array(img)[:,:,:3]
            label = self.list_label[item]
            return self.transform(img),torch.LongTensor([label])
        elif self.mode == 'test':
            img = Image.open(self.list_img[item])
            img = img.resize((img_H,img_W))
            img = np.array(img)[:,:,:3]
            label = self.list_label[item]
            return self.transform(img)
        else:
            print('None')
            
    def __len__(self):
        return self.data_size
    
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_r = nn.Conv2d(64, 64, 4, padding=1, stride=2)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_r = nn.Conv2d(128, 128, 4, padding=1, stride=2)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_r = nn.Conv2d(256, 256, 4, padding=1, stride=2)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_r2 = nn.Conv2d(256, 256, 4, padding=1, stride=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 2)
        
    def forward(self,x):
        out = self.conv1_1(x)
        out = F.relu(out)
        out = self.conv1_2(out)
        out = F.relu(out)
        out = F.max_pool2d(out,2)
        
        out = self.conv2_1(out)
        out = F.relu(out)
        out = self.conv2_2(out)
        out = F.relu(out)
        out = F.max_pool2d(out,2)
        
        out = self.conv3_1(out)
        out = F.relu(out)
        out = self.conv3_2(out)
        out = F.relu(out)
        out = self.conv3_3(out)
        out = F.relu(out)
        out = F.max_pool2d(out,2)
        
        out = out.view(out.size()[0],-1)
        out = self.fc1(out)
        out = F.dropout(out,p=0.5)
        out = F.relu(out)
        out = self.fc2(out)
        
        return F.softmax(out,dim = 1)
    
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader as DataLoader
import os
​
datadir = '../input/confusion-matrix/test_nocrack/test_nocrack/'
​
def confusion_matrix():
    
    model = Net()
    model.cuda()
    model = torch.load('../input/crackmodel-cp2-0000164/model_cp2_0.000164.pkl')
    model.eval()
    
    datafile = DS('test',datadir)
    print(len(datafile))
    
    TN = 0
    FP = 0
    for index in range(len(datafile)):
        #index = np.random.randint(0,datafile.data_size,1)[0]
        img = datafile.__getitem__(index)
        img = img.unsqueeze(0)
        img = Variable(img).cuda()
        out = model(img)
        #print(out)
        if out[0,0] > out[0,1]:
            #print('surface with cracks')
            FP += 1
        else:
            #print('surface without cracks')
            TN += 1
    
    print('FP = {}, TN = {}'.format(FP, TN))
    
if __name__=='__main__':
    confusion_matrix()