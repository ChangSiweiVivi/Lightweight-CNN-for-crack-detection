# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 14:21:43 2021

@author: ChangSiwei
"""

# prepare dataset
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
    
#build model
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1_1 = nn.Conv2d(3,64,3,padding = 1)
        self.conv1_2 = nn.Conv2d(64,64,3,padding = 1)
        
        self.conv2_1 = nn.Conv2d(64,128,3,padding = 1)
        self.conv2_2 = nn.Conv2d(128,128,3,padding = 1)
                
        self.conv3_1 = nn.Conv2d(128,256,3,padding = 1)
        self.conv3_2 = nn.Conv2d(256,256,3,padding = 1)
        self.conv3_3 = nn.Conv2d(256,256,3,padding = 1)
                    
        self.fc1 = nn.Linear(256*28*28,1024)
        self.fc2 = nn.Linear(1024,2)
        
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
    
#train and test model
from torch.utils.data import DataLoader 
from torch.autograd import Variable
import matplotlib.pyplot as plt
from numpy import *


datadir_train = '../input/crack-detection/crack_train/train/'
datadir_test = ''
model_cp = './'
workers = 10
lr = 0.0001
batch_size = 64


def train():
    datafile_train = DS('train',datadir_train)
    dataloader_train = DataLoader(datafile_train,batch_size = batch_size, shuffle = True, num_workers = workers,drop_last = True)

    print(len(datafile_train))

    
    model = Net()
    #model = torch.load('../input/crackmodel-cp2-0000164/model_cp2_0.000164.pkl')
    model.cuda()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
   
    Loss_epoch = []
    acc_epoch = []
    for i in range(30):
        Loss = []
        acc = []
        total = 0
        correct = 0
        for img, label in dataloader_train:
            img, label = Variable(img).cuda(), Variable(label).cuda()
            out = model(img)
            loss = criterion(out,label.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Loss.append(loss.data.cpu().numpy())
            #print('Frame {0}, train_loss {1}'.format(i*batch_size, loss)) 
            _,predicted = torch.max(out.data,1)
            total = label.size(0)
            correct = (predicted == label).sum().item()          
            acc.append(correct/total)
                
        Loss_epoch.append(mean(Loss))
        acc_epoch.append(mean(acc))
        print('epoch{0}, train_loss{1}, train_accuray'.format(i+1, Loss_epoch,acc_epoch))
            
    torch.save(model, 'cp2_3.pkl')
    plt.figure('efigure')
    plt.plot(epoch,Loss_epoch)
    plt.plot(epoch,acc_epoch)
    plt.show()
    
    
if __name__ == '__main__':
    train()
    
#test
def test():
    datafile_test = DS('test',datadir_test)
    dataloader_test = DataLoader(datafile_test,batch_size = batch_size, shuffle = False, num_workers = workers,drop_last = True)
    print(len(datafile_test))
    
    model = Net()
    model.cuda()
    model = torch.load('cp2_3.pkl')
    model.eval()
    
    acc_test_epoch = []
    for img, label in dataloader_test:
        out = model(img)
        _,predicted = torch.max(out.data,1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        acc_test_epoch.append(correct/total)
    
    print('test_accuracy{0}'.format(acc_test_epoch))
    
if __name__ == '__main__':
    test()
        
###anothor function
    
from torch.utils.data import DataLoader 
from torch.autograd import Variable
import matplotlib.pyplot as plt
from numpy import *
import numpy as np


datadir_train = '../input/crack-detect/train/'
datadir_test = '../input/crack-detect/test/'
model_cp = './'
workers = 10
lr = 0.0001
batch_size = 64


def train_test_accuracy():
    datafile_train = DS('train',datadir_train)
    dataloader_train = DataLoader(datafile_train,batch_size = batch_size, shuffle = True, num_workers = workers, drop_last = True)
    datafile_test = DS('test',datadir_test)
    dataloader_test = DataLoader(datafile_test,batch_size = batch_size, shuffle = False, num_workers = workers, drop_last = True)

    print(len(datafile_train))
    print(len(datafile_test))

    
    model = Net()
    model.cuda()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
   
    Loss_epoch = []
    acc_epoch = []
    epoch = []
    for i in range(30):
        Loss = []
        acc = []
        for img, label in dataloader_train:
            img, label = Variable(img).cuda(), Variable(label).cuda()
            label_list = label.data.cpu().numpy().tolist()
            #print(label_list)
            out = model(img)
            loss = criterion(out,label.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Loss.append(loss.data.cpu().numpy())
            #print('Frame {0}, train_loss {1}'.format(i*batch_size, loss)) 
            _, predicted = torch.max(out.data,1)
            predicted_list = predicted.data.cpu().numpy().tolist()
            predicted_list_split = [predicted_list[i:i+1] for i in range(0,len(predicted),1)]
            #print(predicted_list_split)
            right = []
            for j in range(len(predicted_list)):
                if label_list[j] == predicted_list_split[j]:
                    right.append('y')
            acc.append(len(right) / len(predicted_list))
            #print(acc)
                
        Loss_epoch.append(mean(Loss))
        acc_epoch.append(mean(acc))
        print('epoch{0}, train_loss{1}, train_accuray{2}'.format(i+1, Loss_epoch,acc_epoch))
        
        model.eval()
        acc_test_epoch = []
        with torch.no_grad():
            acc_test = []
            for img_t,label_t in dataloader_test:
                img_t, label_t = Variable(img).cuda(), Variable(label).cuda()
                label_t_list = label_t.data.cpu().numpy().tolist()
                out_t = model(img_t)
                _, predicted_t = torch.max(out_t.data, 1)
                predicted_t_list = predicted_t.data.cpu().numpy().tolist()
                predicted_t_split_list = [predicted_t_list[i:i+1] for i in range(0,len(predicted_t_list),1)]
                right_t = []
                for j in range(len(predicted_t_list)):
                    if label_t_list[j] == predicted_t_split_list[j]:
                        right_t.append('y')
                acc_test.append(len(right_t) / len(predicted_t_list))
                
        acc_test_epoch.append(mean(acc_test))
        print('epoch{0}, test_accuracy{1}'.format( i+1, acc_test_epoch))
        
        epoch.append(i+1)
            
    torch.save(model, model_cp + 'cp2_3.pkl')
    plt.figure('efigure')
    plt.plot(epoch,Loss_epoch)
    plt.plot(epoch,acc_epoch)
    plt.plot(epoch,acc_test_epoch)
    plt.show()
    
    
if __name__ == '__main__':
    train_test_accuracy() 