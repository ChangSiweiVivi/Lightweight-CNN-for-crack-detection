# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 18:25:06 2022

@author: USER
"""

import cv2
import torch
from torchvision.transforms import transforms
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import torch.nn.modules.batchnorm


'''build network'''
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1_1 = nn.Conv2d(3,64,3,padding = 1)
        self.conv1_r = nn.Conv2d(64,64,4, padding = 1, stride = 2)
        self.conv1_2 = nn.Conv2d(64,64,3,padding = 1)
        
        self.conv2_1 = nn.Conv2d(64,128,3,padding = 1)
        self.conv2_r = nn.Conv2d(128,128,4, padding = 1, stride = 2)
        self.conv2_2 = nn.Conv2d(128,128,3,padding = 1)
        
        self.conv3_1 = nn.Conv2d(128,256,3,padding = 1)
        self.conv3_r = nn.Conv2d(256,256,4, padding = 1, stride = 2)
        self.conv3_2 = nn.Conv2d(256,256,3,padding = 1)
        self.conv3_r2 = nn.Conv2d(256,256,4, padding=1, stride = 1)
        self.conv3_3 = nn.Conv2d(256,256,3,padding = 1)
                
        self.fc1 = nn.Linear(256*3*3,1024)
        self.fc2 = nn.Linear(1024,2)
        
    def forward(self,x): 
        out = self.conv1_1(x)
        out = F.relu(out)
        out = self.conv1_r(out)
        out = F.relu(out)
        out = self.conv1_2(out)
        out = F.relu(out)
        out = F.max_pool2d(out,2)


        out = self.conv2_1(out)
        out = F.relu(out)
        out = self.conv2_r(out)
        out = F.relu(out)
        out = self.conv2_2(out)
        out = F.relu(out)
        out = F.max_pool2d(out,2)
        
        out = self.conv3_1(out)q
        out = F.relu(out)
        out = self.conv3_r(out)
        out = F.relu(out)
        out = self.conv3_2(out)
        out = F.relu(out)
        out = self.conv3_r2(out)
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
'''build network'''


'''test'''
model = Net()
model = torch.load(r'C:\Users\USER\Desktop\document\11 13 layer CNN\13-layer_75.pkl')

data_transform = transforms.Compose([transforms.ToTensor()])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (224,224))
    frame_t = np.array(frame)[:,:,:3]
    frame_t = data_transform(frame_t)
    frame_t = frame_t.unsqueeze(0)
    frame_t = Variable(frame_t).cuda()
    out = model(frame_t)
    if out[0,0]>out[0,1]:
        cv2.putText(frame,'Crack',(20,35),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,225),2)
    else:
        cv2.putText(frame,'No Crack',(20,35),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,225),2)
    
    '''for idx,f in enumerate(out):
        (SX,SY) = f[0], f[1]
        (EX,EY) = f[2], f[3]
        cv2.rectangle(frame,(SX,SY),(EX,EY), (0,255,0),2)
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    pad_x = max(frame_t.shape[0] - frame_t.shape[1], 0) * (224 / max(frame_t.shape))
    pad_y = max(frame_t.shape[1] - frame_t.shape[0], 0) * (224 / max(frame_t.shape))
    unpad_h = 224 - pad_y
    unpad_w = 224 - pad_x
    if out is not None:
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in out:
            box_h = ((y2 - y1) / unpad_h) * frame_t.shape[0]
            box_w = ((x2 - x1) / unpad_w) * frame_t.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * frame_t.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * frame_t.shape[1]
            bbox = cv2.Rectangle(frame_t, (x1, y1), (box_w, box_h), (0,255,0),2)'''
  
    frame = cv2.resize(frame,(1280,960))
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == ord('q'):
        break
    
cv2.destroyAllWindows()
cap.release()
'''test'''