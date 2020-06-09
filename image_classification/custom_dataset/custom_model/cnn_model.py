# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 22:13:11 2020

@author: SS
"""
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        self.conv1 = nn.Conv2d(3, 16, 3, padding= 1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding= 1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding= 1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding= 1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding = 1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        
        self.conv_bn1 = nn.BatchNorm2d(224, 3)
        self.conv_bn2 = nn.BatchNorm2d(16)
        self.conv_bn3 = nn.BatchNorm2d(32)
        self.conv_bn4 = nn.BatchNorm2d(64)
        self.conv_bn5 = nn.BatchNorm2d(128)
        self.conv_bn6 = nn.BatchNorm2d(256)
        
        self.fc1 = nn.Linear(256*7*7, 512)
        self.fc2 = nn.Linear(512, 133)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv_bn2(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv_bn3(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.conv_bn4(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.conv_bn5(x)
        x = self.pool(F.relu(self.conv5(x)))
        x = self.conv_bn6(x)
        
        x = x.view(-1, 256*7*7)
        x = self.dropout(x)
        x = F.relu(self.fc1(x)) 
        x = self.dropout(x)    
        x = self.fc2(x)
        
        return x