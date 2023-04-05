import os, sys, time, random
import torch, torchvision

import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim

class CNN_Model1(nn.Module):
    def __init__(self, in_channels, img_size, out_nodes):
        super(CNN_Model1, self).__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.out_nodes = out_nodes
        self.convlayer1 = 10
        self.convlayer2 = 20
        
        # Initializing the convolutional layers and full-connected layers
        self.encConv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.convlayer1, kernel_size=3, padding=1)
        self.encConv2 = nn.Conv2d(in_channels=self.convlayer1, out_channels=self.convlayer2, kernel_size=3, padding=1)
        self.encFC1 = nn.Linear(self.convlayer2*self.img_size[0]*self.img_size[1], self.out_nodes*2)

    def forward(self, x):
        a = 1