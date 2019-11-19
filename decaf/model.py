from __future__ import print_function, division
import os
import torch
import numpy as np
import csv
import torchvision 
import torchvision.models as models
import torch.nn as nn


class VGG(nn.Module):

    def __init__(self, vgg, num_classes=5, init_weights=True):
        super(VGG, self).__init__()
        self.features = vgg.features[:28]

    def forward(self, x):
        #print(self.features)
        x = self.features(x)
        
        x = torch.flatten(x, 1)
        return x



cfgs = {
    #'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'] standard vgg19
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 'M']
}

def _vgg(arch, cfg, batch_norm, pretrained):
    vggX = models.vgg19(pretrained=True)
    
    model = VGG(vggX)
    
    return model

def vgg19_decaf(pretrained=True):

    return _vgg('vgg19', 'E', False, pretrained)
