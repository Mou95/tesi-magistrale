from __future__ import print_function, division
import os
import torch
import numpy as np
import csv
import torchvision 
import torchvision.models as models
import torch.nn as nn


class VGG(nn.Module):

    def __init__(self, vggX, vggY, num_classes=5, init_weights=True):
        super(VGG, self).__init__()
        self.featuresX = vggX.features[:28]
        self.featuresY = vggY.features[:28]
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7 * 2, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x, y):
        x = self.featuresX(x)
        y = self.featuresY(y)
        x = self.avgpool(x)
        y = self.avgpool(y)
        N ,_,_,_ = x.size()
        x = x.view(N,-1)
        y = y.view(N,-1)
        z = torch.cat((x,y),1)
        #x = torch.flatten(x, 1)
        z = self.classifier(z)
        return z

    def _initialize_weights(self):
        for m in self.modules():
            '''if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            el'''
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


'''def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)'''


cfgs = {
    #'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'] standard vgg19
    'E' : [64, 64, 'M', 64, 64, 'M', 64, 64, 64, 64, 'M']
}

def _vgg(arch, cfg, batch_norm, pretrained):
    vggX = models.vgg19(pretrained=True)
    vggY = vggX
    
    model = VGG(vggX, vggY)
    
    return model

def vgg19_multiI(pretrained=True):

    return _vgg('vgg19', 'E', False, pretrained)
