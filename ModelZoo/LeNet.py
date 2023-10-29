import torch
import torch.nn as nn


class Lenet(nn.Module):
    def __init__(self):
        super(Lenet,self).__init__()
        self.conv1=nn.Conv2d(3,16,3,1,1)
    def forward(self,x):
        x=self.conv1(x)
        return x

