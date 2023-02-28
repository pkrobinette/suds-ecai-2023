"""
Convolutional neural network classifier..

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# using a 32x32 image
class Classifier(nn.Module):
    """
    VAE implementation using convolutions in encoder and decoder.
    """
    def __init__(self, c_in=1, k_num=128, class_num=10, im_size=32):
        super(Classifier, self).__init__()
        
        # model
        self.conv1 = _conv(c_in, k_num//4)     # leakyrelu baked in
        self.conv2 = _conv(k_num//4, k_num//2) # leakyrelu baked in
        self.conv3 = _conv(k_num//2, k_num)    # leakyrelu baked in
        
        self.f_size = im_size // 8 # 32 for image size
        self.f_dim = k_num * (self.f_size ** 2)
        self.k_num = k_num
        
        # fdim taken by looking at the shape after each conv. layer
        self.fc1 = _linear(self.f_dim, class_num, lrelu=False) # one for mean
        
        
    def forward(self, x):
        """
        Return classification given an image.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, self.f_dim)
        
        out = self.fc1(x)
        
        return out
    
    def get_predict(self, x):
        """
        Return class number.
        """
        out = self.forward(x)
        return torch.argmax(out, dim=1)

# =========================================
# Layers
# =========================================


def _conv(channel_size, kernel_num):
    return nn.Sequential(
        nn.Conv2d(
            channel_size, kernel_num,
            kernel_size=4, stride=2, padding=1,
        ),
        nn.BatchNorm2d(kernel_num),
        nn.LeakyReLU(),
    )

def _linear(in_size, out_size, lrelu=True):
    return nn.Sequential(
        nn.Linear(in_size, out_size),
        nn.LeakyReLU(),
    ) if lrelu else nn.Linear(in_size, out_size)