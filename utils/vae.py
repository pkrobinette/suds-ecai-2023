"""
Convolutional neural network version of VAE.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# using a 32x32 image
class CNN_VAE(nn.Module):
    """
    VAE implementation using convolutions in encoder and decoder.
    """
    def __init__(self, c_in=3, k_num=128, z_size=128, im_size=32):
        super(CNN_VAE, self).__init__()
        
        # ==================
        # ENCODER
        # ==================
        self.enc_conv1 = _conv(c_in, k_num//4)     # leakyrelu baked in
        self.enc_conv2 = _conv(k_num//4, k_num//2) # leakyrelu baked in
        self.enc_conv3 = _conv(k_num//2, k_num)    # leakyrelu baked in
        
        self.f_size = im_size // 8 # 32 for image size
        self.f_dim = k_num * (self.f_size ** 2)
        self.k_num = k_num
        
        # fdim taken by looking at the shape after each conv. layer
        self.enc_fc1 = _linear(self.f_dim, z_size, lrelu=False) # one for mean
        self.enc_fc2 = _linear(self.f_dim, z_size, lrelu=False) # one for logvar
        
        # ==================
        # DECODER
        # ==================
        self.dec_fc1 = _linear(z_size, self.f_dim, lrelu=False)
        
        self.dec_conv1 = _deconv(k_num, k_num//2)
        self.dec_conv2 = _deconv(k_num//2, k_num//4)
        self.dec_conv3 = _deconv(k_num//4, c_in)
        
    def encode(self, x):
        """
        Encode an image to a latent variable z.
        
        Parameters
        ----------
        x : img
        
        Returns
        -------
        z : latent variable
            a sample from the learned distributions
        """
        mu, logvar = self.encode_train(x)
        return self.sample_z(mu, logvar)
        
    def encode_train(self, x):
        """
        Encode to distributions.
        """
        x = self.enc_conv1(x)
        x = self.enc_conv2(x)
        x = self.enc_conv3(x)
        x = x.view(-1, self.f_dim)
        
        mu = self.enc_fc1(x)
        logvar = self.enc_fc2(x)
        
        return mu, logvar
    
    def sample_z(self, mu, logvar):
        """
        Sample from the distributions.
        """
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        
        return mu + std*eps
    
    def decode(self, z):
        """
        Decode an image from a latent variable z.
        
        Parameters
        ----------
        z : z_size tensor
            latent variable
        
        Returns
        -------
        Normalized image, x_hat. Using tanh and then normalizing. Seems to get closer to actual color. sigmoid had dark color when used as activation function.
        """
        x = self.dec_fc1(z)
        x = x.view(-1, self.k_num,
            self.f_size,
            self.f_size,)
        x = self.dec_conv1(x)
        x = self.dec_conv2(x)
        x = self.dec_conv3(x)
        # trying a normalization right here and tanh. Previous was just sigmoid
        x = torch.tanh(x) # sigmoid instead of tanh?
        
        # normalize and return
        return (x - x.min()) / (x.max() - x.min())
    
    def forward_train(self, x):
        """
        Return x_hat and latent variable distribution -- the outputs from decoder and encoder respectively.
        """
        mu, logvar = self.encode_train(x)
        z = self.sample_z(mu, logvar) # take a sample
        out = self.decode(z)
        
        return out, mu, logvar
        

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

def _deconv(channel_num, kernel_num):
    return nn.Sequential(
        nn.ConvTranspose2d(
            channel_num, kernel_num,
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

