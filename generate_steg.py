"""
Generate examples of steganography with LSB, DDH, and/or UDH.

"""

from utils.StegoPy import encode_img, decode_img, encode_msg, decode_msg
from load_models import load_ddh, load_udh, load_vae_sani
from run_stats import load_data

import tensorflow as tf
import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.cnn_vae import CNN_VAE
from PIL import Image

import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import random
import os
import copy
from torch.nn.functional import normalize
from skimage.util import random_noise

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# from data.ImageFolderDataset import MyImageFolder
from models.HidingUNet import UnetGenerator
from models.RevealNet import RevealNet
from main import weights_init

from torchvision.utils import save_image

np.random.seed(4)
random.seed(4)
# torch.manual_seed(4)