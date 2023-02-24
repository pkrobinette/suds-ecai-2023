"""
Generate examples of steganography with LSB, DDH, and/or UDH.

"""

from utils.StegoPy import encode_img, decode_img, encode_msg, decode_msg
from utils.utils import load_udh_mnist, load_ddh_mnist, load_data, use_lsb, use_ddh, use_udh
from utils.vae import CNN_VAE

import tensorflow as tf
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image

import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import copy
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
np.random.seed(4)
random.seed(4)


def get_args():
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(description='Argument parser for ddh, udh, and lsb')
    
    parser.add_argument('--ddh', action='store_true', help='Enable DDH option')
    parser.add_argument('--udh', action='store_true', help='Enable UDH option')
    parser.add_argument('--lsb', action='store_true', help='Enable LSB option')
    parser.add_argument('--savedir', type=str, default="results/hiding_demo", help='The directory path to save demo imgs.')
    
    args = parser.parse_args()
    return args

def make_folder(path):
    """
    Creates a folder at path if one does not already exist.
    
    Parameters
    ---------
    path : str
        path of intended folder
    """
    if os.path.exists(path) == 0:
        os.mkdir(path)
        
def save_img(img, path):
    """
    Saves an image to the indicated path.
    
    Parameters
    ----------
    img : tensor
        an image tensor
    path : str
        path to save the image
    """
    if img.shape[0] == 1:
        img = img[0]
    elif img.shape[0] == 3:
        img = img.permute(1, 2, 0)
    
    transform = 255 if img.max() <= 1 else 1
    img_data = np.array(img*transform).astype(np.uint8)
    Image.fromarray(img_data).save(path+".jpg")
    return 0

def save_images(imgs, folder):
    """
    Saves a tensor of images into a specified folder by calling save_img
    
    Parameters
    ----------
    imgs : tensor
        a tensor of tensor images
    folder : str
        the overall directory of where to save the images
    """
    for i in range(len(imgs)):
        save_img(imgs[i], folder+"/"+str(i))
        
def norm(x):
    """
    Normalize function
    
    Parameters 
    ----------
    x : tensor
        image to normalize
    """
    z = (x - x.min())/(x.max() - x.min())
    return z

def main():
    """ main function """
    #
    # get args and check inputs
    #
    args = get_args()
    if not args.udh and not args.ddh and not args.lsb:
        print("Running everything")
        args.ddh = True
        args.lsb = True
        args.udh = True
    #
    # Get unique covers and secrets
    #
    train_loader, test_loader = load_data("mnist")
    inputs, labels = next(iter(test_loader))
    # make sure that cover label != secret label
    idxs = []
    for i in range(10):
        idx = np.random.randint(inputs.shape[0])
        while labels[i] == labels[idx]:
            idx = np.random.randint(inputs.shape[0])
        idxs.append(idx)
    covers = inputs[:10]
    secrets = inputs[idxs]
    # 
    # Run and save lsb
    # 
    make_folder(args.savedir)
    if args.lsb:
        print(f"Demoing LSB, imgs saved to: {args.savedir}/lsb_demo/")
        save_path = f"{args.savedir}/lsb_demo/"
        C_folder = "C"
        Cprime_folder = "C_prime"
        Cres_folder = "C_res"
        S_folder = "S"
        Sprime_folder = "S_prime"
        Sres_folder = "S_res"
        #
        # make save directories
        #
        make_folder(save_path)
        make_folder(save_path+C_folder)
        make_folder(save_path+Cprime_folder)
        make_folder(save_path+Cres_folder)
        make_folder(save_path+S_folder)
        make_folder(save_path+Sprime_folder)
        make_folder(save_path+Sres_folder)
        #
        # Generate steg 
        #
        containers, C_res, reveal_secret, S_res = use_lsb(covers, secrets)
        # 
        # Normalize and save
        # 
        C_res = norm(C_res)
        S_res = norm(S_res)
        save_images(covers, save_path+C_folder)
        save_images(secrets, save_path+S_folder)
        save_images(containers, save_path+Cprime_folder)
        save_images(reveal_secret, save_path+Sprime_folder)
        save_images(C_res, save_path+Cres_folder)
        save_images(S_res, save_path+Sres_folder)
    # 
    # Run and save ddh
    # 
    if args.ddh:
        print(f"Demoing DDH, imgs saved to: {args.savedir}/ddh_demo/")
        HnetD, RnetD = load_ddh_mnist()
        save_path = f"{args.savedir}/ddh_demo/"
        C_folder = "C"
        Cprime_folder = "C_prime"
        Cres_folder = "C_res"
        S_folder = "S"
        Sprime_folder = "S_prime"
        Sres_folder = "S_res"
        #
        # make save directories
        #
        make_folder(save_path)
        make_folder(save_path+C_folder)
        make_folder(save_path+Cprime_folder)
        make_folder(save_path+Cres_folder)
        make_folder(save_path+S_folder)
        make_folder(save_path+Sprime_folder)
        make_folder(save_path+Sres_folder)
        #
        # Generate steg 
        #
        containers, C_res, reveal_secret, S_res = use_ddh(covers, secrets, HnetD, RnetD)
        # 
        # Normalize and save
        # 
        C_res = norm(C_res)
        S_res = norm(S_res)
        save_images(covers, save_path+C_folder)
        save_images(secrets, save_path+S_folder)
        save_images(containers, save_path+Cprime_folder)
        save_images(reveal_secret, save_path+Sprime_folder)
        save_images(C_res, save_path+Cres_folder)
        save_images(S_res, save_path+Sres_folder)
    # 
    # Run and save udh
    # 
    if args.udh:
        print(f"Demoing UDH, imgs saved to: {args.savedir}/udh_demo/")
        Hnet, Rnet = load_udh_mnist()
        save_path = f"{args.savedir}/udh_demo/"
        C_folder = "C"
        Cprime_folder = "C_prime"
        Cres_folder = "C_res"
        S_folder = "S"
        Sprime_folder = "S_prime"
        Sres_folder = "S_res"
        #
        # make save directories
        #
        make_folder(save_path)
        make_folder(save_path+C_folder)
        make_folder(save_path+Cprime_folder)
        make_folder(save_path+Cres_folder)
        make_folder(save_path+S_folder)
        make_folder(save_path+Sprime_folder)
        make_folder(save_path+Sres_folder)
        #
        # Generate steg 
        #
        containers, C_res, reveal_secret, S_res = use_udh(covers, secrets, Hnet, Rnet)
        # 
        # Normalize and save
        # 
        C_res = norm(C_res)
        S_res = norm(S_res)
        containers = norm(containers) # have to normalize containers on this one as well
        save_images(covers, save_path+C_folder)
        save_images(secrets, save_path+S_folder)
        save_images(containers, save_path+Cprime_folder)
        save_images(reveal_secret, save_path+Sprime_folder)
        save_images(C_res, save_path+Cres_folder)
        save_images(S_res, save_path+Sres_folder)
        
if __name__ == "__main__":
    main()