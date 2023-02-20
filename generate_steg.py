"""
Generate examples of steganography with LSB, DDH, and/or UDH.

"""

from utils.StegoPy import encode_img, decode_img, encode_msg, decode_msg
from utils.utils import load_udh_mnist, load_ddh_mnist, load_data
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
    parser = argparse.ArgumentParser(description='Argument parser for ddh, udh, and lsb')
    
    parser.add_argument('--ddh', action='store_true', help='Enable DDH option')
    parser.add_argument('--udh', action='store_true', help='Enable UDH option')
    parser.add_argument('--lsb', action='store_true', help='Enable LSB option')
    parser.add_argument('--savedir', type=str, default="results/hiding_demo", help='The directory path to save demo imgs.')
    
    args = parser.parse_args()
    return args

def make_folder(path):
    if os.path.exists(path) == 0:
        os.mkdir(path)
        
def save_img(img, path):
    if img.shape[0] == 1:
        img = img[0]
    elif img.shape[0] == 3:
        img = img.permute(1, 2, 0)
    
    transform = 255 if img.max() <= 1 else 1
    img_data = np.array(img*transform).astype(np.uint8)
    Image.fromarray(img_data).save(path+".jpg")
    return

def save_images(imgs, folder):
    for i in range(len(imgs)):
        save_img(imgs[i], folder+"/"+str(i))
        
def norm(x):
    z = (x - x.min())/(x.max() - x.min())
    return z

def main():
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
        containers = encode_img(covers*255, secrets*255, train_mode=True) # steg function is on pixels [0, 255]
        C_res = containers/255 - covers
        reveal_secret = decode_img(containers, train_mode=True)
        S_res = abs(reveal_secret/255 - secrets)
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
        H_input = torch.cat((covers, secrets), dim=1)
        with torch.no_grad():
            containers = HnetD(H_input)
        
        C_res = containers - covers
        with torch.no_grad():
            reveal_secret = RnetD(containers)
        S_res = reveal_secret - secrets
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
        with torch.no_grad():
            C_res = Hnet(secrets)
        containers = covers + C_res
        with torch.no_grad():
            reveal_secret = Rnet(containers)
        S_res = reveal_secret - secrets
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
        
if __name__ == "__main__":
    main()