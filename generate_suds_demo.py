"""
Generate examples of sanitizing steganography using SUDS on LSB, DDH, and/or UDH.

"""

from utils.StegoPy import encode_img, decode_img, encode_msg, decode_msg
from utils.utils import load_udh_mnist, load_ddh_mnist, load_data, load_vae_suds, use_lsb, use_ddh, use_udh
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
    parser.add_argument('--savedir', type=str, default="results/sanitize_demo", help='The directory path to save demo imgs.')
    
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
        
def save_plot(imgs, folder):
    """ 
    Alternate save function.
    
    Parameters
    ----------
    imgs : tensor
        a tensor of tensor images
    folder : str
        the overall directory fo where to save the images
    """
    # Check on image render coloring
    maps = 'gray' if imgs.shape[1] == 1 else None
    for i in range(len(imgs)):
        plt.clf();
        plt.imshow(imgs[i].permute(1, 2, 0), cmap=maps);
        plt.axis("off");
        plt.tight_layout();
        plt.savefig(f"{folder}/img{i}.jpg", bbox_inches='tight');
        
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
    idxs_c = []
    idxs_s = []
    for i in range(10):
        c = np.random.randint(inputs.shape[0])
        s = np.random.randint(inputs.shape[0])
        while labels[c] == labels[s]:
            s = np.random.randint(inputs.shape[0])
        idxs_c.append(c)
        idxs_s.append(s)
    covers = inputs[idxs_c]
    secrets = inputs[idxs_s]
    #
    # Load the models
    #
    suds_model = load_vae_suds()
    HnetD, RnetD = load_ddh_mnist()
    Hnet, Rnet = load_udh_mnist()
    # 
    # Run and save lsb
    # 
    make_folder(args.savedir)
    if args.lsb:
        print(f"Demoing LSB, imgs saved to: {args.savedir}/lsb_demo/")
        save_path = f"{args.savedir}/lsb_demo/"
        C_folder = "C"
        Cprime_folder = "C_prime"
        Chat_folder = "C_hat"
        Cres_folder = "C_res"
        S_folder = "S"
        Shat_ddh_folder = "S_hat_ddh"
        Shat_udh_folder = "S_hat_udh"
        Shat_lsb_folder = "S_hat_lsb"
        Sprime_folder = "S_prime"
        Sres_folder = "S_res"
        #
        # make save directories
        #
        make_folder(save_path)
        make_folder(save_path+C_folder)
        make_folder(save_path+Cprime_folder)
        make_folder(save_path+Chat_folder)
        make_folder(save_path+Cres_folder)
        make_folder(save_path+S_folder)
        make_folder(save_path+Shat_ddh_folder)
        make_folder(save_path+Shat_udh_folder)
        make_folder(save_path+Shat_lsb_folder)
        make_folder(save_path+Sprime_folder)
        make_folder(save_path+Sres_folder)
        #
        # Generate steg 
        #
        containers, chat, _ , reveal_secret, lsb_sani_secret, S_res = use_lsb(covers, secrets, suds_model)
        with torch.no_grad():
            ddh_sani_secret = RnetD(chat)
            udh_sani_secret = Rnet(chat)
        C_res = chat - covers
        
        # 
        # Normalize and save
        # 
        C_res = norm(C_res)
        S_res = norm(S_res)
        save_plot(covers, save_path+C_folder)
        save_plot(secrets, save_path+S_folder)
        save_plot(containers, save_path+Cprime_folder)
        save_plot(chat, save_path+Chat_folder)
        save_plot(reveal_secret, save_path+Sprime_folder)
        save_plot(ddh_sani_secret, save_path+Shat_ddh_folder)
        save_plot(udh_sani_secret, save_path+Shat_udh_folder)
        save_plot(lsb_sani_secret, save_path+Shat_lsb_folder)
        save_plot(C_res, save_path+Cres_folder)
        save_plot(S_res, save_path+Sres_folder)
    # 
    # Run and save ddh
    # 
    if args.ddh:
        print(f"Demoing DDH, imgs saved to: {args.savedir}/ddh_demo/")
        save_path = f"{args.savedir}/ddh_demo/"
        C_folder = "C"
        Cprime_folder = "C_prime"
        Chat_folder = "C_hat"
        Cres_folder = "C_res"
        S_folder = "S"
        Shat_ddh_folder = "S_hat_ddh"
        Shat_udh_folder = "S_hat_udh"
        Shat_lsb_folder = "S_hat_lsb"
        Sprime_folder = "S_prime"
        Sres_folder = "S_res"
        #
        # make save directories
        #
        make_folder(save_path)
        make_folder(save_path+C_folder)
        make_folder(save_path+Cprime_folder)
        make_folder(save_path+Chat_folder)
        make_folder(save_path+Cres_folder)
        make_folder(save_path+S_folder)
        make_folder(save_path+Shat_ddh_folder)
        make_folder(save_path+Shat_udh_folder)
        make_folder(save_path+Shat_lsb_folder)
        make_folder(save_path+Sprime_folder)
        make_folder(save_path+Sres_folder)
        #
        # Generate steg 
        #
        containers, chat, _, reveal_secret, ddh_sani_secret, S_res = use_ddh(covers, secrets, HnetD, RnetD, suds_model)
        with torch.no_grad():
            lsb_sani_secret = decode_img(chat*255, train_mode=True)
            udh_sani_secret = Rnet(chat)
        C_res = chat - covers
        # 
        # Normalize and save
        # 
        C_res = norm(C_res)
        S_res = norm(S_res)
        save_plot(covers, save_path+C_folder)
        save_plot(secrets, save_path+S_folder)
        save_plot(containers, save_path+Cprime_folder)
        save_plot(chat, save_path+Chat_folder)
        save_plot(reveal_secret, save_path+Sprime_folder)
        save_plot(ddh_sani_secret, save_path+Shat_ddh_folder)
        save_plot(udh_sani_secret, save_path+Shat_udh_folder)
        save_plot(lsb_sani_secret/255, save_path+Shat_lsb_folder)
        save_plot(C_res, save_path+Cres_folder)
        save_plot(S_res, save_path+Sres_folder)
    # 
    # Run and save udh
    # 
    if args.udh:
        print(f"Demoing UDH, imgs saved to: {args.savedir}/udh_demo/")
        save_path = f"{args.savedir}/udh_demo/"
        C_folder = "C"
        Cprime_folder = "C_prime"
        Chat_folder = "C_hat"
        Cres_folder = "C_res"
        S_folder = "S"
        Shat_ddh_folder = "S_hat_ddh"
        Shat_udh_folder = "S_hat_udh"
        Shat_lsb_folder = "S_hat_lsb"
        Sprime_folder = "S_prime"
        Sres_folder = "S_res"
        #
        # make save directories
        #
        make_folder(save_path)
        make_folder(save_path+C_folder)
        make_folder(save_path+Cprime_folder)
        make_folder(save_path+Chat_folder)
        make_folder(save_path+Cres_folder)
        make_folder(save_path+S_folder)
        make_folder(save_path+Shat_ddh_folder)
        make_folder(save_path+Shat_udh_folder)
        make_folder(save_path+Shat_lsb_folder)
        make_folder(save_path+Sprime_folder)
        make_folder(save_path+Sres_folder)
        #
        # Generate steg 
        #
        containers, chat, _, reveal_secret, udh_sani_secret, S_res = use_udh(covers, secrets, Hnet, Rnet, suds_model)
        with torch.no_grad():
            lsb_sani_secret = decode_img(chat*255, train_mode=True)
            ddh_sani_secret = RnetD(chat)
        C_res = chat - covers
        # 
        # Normalize and save
        # 
        C_res = norm(C_res)
        S_res = norm(S_res)
        containers = norm(containers)
        save_plot(covers, save_path+C_folder)
        save_plot(secrets, save_path+S_folder)
        save_plot(containers, save_path+Cprime_folder)
        save_plot(chat, save_path+Chat_folder)
        save_plot(reveal_secret, save_path+Sprime_folder)
        save_plot(ddh_sani_secret, save_path+Shat_ddh_folder)
        save_plot(udh_sani_secret, save_path+Shat_udh_folder)
        save_plot(lsb_sani_secret/255, save_path+Shat_lsb_folder)
        save_plot(C_res, save_path+Cres_folder)
        save_plot(S_res, save_path+Sres_folder)
        
if __name__ == "__main__":
    main()