"""
Generate image examples of sanitizing steganography using NOISE on LSB, DDH, and/or UDH.

"""

from utils.StegoPy import encode_img, decode_img, encode_msg, decode_msg
from utils.utils import\
    load_udh_mnist,\
    load_ddh_mnist,\
    load_data,\
    load_vae_suds,\
    add_gauss,\
    add_saltnpep,\
    add_speckle

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
import pandas as pd

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
    parser.add_argument('--savedir', type=str, default="results/noise_demo", help='The directory path to save demo imgs.')
    
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
        plt.savefig(f"{folder}/{i}.jpg", bbox_inches='tight');
        
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
    """
    main function.
    """
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
    idx_s = torch.randperm(inputs.shape[0])
    # make sure that cover label != secret label
    covers = inputs[:5]
    secrets = inputs[idx_s[:5]]
    #
    # Load the models
    #
    # suds_model = load_vae_suds()
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
        S_folder = "S"
        salt_folder = "salt_n_pepper"
        speckle_folder = "speckle"
        gauss_folder = "gauss"
        R_salt_folder = "R_salt_n_pepper"
        R_speckle_folder = "R_speckle"
        R_gauss_folder = "R_gauss"
        #
        # make save directories
        #
        make_folder(save_path)
        make_folder(save_path+C_folder)
        make_folder(save_path+Cprime_folder)
        make_folder(save_path+S_folder)
        make_folder(save_path+salt_folder)
        make_folder(save_path+speckle_folder)
        make_folder(save_path+gauss_folder)
        make_folder(save_path+R_salt_folder)
        make_folder(save_path+R_speckle_folder)
        make_folder(save_path+R_gauss_folder)
        #
        # Generate steg 
        #
        # create containers
        containers = encode_img(covers*255, secrets*255, train_mode=True)
        # add a little guassian noise
        noisy_imgs_salt = add_saltnpep(containers/255) # between 1 and 0
        noisy_imgs_gauss = add_gauss(containers)
        noisy_imgs_speckle = add_speckle(containers/255)
        # decode
        reveal_secrets_salt = decode_img(noisy_imgs_salt*255, train_mode=True)
        reveal_secrets_gauss = decode_img(noisy_imgs_gauss*255, train_mode=True)
        reveal_secrets_speckle = decode_img(noisy_imgs_speckle*255, train_mode=True)
        # 
        # Normalize and save
        # 
        save_images(covers, save_path+C_folder)
        save_images(secrets, save_path+S_folder)
        save_images(containers, save_path+Cprime_folder)
        save_images(reveal_secrets_salt, save_path+R_salt_folder)
        save_images(reveal_secrets_gauss, save_path+R_gauss_folder)
        save_images(reveal_secrets_speckle, save_path+R_speckle_folder)
        save_images(noisy_imgs_salt, save_path+salt_folder)
        save_images(noisy_imgs_gauss, save_path+gauss_folder)
        save_images(noisy_imgs_speckle, save_path+speckle_folder)
        #
        # Combine Results
        #
        fig, ax = plt.subplots(5, 4)

        for i in range(5):
            ax[i, 0].imshow(secrets[i].permute(1, 2, 0), cmap='gray')
            ax[i, 1].imshow(reveal_secrets_gauss[i].permute(1, 2, 0), cmap='gray')
            ax[i, 2].imshow(reveal_secrets_speckle[i].permute(1, 2, 0), cmap='gray')
            ax[i, 3].imshow(reveal_secrets_salt[i].permute(1, 2, 0), cmap='gray')
        plt.axis("off")
        cols = ["S", "Gauss", "Speck", "Salt"]

        for a, col in zip(ax[0], cols):
            a.set_title(col)
        plt.savefig(save_path+"/overview.pdf")
        
    #
    # Run and save ddh
    # 
    if args.ddh:
        print(f"Demoing DDH, imgs saved to: {args.savedir}/ddh_demo/")
        save_path = f"{args.savedir}/ddh_demo/"
        C_folder = "C"
        Cprime_folder = "C_prime"
        S_folder = "S"
        salt_folder = "salt_n_pepper"
        speckle_folder = "speckle"
        gauss_folder = "gauss"
        R_salt_folder = "R_salt_n_pepper"
        R_speckle_folder = "R_speckle"
        R_gauss_folder = "R_gauss"
        #
        # make save directories
        #
        make_folder(save_path)
        make_folder(save_path+C_folder)
        make_folder(save_path+Cprime_folder)
        make_folder(save_path+S_folder)
        make_folder(save_path+salt_folder)
        make_folder(save_path+speckle_folder)
        make_folder(save_path+gauss_folder)
        make_folder(save_path+R_salt_folder)
        make_folder(save_path+R_speckle_folder)
        make_folder(save_path+R_gauss_folder)
        #
        # Generate steg 
        #
        # create containers
        H_input = torch.cat((covers, secrets), dim=1)
        # create containers
        with torch.no_grad():
            containers = HnetD(H_input)
        # add a little guassian noise
        noisy_imgs_salt = add_saltnpep(containers, pep=0.1) # between 1 and 0
        noisy_imgs_gauss = add_gauss(containers)
        noisy_imgs_speckle = add_speckle(containers)
        # decode
        # print(containers[0], noisy_imgs[0])
        with torch.no_grad():
            reveal_secrets_salt = RnetD(noisy_imgs_salt)
            reveal_secrets_gauss = RnetD(noisy_imgs_gauss)
            reveal_secrets_speckle = RnetD(noisy_imgs_speckle)
        # 
        # Normalize and save
        # 
        save_images(covers, save_path+C_folder)
        save_images(secrets, save_path+S_folder)
        save_images(containers, save_path+Cprime_folder)
        save_images(reveal_secrets_salt, save_path+R_salt_folder)
        save_images(reveal_secrets_gauss, save_path+R_gauss_folder)
        save_images(reveal_secrets_speckle, save_path+R_speckle_folder)
        save_images(noisy_imgs_salt, save_path+salt_folder)
        save_images(noisy_imgs_gauss, save_path+gauss_folder)
        save_images(noisy_imgs_speckle, save_path+speckle_folder)
        #
        # Combine Results
        #
        fig, ax = plt.subplots(5, 4)

        for i in range(5):
            ax[i, 0].imshow(secrets[i].permute(1, 2, 0), cmap='gray')
            ax[i, 1].imshow(reveal_secrets_gauss[i].permute(1, 2, 0), cmap='gray')
            ax[i, 2].imshow(reveal_secrets_speckle[i].permute(1, 2, 0), cmap='gray')
            ax[i, 3].imshow(reveal_secrets_salt[i].permute(1, 2, 0), cmap='gray')
        plt.axis("off")
        cols = ["S", "Gauss", "Speck", "Salt"]

        for a, col in zip(ax[0], cols):
            a.set_title(col)
        plt.savefig(save_path+"/overview.pdf")
    # 
    # Run and save udh
    # 
    if args.udh:
        print(f"Demoing UDH, imgs saved to: {args.savedir}/udh_demo/")
        save_path = f"{args.savedir}/udh_demo/"
        C_folder = "C"
        Cprime_folder = "C_prime"
        S_folder = "S"
        salt_folder = "salt_n_pepper"
        speckle_folder = "speckle"
        gauss_folder = "gauss"
        R_salt_folder = "R_salt_n_pepper"
        R_speckle_folder = "R_speckle"
        R_gauss_folder = "R_gauss"
        #
        # make save directories
        #
        make_folder(save_path)
        make_folder(save_path+C_folder)
        make_folder(save_path+Cprime_folder)
        make_folder(save_path+S_folder)
        make_folder(save_path+salt_folder)
        make_folder(save_path+speckle_folder)
        make_folder(save_path+gauss_folder)
        make_folder(save_path+R_salt_folder)
        make_folder(save_path+R_speckle_folder)
        make_folder(save_path+R_gauss_folder)
        #
        # Generate steg 
        #
        # create containers
        # create containers
        with torch.no_grad():
            containers = Hnet(secrets) + covers
        # add a little guassian noise
        noisy_imgs_salt = add_saltnpep(containers, 0.9) # between 1 and 0
        noisy_imgs_gauss = add_gauss(containers)
        noisy_imgs_speckle = add_speckle(containers)
        # decode
        # print(containers[0], noisy_imgs[0])
        with torch.no_grad():
            reveal_secrets_salt = Rnet(noisy_imgs_salt)
            reveal_secrets_gauss = Rnet(noisy_imgs_gauss)
            reveal_secrets_speckle = Rnet(noisy_imgs_speckle)
        # 
        # Normalize and save
        # 
        save_plot(covers, save_path+C_folder)
        save_plot(secrets, save_path+S_folder)
        save_plot(containers, save_path+Cprime_folder)
        save_plot(reveal_secrets_salt, save_path+R_salt_folder)
        save_plot(reveal_secrets_gauss, save_path+R_gauss_folder)
        save_plot(reveal_secrets_speckle, save_path+R_speckle_folder)
        save_plot(noisy_imgs_salt, save_path+salt_folder)
        save_plot(noisy_imgs_gauss, save_path+gauss_folder)
        save_plot(noisy_imgs_speckle, save_path+speckle_folder)
        #
        # Combine Results
        #
        fig, ax = plt.subplots(5, 4)

        for i in range(5):
            ax[i, 0].imshow(secrets[i].permute(1, 2, 0), cmap='gray')
            ax[i, 1].imshow(reveal_secrets_gauss[i].permute(1, 2, 0), cmap='gray')
            ax[i, 2].imshow(reveal_secrets_speckle[i].permute(1, 2, 0), cmap='gray')
            ax[i, 3].imshow(reveal_secrets_salt[i].permute(1, 2, 0), cmap='gray')
        plt.axis("off")
        cols = ["S", "Gauss", "Speck", "Salt"]

        for a, col in zip(ax[0], cols):
            a.set_title(col)
        plt.savefig(save_path+"/overview.pdf")
        
if __name__ == "__main__":
    main()