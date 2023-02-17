import argparse

import tensorflow as tf
import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.vae import CNN_VAE

from utils.utils import load_data

import matplotlib.pyplot as plt
import numpy as np
import random
import os
import sys

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Needed if training with stego images as well
from utils.StegoPy import encode_img, decode_img, encode_msg, decode_msg
from utils.HidingUNet import UnetGenerator
from utils.RevealNet import RevealNet
from dhide_main import weights_init

np.random.seed(4)
random.seed(4)

if torch.cuda.is_available(): 
    dev = "cuda:0" 
else: 
    dev = "cpu" 
    
DEVICE = torch.device(dev)

def get_args():
    """ Get training arguments """
    parser = argparse.ArgumentParser()
    # --- Training parameters
    parser.add_argument("--z_dim", type=int, default=128, help="Size of the latent variable z used during training.")
    parser.add_argument("--channels", type=int, default=1, help="Number of color channels in the training data. 1 if grayscale, 3 if color images.")
    parser.add_argument("--batch_size", type=int, default=128, help="Size of training batches.")
    parser.add_argument("--im_size", type=int, default=32, help="Size of the images. 28, 32, etc.")
    parser.add_argument("--k_num", type=int, default=128, help="The number of kernels to use in the CNN of the VAE.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to use during training.")
    parser.add_argument("-d", "--dataset", type=str, default="mnist", help="The dataset to train on.")
    
    # ----
    parser.add_argument("--savedir", type=str, default="models/sanitization/", help="Meta directory used to save different experiments during training.")
    parser.add_argument("--expr_name", type=str, default=None, help="Experiment name")
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--log', dest='log', action='store_true')
    feature_parser.add_argument('--no-log', dest='log', action='store_false')
    parser.set_defaults(log=True)
    parser.add_argument("--start_from", type=str, default=None, help="indicate which directory to start from if continuing training.")
    parser.add_argument("--model", type=str, default="model.pth", help="If last training did not complete, where to start from.")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    #
    # Assert save directory is correct
    #
    assert (args.expr_name != None),"Please indicate an experiment name. This will be used to save the trained model."
    # check if path exists if starting from older training
    if args.start_from != None:
        old_path = args.savedir + "/" + args.start_from + "/"
        assert (os.path.exists(old_path+args.model)), "Old path does not exist, try again."
        
    path = args.savedir + "/" + args.expr_name
    if os.path.exists(path):
        cont = input(f'\n >> {args.expr_name} already exists in the save directory. Do you wish to continue and overwrite this directory? (Y/N)\n')
        if cont.lower() == "n":
            sys.exit("Exiting ...")
    else:
        os.mkdir(path)
    print(f"\nPath successfully created: {path}")
    print(f"Logging = {args.log}")
    #
    # Load training data
    # 
    print(f"Loading {args.dataset.upper()} dataset ...")
    train_loader, test_loader = load_data(args.dataset, args.batch_size)
    #
    # Set up models
    # 
    vae_instance = CNN_VAE(c_in=args.channels, k_num=args.k_num, z_size=args.z_dim, im_size=args.im_size)
    if args.start_from != None:
        print("Loading previous training session ...")
        vae_instance.load_state_dict(torch.load(old_path+args.model));
    vae_instance = vae_instance.to(DEVICE)
    #
    # Set up training functions
    #
    mse_loss_func = torch.nn.MSELoss()
    # Define the optimizer.
    optimizer = torch.optim.Adam(vae_instance.parameters(), lr=0.0001)
    # check if continuing training from a previous save. Optimizer might not be saved.
    if args.start_from != None:
        try:
            optimizer.load_state_dict(torch.load(old_path+"optimizer.pth"));
        except:
            print("Optimizer not found ...")
    #
    # Create summary writers if logging
    #
    if args.log:
        writer_cover = SummaryWriter(f"logs/{args.expr_name}/cover")
        writer_sani = SummaryWriter(f"logs/{args.expr_name}/sanitized")
        writer_data = SummaryWriter(f"logs/{args.expr_name}/training_data")
    #
    # Start training
    #
    print(f"Starting training with device={DEVICE}")
    print("--------------------------------------\n")
    for epoch in range(args.epochs):
        #
        # Initialize performance variables
        #
        running_loss = 0.0
        running_kl_loss = 0.0
        running_mse_loss = 0.0
        running_n = 0
        #
        # the training loop
        #
        for i, data in enumerate(train_loader, 0):
            #
            # clone covers
            #
            covers, labels = data
            covers = covers.clone().detach().to(DEVICE)
            #
            # zero the parameter gradients
            #
            optimizer.zero_grad()
            #
            # forward + backward + optimize
            #
            x_hat, mean, logvar  = vae_instance.forward_train(covers)
            x_hat = x_hat.to(DEVICE)
            mse_loss = mse_loss_func(x_hat*255, covers*255).to(DEVICE)
            kl_div_loss = ((mean**2 + logvar.exp() - 1 - logvar) / 2).mean().to(DEVICE)
            loss = mse_loss + kl_div_loss
            loss.backward()
            optimizer.step()
            #
            # print statistics
            #
            running_loss += loss.item()
            running_kl_loss += kl_div_loss.item()
            running_mse_loss += mse_loss.item()
            running_n += covers.shape[0]
            
            if i+1 % 25 == 0:
                print(f"{epoch}/{i}")
    
        print(
            f'[{epoch + 1}] loss: {running_loss / running_n:.6f}',
            f'KL loss: {running_kl_loss / running_n:.6f}',
            f'MSE loss: {running_mse_loss / running_n:.6f}')
        #
        # only log if indicated
        #
        if args.log:
            writer_data.add_scalar("loss", round(running_loss/running_n, 6), global_step=epoch)
            writer_data.add_scalar("mse loss", round(running_mse_loss/running_n, 6), global_step=epoch)
            writer_data.add_scalar("kl loss", round(running_kl_loss/running_n, 6), global_step=epoch)
        
            with torch.no_grad():
                x_hat = x_hat.reshape(-1, 1, 32, 32).clone().detach()
                sani_grid = torchvision.utils.make_grid(x_hat, normalize=True)
                cover_grid = torchvision.utils.make_grid(covers, normalize=True)
                writer_sani.add_image(
                    "Decoded Images", sani_grid, global_step=epoch
                )
                writer_cover.add_image(
                    "Encoded Images", cover_grid, global_step=epoch
                )
        #
        # Reset stats
        #
        running_loss = 0.0
        running_kl_loss = 0.0
        running_mse_loss = 0.0
        running_n = 0
        
        torch.save(vae_instance.state_dict(), f'{path}/{epoch}.pth')
    #
    # Save the trained model
    #
    torch.save(vae_instance.state_dict(), f'{path}/model.pth')
    # Save the optimization states. This is helpful for continual training.
    torch.save(optimizer.state_dict(), f'{path}/optimizer.pth')

    print('Finished Training')
    
    