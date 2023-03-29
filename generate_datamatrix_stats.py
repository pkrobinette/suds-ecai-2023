"""
Run data matrix hiding to get suds performance evaluation.
"""

from utils.StegoPy import encode_img, decode_img, msg_to_map, map_to_msg
from utils.utils import load_data, load_vae_suds, get_embed_data, load_udh_mnist, load_ddh_mnist, use_ddh, use_udh

from tqdm import tqdm
import torch
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
from skimage.metrics import mean_squared_error as MSE
from torch.nn.functional import normalize

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
    parser.add_argument('--savedir', type=str, default="results/datamatrix_demo", help='The directory path to save stats.')
    parser.add_argument('-f', '--filename', type=str, default="datamatrix_stats.csv", help='The name of the file to save.')
    
    args = parser.parse_args()
    return args

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
    # load necessary models
    # 
    train_loader, test_loader = load_data("mnist")
    suds = load_vae_suds()
    sentences = np.array(get_embed_data())
    loss = torch.nn.MSELoss()
    #
    # Create save directory and file. 
    #
    if os.path.exists(args.savedir) == 0:
        os.mkdir(args.savedir)
    filename = args.filename if ".csv" in args.filename else args.filename + ".csv"
    path = args.savedir + "/"+filename
    result = {} # saved in this order (mse covers, mse bit, recoverable)
    
    if args.lsb:
        print("Running LSB ...")
        total_mse_covers = 0
        total_mse_bitmaps = 0
        total_recoverable = 0
        
        L = len(sentences)
        
        for i, data in enumerate(tqdm(test_loader), 0):
            inputs, labels = data
            d = inputs.shape[0]
            #
            # create bitmaps
            #
            sent_idx = random.sample(range(0, L), inputs.shape[0])
            bin_maps = msg_to_map(sentences[sent_idx], train_mode=True)
            containers = encode_img(inputs*255, bin_maps*255, train_mode=True)
            reveal = decode_img(containers, train_mode=True)
            with torch.no_grad():
                sani, _, _ = suds.forward_train(containers/255) # must be 0 to 1 range
            reveal_sani = decode_img(sani, train_mode=True) # decode works with either
            #
            # see if secrets are recoverable
            #
            decode = [map_to_msg(img) for img in reveal_sani]
            res = [e == d for e, d in zip(sentences[sent_idx], decode)]
            total_recoverable += np.mean(res)
            #
            # MSE STATS
            # get mse of covers (in [0, 255])
            sani = sani.view(d, -1)*255              # in 255 range
            containers = containers.view(d, -1)      # in 255 range
            bin_maps = bin_maps.view(d, -1)*255       # in 255 range
            reveal_sani = reveal_sani.view(d, -1)     # in 255 range
            total_mse_covers += loss(containers, sani)
            # # get mse of bit_maps (in [0, 255])
            total_mse_bitmaps += loss(bin_maps, reveal_sani)

        
        mean_mse_covers = round((total_mse_covers.item() / len(test_loader))/255, 2)
        mean_mse_bitmaps = round((total_mse_bitmaps.item() / len(test_loader))/255, 2)
        mean_recoverable = round((total_recoverable.item() / len(test_loader))/255, 2)
        print("MSE Covers: ", mean_mse_covers)
        print("MSE Bitmaps: ", mean_mse_bitmaps)
        print("% Recoverable: ", mean_recoverable)
        result["lsb"] = [mean_mse_covers, mean_mse_bitmaps, mean_recoverable]
        
    if args.ddh:
        HnetD, RnetD = load_ddh_mnist()
        print("Running DDH ...")
        total_mse_covers = 0
        total_mse_bitmaps = 0
        total_recoverable = 0
        
        L = len(sentences)
        
        for i, data in enumerate(tqdm(test_loader), 0):
            inputs, labels = data
            d = inputs.shape[0]
            #
            # create bitmaps
            #
            sent_idx = random.sample(range(0, L), inputs.shape[0])
            bin_maps = msg_to_map(sentences[sent_idx], train_mode=True)
            containers, sani, _, reveal_secrets, reveal_sani_secrets, _ = use_ddh(inputs, bin_maps, HnetD, RnetD, suds)
            #
            # see if secrets are recoverable
            #
            decode = [map_to_msg(img) for img in reveal_sani_secrets]
            res = [e == d for e, d in zip(sentences[sent_idx], decode)]
            total_recoverable += np.mean(res)
            #
            # MSE STATS
            # get mse of covers (in [0, 255])
            sani = sani.view(d, -1)*255              # in 255 range
            containers = containers.view(d, -1)*255       # in 255 range
            bin_maps = bin_maps.view(d, -1)*255       # in 255 range
            reveal_sani_secrets = reveal_sani_secrets.view(d, -1)*255     # in 255 range
            total_mse_covers += loss(containers, sani)
            # # get mse of bit_maps (in [0, 255])
            total_mse_bitmaps += loss(bin_maps, reveal_sani_secrets)
        
        mean_mse_covers = round((total_mse_covers.item() / len(test_loader))/255, 2)
        mean_mse_bitmaps = round((total_mse_bitmaps.item() / len(test_loader))/255, 2)
        mean_recoverable = round((total_recoverable.item() / len(test_loader))/255, 2)
        print("MSE Covers: ", mean_mse_covers)
        print("MSE Bitmaps: ", mean_mse_bitmaps)
        print("% Recoverable: ", mean_recoverable)
        result["lsb"] = [mean_mse_covers, mean_mse_bitmaps, mean_recoverable]
        
    if args.udh:
        Hnet, Rnet = load_udh_mnist()
        print("Running UDH ...")
        total_mse_covers = 0
        total_mse_bitmaps = 0
        total_recoverable = 0
        
        L = len(sentences)
        
        for i, data in enumerate(tqdm(test_loader), 0):
            inputs, labels = data
            d = inputs.shape[0]
            #
            # create bitmaps
            #
            sent_idx = random.sample(range(0, L), inputs.shape[0])
            bin_maps = msg_to_map(sentences[sent_idx], train_mode=True)
            containers, sani, _, reveal_secrets, reveal_sani_secrets, _ = use_udh(inputs, bin_maps, Hnet, Rnet, suds)
            #
            # see if secrets are recoverable
            #
            decode = [map_to_msg(img) for img in reveal_sani_secrets]
            res = [e == d for e, d in zip(sentences[sent_idx], decode)]
            total_recoverable += np.mean(res)
            #
            # MSE STATS
            # get mse of covers (in [0, 255])
            sani = sani.view(d, -1)*255               # in 255 range
            containers = containers.view(d, -1)*255      # in 255 range
            bin_maps = bin_maps.view(d, -1)*255       # in 255 range
            reveal_sani_secrets = reveal_sani_secrets.view(d, -1)*255     # in 255 range
            total_mse_covers += loss(containers, sani)
            # # get mse of bit_maps (in [0, 255])
            total_mse_bitmaps += loss(bin_maps, reveal_sani_secrets)
        
        mean_mse_covers = round((total_mse_covers.item() / len(test_loader))/255, 2)
        mean_mse_bitmaps = round((total_mse_bitmaps.item() / len(test_loader))/255, 2)
        mean_recoverable = round((total_recoverable.item() / len(test_loader))/255, 2)
        print("MSE Covers: ", mean_mse_covers)
        print("MSE Bitmaps: ", mean_mse_bitmaps)
        print("% Recoverable: ", mean_recoverable)
        result["lsb"] = [mean_mse_covers, mean_mse_bitmaps, mean_recoverable]
        
    # save all data:
    # with open(path, "w") as f:
    #     for key in result.keys():
    #         f.write(result[key] + "\n")
    df = pd.DataFrame(result) 
    
    # saving the dataframe 
    df.to_csv(path) 
            
    print(f"Data saved to: {path}")
    

if __name__ == "__main__":
    main()