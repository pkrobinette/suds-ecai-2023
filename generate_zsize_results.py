"""
Generate Latent Space stats used to create Figure 8 and Table 6.
"""

from utils.StegoPy import encode_img, decode_img, encode_msg, decode_msg
from utils.utils import load_udh_mnist, load_ddh_mnist, load_data, load_vae_suds, use_lsb, use_ddh, use_udh
from utils.vae import CNN_VAE

import tensorflow as tf
import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import json
import numpy as np
import random
import os
import pandas as pd
import argparse

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import mean_squared_error as MSE
from torch.nn.functional import normalize

np.random.seed(4)
random.seed(4)

sns.set()
mpl.rc('font',family='Verdana', size=12)


############################
#  Vars
###########################

MODELS = [2, 4, 8, 16, 32, 64, 128]


def get_args():
    """ 
    Get training arguments 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--hide", type=str, default="lsb", help="The hiding method to use during testing")
    parser.add_argument("--model", type=int, default=None, help="The model to use during testing.")
    parser.add_argument("--savedir", type=str, default="results/feature_size_img_stats")
    
    args = parser.parse_args()
    
    return args


def generate_stats(args):
    """
    Generate image quality metric stats for different feature sizes (z-sizes).
    
    Parameters
    ----------
    args : user defined arguments
    """
    #
    # check args
    #
    assert (args.hide in ["lsb", "udh", "ddh"]), "--hide is either lsb, udh, or ddh."
    if os.path.exists(args.savedir) == 0:
        os.mkdir(args.savedir)
    #
    # load models
    #
    if args.hide == "ddh":
        HnetD, RnetD = load_ddh()
    elif args.hide == "udh":
        Hnet, Rnet = load_udh()
        
    train_loader, test_loader = load_data("mnist")
    #
    # Generate image quality stats for each model (z-size)
    #
    for model in MODELS:
        vae_model = load_vae_suds(z_size=model)
        #
        # Initialize data save
        #
        mse = {
            "reconstruction": {}, # evaluates the reconstruction ability of SUDS
            "secret_before" : {}, # evalutes reveal functions before SUDS
            "secret_after": {}, # evaluates the reveal functions after SUDS
        }
        psnr = {
            "reconstruction": {}, # evaluates the reconstruction ability of SUDS
            "secret_before" : {}, # evalutes reveal functions before SUDS
            "secret_after": {}, # evaluates the reveal functions after SUDS
        }
        #
        # Clean Images
        # 
        print(f"\n--------- Starting Image Quality Tests: Model {model} ----------\n")
        print("1. Clean Images ...")
        mse_loss = nn.MSELoss()
        total_mse = 0
        total_psnr = 0
        cnt = 0
        for i, data in enumerate(tqdm(test_loader), 0):
            covers, labels = data
            d = covers.shape[0]
            with torch.no_grad():
                sani, _, _ = vae_model.forward_train(covers)
            sani = (normalize(sani.view(d, -1))*255).numpy()
            covers = (normalize(covers.view(d, -1))*255).numpy()
            
            total_mse += MSE(sani, covers)
            total_psnr += PSNR(covers, sani, data_range=255)
            cnt += 1 # average across batches
            
        mse["reconstruction"]["cover"] = (total_mse/cnt).item()
        psnr["reconstruction"]["cover"] = (total_psnr/cnt).item()
        print("mse: ", mse["reconstruction"], " psnr: ", psnr["reconstruction"])
        #
        # LSB Images
        #
        # master_idx = []
        total_mse = 0
        total_mse_sbefore = 0
        total_mse_post = 0
        total_psnr = 0
        total_psnr_sbefore = 0
        total_psnr_post = 0
        for i, data in enumerate(tqdm(test_loader), 0):
            covers, labels = data
            # create containers
            d = covers.shape[0]
            idx = torch.randperm(covers.shape[0])
            # master_idx.append(idx)
            secrets = covers[idx]
            if args.hide == "lsb":
                containers, sani, _, reveal_secrets, reveal_sani_secrets, _ = use_lsb(covers, secrets, vae_model)
            elif args.hide == "ddh":
                containers, sani, _, reveal_secrets, reveal_sani_secrets, _ = use_ddh(covers, secrets, HnetD, RnetD, vae_model)
            elif args.hide == "ddh":
                containers, sani, _, reveal_secrets, reveal_sani_secrets, _ = use_udh(covers, secrets, Hnet, Rnet, vae_model)

            containers = (normalize(containers.view(d, -1))*255).numpy()
            secrets = (normalize(secrets.view(d, -1))*255).numpy()
            reveal_secrets = (normalize(reveal_secrets.view(d, -1))*255).numpy()
            sani = (normalize(sani.view(d, -1))*255).numpy()
            reveal_sani_secrets = (normalize(reveal_sani_secrets.view(d, -1))*255).numpy()
            
            # encoded vs. decoded (reconstruction)
            total_mse += MSE(containers, sani)
            total_psnr += PSNR(containers, sani, data_range=255)
            # secret retrieval before sani
            total_mse_sbefore += MSE(secrets, reveal_secrets)
            total_psnr_sbefore += PSNR(secrets, reveal_secrets, data_range=255)
            # secret retrieval post sani
            total_mse_post += MSE(secrets, reveal_sani_secrets)
            total_psnr_post += PSNR(secrets, reveal_sani_secrets, data_range=255)
        
        psnr["reconstruction"][args.hide] = (total_psnr/cnt).item()
        psnr["secret_before"][args.hide] = (total_psnr_sbefore/cnt).item()
        psnr["secret_after"][args.hide] = (total_psnr_post/cnt).item()
        
        mse["reconstruction"][args.hide] = (total_mse/cnt).item()
        mse["secret_before"][args.hide] = (total_mse_sbefore/cnt).item()
        mse["secret_after"][args.hide] = (total_mse_post/cnt).item()
        
        print("MSE:")
        for k in mse.keys():
            print("mse ", k, ":", mse[k][args.hide])
        print("PSNR:")
        for k in psnr.keys():
            print("psnr ", k, ":", psnr[k][args.hide])
            
        print("------> Finished generating stats. Saving data ...")
        d = {"mse":mse, "psnr":psnr}
        result = pd.DataFrame.from_dict(d)
        path = f"{args.savedir}/suds_{model}.csv"
        result.to_csv(path)
        print(f"Results saved to: {path}\n\n")
        
            
def generate_plots(args):
    """
    Generate feature size performance comparison plot (Figure 8).
    
    Parameters
    ---------
    args : user defined arguments
    """
    #
    # Load data
    #
    print("\nLoading data ... \n")
    get_name = lambda a : "results/feature_size_img_stats/suds_"+str(a)+".csv"
    df2 = pd.read_csv(get_name(2))
    df4 = pd.read_csv(get_name(4))
    df8 = pd.read_csv(get_name(8))
    df16 = pd.read_csv(get_name(16))
    df32 = pd.read_csv(get_name(32))
    df64 = pd.read_csv(get_name(64))
    df128 = pd.read_csv(get_name(128))
    idx = {
        2: df2,
        4: df4,
        8: df8,
        16: df16,
        32: df32,
        64: df64,
        128: df128,
    }
    #
    # Create functions to easily pull data
    #
    get_mse_post = lambda df: json.loads(df["mse"][2].replace("'",  "\""))[args.hide]
    get_mse_pre = lambda df: json.loads(df["mse"][1].replace("'",  "\""))[args.hide]
    get_mse_recon = lambda df: json.loads(df["mse"][0].replace("'",  "\""))["cover"]
    get_psnr_pre = lambda df: json.loads(df["psnr"][1].replace("'",  "\""))[args.hide]
    get_psnr_post = lambda df: json.loads(df["psnr"][2].replace("'",  "\""))[args.hide]
    get_psnr_recon = lambda df: json.loads(df["psnr"][0].replace("'",  "\""))["cover"]
    get_df = lambda idx, i: idx[i]
    #
    # Get data
    #
    pre_mse = [get_mse_pre(get_df(idx, 2**i)) for i in range(1, 8)]
    post_mse = [get_mse_post(get_df(idx, 2**i)) for i in range(1, 8)]
    recon_mse = [get_mse_recon(get_df(idx, 2**i)) for i in range(1, 8)]
    pre_psnr = [get_psnr_pre(get_df(idx, 2**i)) for i in range(1, 8)]
    post_psnr = [get_psnr_post(get_df(idx, 2**i)) for i in range(1, 8)]
    recon_psnr = [get_psnr_recon(get_df(idx, 2**i)) for i in range(1, 8)]
    #
    # Create plots
    #
    csfont = {'fontname': 'Verdana', 'fontsize': "large"}
    plt.figure(1)
    plt.plot([2**i for i in range(1, 8)], recon_mse, ".-", color="orange", label="Input Alteration")
    plt.plot([2**i for i in range(1, 8)], pre_mse, ".-", color="blue", label="Pre-Sanitization Secret")
    plt.plot([2**i for i in range(1, 8)], post_mse, '.-', color="green", label="Post-Sanitization Secret")
    plt.legend(fontsize=12)
    plt.xlabel("Number of Features (n)" ,**csfont)
    plt.ylabel("MSE", **csfont)
    plt.ylim([-5, 75])
    plt.xlim([0, 130])
    plt.savefig(f"{args.savedir}/mse_zsize_results.pdf")
    # plt.show()
    plt.figure(2)
    plt.plot([2**i for i in range(1, 8)], recon_psnr, ".-", color="orange", label="Input Alteration")
    plt.plot([2**i for i in range(1, 8)], pre_psnr, ".-", color="blue", label="Pre-Sanitization Secret")
    plt.plot([2**i for i in range(1, 8)], post_psnr, '.-', color="green", label="Post-Sanitization Secret")
    plt.legend(fontsize=12)
    plt.xlabel("Number of Features (n)", **csfont)
    plt.ylabel("PSNR", **csfont)
    plt.xticks(fontsize=12)
    plt.ylim([-5, 75])
    plt.xlim([0, 130])
    plt.savefig(f"{args.savedir}/psnr_zsize_results.pdf")
    # plt.show()
    plt.figure(3)
    plt.plot([2**i for i in range(1, 8)], recon_mse, "^-", color="green", label="MSE C-hat - C'")
    plt.plot([2**i for i in range(1, 8)], pre_mse, "|-", color="green", label="MSE S'")
    plt.plot([2**i for i in range(1, 8)], post_mse, 'x-', color="green", label="MSE S-hat")
    plt.plot([2**i for i in range(1, 8)], recon_psnr, "^--", color="blue", label="PSNR C-hat - C'")
    plt.plot([2**i for i in range(1, 8)], pre_psnr, "|--", color="blue", label="PSNR S'")
    plt.plot([2**i for i in range(1, 8)], post_psnr, 'x--', color="blue", label="PSNR S-hat")
    plt.legend(fontsize=12, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.2))
    plt.xlabel("Number of Features (n)" ,**csfont)
    plt.ylabel("MSE/PSNR", **csfont)
    plt.ylim([-5, 75])
    plt.xlim([0, 130])
    plt.tight_layout()
    plt.savefig(f"{args.savedir}/zsize_results.pdf")
    # plt.show()
    print(f"\nAll plots saved to: {args.savedir}..\n")

def generate_table(args):
    """
    Generate a table of feature size image quality data.
    
    Parameters
    ----------
    args : user defined arguments
    """
    print("\nLoading data ... \n")
    get_name = lambda a : "results/feature_size_img_stats/suds_"+str(a)+".csv"
    df2 = pd.read_csv(get_name(2))
    df4 = pd.read_csv(get_name(4))
    df8 = pd.read_csv(get_name(8))
    df16 = pd.read_csv(get_name(16))
    df32 = pd.read_csv(get_name(32))
    df64 = pd.read_csv(get_name(64))
    df128 = pd.read_csv(get_name(128))
    idx = {
        2: df2,
        4: df4,
        8: df8,
        16: df16,
        32: df32,
        64: df64,
        128: df128,
    }
    #
    # Create functions to easily pull data
    #
    get_mse_post = lambda df: json.loads(df["mse"][2].replace("'",  "\""))[args.hide]
    get_mse_pre = lambda df: json.loads(df["mse"][1].replace("'",  "\""))[args.hide]
    get_mse_recon = lambda df: json.loads(df["mse"][0].replace("'",  "\""))[args.hide]
    get_psnr_pre = lambda df: json.loads(df["psnr"][1].replace("'",  "\""))[args.hide]
    get_psnr_post = lambda df: json.loads(df["psnr"][2].replace("'",  "\""))[args.hide]
    get_psnr_recon = lambda df: json.loads(df["psnr"][0].replace("'",  "\""))[args.hide]
    get_mse_recon_cover = lambda df: json.loads(df["mse"][0].replace("'",  "\""))["cover"]
    get_psnr_recon_cover = lambda df: json.loads(df["psnr"][0].replace("'",  "\""))["cover"]
    get_df = lambda idx, i: idx[i]
    #
    # Get data
    #
    pre_mse = [get_mse_pre(get_df(idx, 2**i)) for i in range(1, 8)]
    post_mse = [get_mse_post(get_df(idx, 2**i)) for i in range(1, 8)]
    recon_mse = [get_mse_recon(get_df(idx, 2**i)) for i in range(1, 8)]
    pre_psnr = [get_psnr_pre(get_df(idx, 2**i)) for i in range(1, 8)]
    post_psnr = [get_psnr_post(get_df(idx, 2**i)) for i in range(1, 8)]
    recon_psnr = [get_psnr_recon(get_df(idx, 2**i)) for i in range(1, 8)]
    recon_mse_cover = [get_mse_recon_cover(get_df(idx, 2**i)) for i in range(1, 8)]
    recon_psnr_cover = [get_psnr_recon_cover(get_df(idx, 2**i)) for i in range(1, 8)]
    #
    # Create a table with the data
    #
    print("Writing data ...\n")
    with open(f"{args.savedir}/zsize_results_table.txt", "w") as f:
        header = "Features(n) | | | Sanitizer Effect | Pre-Sanitization Secret | Post-Sanitization Secret\n"
        f.write(header)
        for i in reversed(range(len(MODELS))):
            model = MODELS[i]
            line1 = f" | Clean | MSE | {round(recon_mse_cover[i], 2)}| - | - \n"
            line2 = f"{model} |   | PSNR | {round(recon_psnr_cover[i], 2)}| - | - \n"
            line3 = f" | LSB | MSE | {round(recon_mse[i], 2)}| {round(pre_mse[i], 2)} | {round(post_mse[i], 2)}\n"
            line4 = f" |     | PSNR | {round(recon_psnr[i], 2)}| {round(pre_psnr[i], 2)} | {round(post_psnr[i],2)}\n"
            f.write(line1)
            f.write(line2)
            f.write(line3)
            f.write(line4)
            
    print(f"Table data saved to: {args.savedir}/zsize_results_table.txt\n")
        
        
    
if __name__ == "__main__":
    args = get_args()
    # generate_stats(args)
    generate_plots(args)
    # generate_table(args)