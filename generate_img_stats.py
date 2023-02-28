"""
Calculate image quality statistics for different
types of sanitization (suds vs. noise).
"""
from utils.utils import lsb_eval_latent_all,\
    ddh_eval_latent_all,\
    udh_eval_latent_all,\
    load_vae_suds,\
    load_udh_mnist,\
    load_ddh_mnist,\
    load_data,\
    use_lsb,\
    use_ddh,\
    use_udh,\
    add_gauss,\
    add_speckle,\
    add_saltnpep

from utils.StegoPy import encode_img, decode_img

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import torch.nn as nn
import numpy as np
import random
import argparse
import torch
from tqdm import tqdm
import os
import argparse
import pandas as pd
import json

np.random.seed(4)
random.seed(4)

from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import mean_squared_error as MSE
from torch.nn.functional import normalize
from skimage.util import random_noise


np.random.seed(4)
random.seed(4)


def get_args():
    """
    Get training arguments
    """
    parser = argparse.ArgumentParser()
    # ----
    parser.add_argument("--savedir", type=str, default="results/noise_comparison", help="Meta directory used to save.")
    parser.add_argument("--save_name", type=str, default=None, help="save name")
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    # parser.add_argument("--noise", type=str, default='gauss', help="which noise to use.")
    parser.add_argument('--gauss', action='store_true', help='Enable gauss noise option')
    parser.add_argument('--speckle', action='store_true', help='Enable speckle noise option')
    parser.add_argument('--saltnpep', action='store_true', help='Enable saltnpep noise option')
    parser.add_argument('--suds', action='store_true', help='Enable suds sanitization option')
    
    args = parser.parse_args()
    return args


def calc_stats(test_loader, args, **kwargs):
    """
    Calculate stats for a particular type of noise.
    
    Parameters
    ----------
    test_loader : DataLoader
    args : user indicated arguments
    kwargs : dictionary
        extra information including models and noise functions
    
    Returns
    -------
    d : dictionary
        Dictionary of data
    """
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
    add_noise = kwargs.get("noise", None)
    Hnet = kwargs.get("Hnet", None)
    Rnet = kwargs.get("Rnet", None)
    HnetD = kwargs.get("HnetD", None)
    RnetD = kwargs.get("RnetD", None)
    
    print("\n--------- Starting Image Quality Tests ----------\n")
    print("1. Clean Images ...")
    mse_loss = nn.MSELoss()
    total_mse = 0
    total_psnr = 0
    cnt = 0
    for i, data in enumerate(tqdm(test_loader), 0):
        covers, labels = data
        covers = covers.clone().detach()
        d = covers.shape[0]
        if args.suds:
            with torch.no_grad():
                sani, _, _ = kwargs["suds"].forward_train(covers)
        else:
            sani = add_noise(covers)
            
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
    print("2. LSB Images ...")
    master_idx = []
    total_mse_lsb = 0
    total_mse_lsb_sbefore = 0
    total_mse_lsb_post = 0
    total_psnr_lsb = 0
    total_psnr_lsb_sbefore = 0
    total_psnr_lsb_post = 0
    for i, data in enumerate(tqdm(test_loader), 0):
        covers, labels = data
        covers = covers.clone().detach()
        # create containers
        d = covers.shape[0]
        idx = torch.randperm(covers.shape[0])
        master_idx.append(idx)
        secrets = covers[idx]
        if args.suds:
            containers, sani, _, reveal_secrets, reveal_sani_secrets, _ = use_lsb(covers, secrets, kwargs["suds"])
        else:
            containers = encode_img(covers*255, secrets*255, train_mode=True)
            reveal_secrets = decode_img(containers, train_mode=True)
            sani = add_noise(containers/255)
            reveal_sani_secrets = decode_img(sani*255, train_mode=True)
        
        
        # covers = normalize(covers.view(d, -1)).numpy()
        containers = (normalize(containers.view(d, -1))*255).numpy()
        secrets = (normalize(secrets.view(d, -1))*255).numpy()
        reveal_secrets = (normalize(reveal_secrets.view(d, -1))*255).numpy()
        sani = (normalize(sani.view(d, -1))*255).numpy()
        reveal_sani_secrets = (normalize(reveal_sani_secrets.view(d, -1))*255).numpy()
        
        # encoded vs. decoded (reconstruction)
        total_mse_lsb += MSE(containers, sani)
        total_psnr_lsb += PSNR(containers, sani, data_range=255)
        # secret retrieval before sani
        total_mse_lsb_sbefore += MSE(secrets, reveal_secrets)
        total_psnr_lsb_sbefore += PSNR(secrets, reveal_secrets, data_range=255)
        # secret retrieval post sani
        total_mse_lsb_post += MSE(secrets, reveal_sani_secrets)
        total_psnr_lsb_post += PSNR(secrets, reveal_sani_secrets, data_range=255)
    
    psnr["reconstruction"]["lsb"] = (total_psnr_lsb/cnt).item()
    psnr["secret_before"]["lsb"] = (total_psnr_lsb_sbefore/cnt).item()
    psnr["secret_after"]["lsb"] = (total_psnr_lsb_post/cnt).item()
    
    mse["reconstruction"]["lsb"] = (total_mse_lsb/cnt).item()
    mse["secret_before"]["lsb"] = (total_mse_lsb_sbefore/cnt).item()
    mse["secret_after"]["lsb"] = (total_mse_lsb_post/cnt).item()
    
    print("MSE:")
    for k in mse.keys():
        print("mse ", k, ":", mse[k]["lsb"])
    print("PSNR:")
    for k in psnr.keys():
        print("psnr ", k, ":", psnr[k]["lsb"])
    #
    # DDH Images
    #
    print("3. DDH Images ...")
    total_mse_ddh = 0
    total_mse_ddh_sbefore = 0
    total_mse_ddh_post = 0
    total_psnr_ddh = 0
    total_psnr_ddh_sbefore = 0
    total_psnr_ddh_post = 0
    for i, data in enumerate(tqdm(test_loader), 0):
        covers, labels = data
        covers = covers.clone().detach()
        # create containers
        d = covers.shape[0]
        secrets = covers[master_idx[i]]
        if args.suds:
            containers, sani, _, reveal_secrets, reveal_sani_secrets, _ = use_ddh(covers, secrets, HnetD, RnetD, kwargs["suds"])
        else:
            H_input = torch.cat((covers, secrets), dim=1)
            with torch.no_grad():
                containers = HnetD(H_input)
                reveal_secrets = RnetD(containers)
                sani = add_noise(containers)
                reveal_sani_secrets = RnetD(sani)
            
        # covers = normalize(covers.view(d, -1)).numpy()
        containers = (normalize(containers.view(d, -1))*255).numpy()
        secrets = (normalize(secrets.view(d, -1))*255).numpy()
        reveal_secrets = (normalize(reveal_secrets.view(d, -1))*255).numpy()
        sani = (normalize(sani.view(d, -1))*255).numpy()
        reveal_sani_secrets = (normalize(reveal_sani_secrets.view(d, -1))*255).numpy()  
        
        # reconstruction
        total_mse_ddh += MSE(containers, sani)
        total_psnr_ddh += PSNR(containers, sani, data_range=255)
        # secrets before
        total_mse_ddh_sbefore += MSE(secrets, reveal_secrets)
        total_psnr_ddh_sbefore += PSNR(secrets, reveal_secrets, data_range=255)
        # secrets after 
        total_mse_ddh_post += MSE(secrets, reveal_sani_secrets)
        total_psnr_ddh_post += PSNR(secrets, reveal_sani_secrets, data_range=255)
        
    mse["reconstruction"]["ddh"] = (total_mse_ddh/cnt).item()
    mse["secret_before"]["ddh"] = (total_mse_ddh_sbefore/cnt).item()
    mse["secret_after"]["ddh"] = (total_mse_ddh_post/cnt).item()
    
    psnr["reconstruction"]["ddh"] = (total_psnr_ddh/cnt).item()
    psnr["secret_before"]["ddh"] = (total_psnr_ddh_sbefore/cnt).item()
    psnr["secret_after"]["ddh"] = (total_psnr_ddh_post/cnt).item()
    
    print("MSE:")
    for k in mse.keys():
        print("mse ", k, ":", mse[k]["ddh"])
    print("PSNR:")
    for k in psnr.keys():
        print("psnr ", k, ":", psnr[k]["ddh"])
    #
    # UDH Images
    #
    print("4. UDH Images ...")
    total_mse_udh = 0
    total_mse_udh_sbefore = 0
    total_mse_udh_post = 0
    total_psnr_udh = 0
    total_psnr_udh_sbefore = 0
    total_psnr_udh_post = 0
    for i, data in enumerate(tqdm(test_loader), 0):
        covers, labels = data
        covers = covers.clone().detach()
        # create containers
        d = covers.shape[0]
        secrets = covers[master_idx[i]]
        if args.suds:
            containers, sani, _, reveal_secrets, reveal_sani_secrets, _ = use_udh(covers, secrets, Hnet, Rnet, kwargs["suds"])
        else:
            with torch.no_grad():
                containers = Hnet(secrets) + covers
                reveal_secrets = Rnet(containers)
                sani = add_noise(containers)
                reveal_sani_secrets = Rnet(sani)
            
        # covers = normalize(covers.view(d, -1)).numpy()
        containers = (normalize(containers.view(d, -1))*255).numpy()
        secrets = (normalize(secrets.view(d, -1))*255).numpy()
        reveal_secrets = (normalize(reveal_secrets.view(d, -1))*255).numpy()
        sani = (normalize(sani.view(d, -1))*255).numpy()
        reveal_sani_secrets = (normalize(reveal_sani_secrets.view(d, -1))*255).numpy() 
        
        # reconstruction
        total_mse_udh += MSE(containers, sani)
        total_psnr_udh += PSNR(containers, sani, data_range=255)
        # secret before
        total_mse_udh_sbefore += MSE(secrets, reveal_secrets)
        total_psnr_udh_sbefore += PSNR(secrets, reveal_secrets, data_range=255)
        # secret after
        total_mse_udh_post += MSE(secrets, reveal_sani_secrets)
        total_psnr_udh_post += PSNR(secrets, reveal_sani_secrets, data_range=255)
        
    mse["reconstruction"]["udh"] = (total_mse_udh/cnt).item()
    mse["secret_before"]["udh"] = (total_mse_udh_sbefore/cnt).item()
    mse["secret_after"]["udh"] = (total_mse_udh_post/cnt).item()
    
    psnr["reconstruction"]["udh"] = (total_psnr_udh/cnt).item()
    psnr["secret_before"]["udh"] = (total_psnr_udh_sbefore/cnt).item()
    psnr["secret_after"]["udh"] = (total_psnr_udh_post/cnt).item()
    print("MSE:")
    for k in mse.keys():
        print("mse ", k, ":", mse[k]["udh"])
    print("PSNR:")
    for k in psnr.keys():
        print("psnr ", k, ":", psnr[k]["udh"])
    
    print(f"------> Finished ...")
    d = {"mse":mse, "psnr":psnr}
    result = pd.DataFrame.from_dict(d)
    try:
        path = args.savedir +'/'+ args.save_name +'.csv'
    except:
        path = args.savedir + "/results.csv"
    result.to_csv(path)
    print(f"Results saved to: {path}")
    
def generate_table(args):
    """
    Generate a table of all results.
    """
    print("\nLoading data ... \n")
    #
    # Load data
    #
    get_name = lambda a : "results/noise_comparison/"+ str(a) + "_im_stats.csv"
    df1 = pd.read_csv(get_name("suds"))
    df2 = pd.read_csv(get_name("gauss"))
    df3 = pd.read_csv(get_name("speckle"))
    df4 = pd.read_csv(get_name("saltnpep"))

    #
    # Create functions to easily pull data
    #
    get_mse_post = lambda df, hide: json.loads(df["mse"][2].replace("'",  "\""))[hide]
    get_mse_pre = lambda df, hide: json.loads(df["mse"][1].replace("'",  "\""))[hide]
    get_mse_recon = lambda df, hide: json.loads(df["mse"][0].replace("'",  "\""))[hide]
    get_psnr_pre = lambda df, hide: json.loads(df["psnr"][1].replace("'",  "\""))[hide]
    get_psnr_post = lambda df, hide: json.loads(df["psnr"][2].replace("'",  "\""))[hide]
    get_psnr_recon = lambda df, hide: json.loads(df["psnr"][0].replace("'",  "\""))[hide]
    get_mse_recon_cover = lambda df: json.loads(df["mse"][0].replace("'",  "\""))["cover"]
    get_psnr_recon_cover = lambda df: json.loads(df["psnr"][0].replace("'",  "\""))["cover"]
    #
    # Write data
    #
    dfs = {
        "suds": df1,
        "gauss": df2,
        "speckle": df3,
        "saltnpep": df4,
    }
    
    hide = ["cover", "lsb", "ddh", "udh"]
    print("Writing data ...\n")
    with open(f"{args.savedir}/all_img_stats.txt", "w") as f:
        header = "Sanitizer |  | | Sanitizer Effect | Pre-Sanitization Secret | Post-Sanitization Secret\n"
        f.write(header)
        f.write("-----------------------------------------\n")
        for key in dfs.keys():
            df = dfs[key]
            for h in hide:
                if h == "cover":
                    line1 =  f"        | Clean | MSE  | {round(get_mse_recon_cover(df), 2)}| - | - \n"
                    line2 =  f"        |       | PSNR | {round(get_psnr_recon_cover(df), 2)}| - | - \n"
                elif h == "ddh":
                    line1 =  f" {key}  | DDH | MSE | {round(get_mse_recon(df, h), 2)}| {round(get_mse_pre(df, h), 2)} | {round(get_mse_post(df, h), 2)}\n"
                    line2 =  f"        |     | PSNR | {round(get_psnr_recon(df, h), 2)}| {round(get_psnr_pre(df,h), 2)} | {round(get_psnr_post(df, h), 2)}\n"
                else:
                    line1 =  f"        | DDH | MSE | {round(get_mse_recon(df, h), 2)}| {round(get_mse_pre(df, h), 2)} | {round(get_mse_post(df, h), 2)}\n"
                    line2 =  f"        |     | PSNR | {round(get_psnr_recon(df, h), 2)}| {round(get_psnr_pre(df, h), 2)} | {round(get_psnr_post(df, h), 2)}\n"
                f.write(line1)
                f.write(line2)
                if h == "udh":
                    f.write("-----------------------------------------\n")
    print(f"Saved table to: {args.savedir}/all_img_stats.txt")
    

def main(args):
    """
    main.
    """
    args = get_args()
    if not args.gauss and not args.speckle and not args.saltnpep and not args.suds:
        print("Running everything")
        args.gauss = True
        args.speckle = True
        args.saltnpep = True
    #
    # load models
    #
    HnetD, RnetD = load_ddh_mnist()
    Hnet, Rnet = load_udh_mnist()
    suds = load_vae_suds()
    train_loader, test_loader = load_data("mnist")
    if os.path.exists(args.savedir) == 0:
        os.mkdir(args.savedir)
    #
    # Run for each type of noise
    #
    noise_catalog = {
        "gauss": add_gauss,
        "speckle": add_speckle,
        "saltnpep": add_saltnpep,
    }
    kwargs = {
        "Hnet": Hnet,
        "HnetD": HnetD,
        "Rnet" : Rnet,
        "RnetD": RnetD,
        "suds": suds,
    }
    if args.gauss:
        kwargs["noise"] = noise_catalog["gauss"]
        args.save_name = "gauss_im_stats"
        calc_stats(test_loader, args, **kwargs)
    if args.speckle:
        kwargs["noise"] = noise_catalog["speckle"]
        args.save_name = "speckle_im_stats"
        calc_stats(test_loader, args, **kwargs)
    if args.saltnpep:
        kwargs["noise"] = noise_catalog["saltnpep"]
        args.save_name = "saltnpep_im_stats"
        calc_stats(test_loader, args, **kwargs)
    # key in suds
    args.suds = True
    if args.suds:
        # kwargs["noise"] = noise_catalog["gauss"]
        args.save_name = "suds_im_stats"
        calc_stats(test_loader, args, **kwargs)


if __name__ == "__main__":
    args = get_args()
    main(args)
    generate_table(args)

    
    