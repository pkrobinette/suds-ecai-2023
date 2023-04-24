"""
Generate Figure 9: A closer look at the latent space which compares latent
variable mappings between covers and containers.

"""

from utils.utils import lsb_eval_latent_all,\
    ddh_eval_latent_all,\
    udh_eval_latent_all,\
    load_vae_suds,\
    load_udh_mnist,\
    load_ddh_mnist,\
    load_data

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import random
import argparse
import torch
from tqdm import tqdm
import os

np.random.seed(4)
random.seed(4)
sns.set()
font = {'family' : 'Verdana',
        'weight' : 'normal',
        'size'   : 30}

mpl.rc('font', **font)
csfont = {'family':'Verdana', 'fontsize':14}

def get_args():
    """ 
    Get training arguments 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", type=str, default="results/RQ4-plots")
    parser.add_argument("--model", type=int, default=8)
    
    args = parser.parse_args()
    
    return args

def get_mean_std(data):
    """
    helper function. Get mean and standard deviation.
    """
    return torch.mean(data, dim=0), torch.std(data, dim=0)

def main(args):
    #
    # Load models
    #
    print("\nLoading Models ...\n")
    Hnet, Rnet = load_udh_mnist()
    HnetD, RnetD = load_ddh_mnist()
    vae_model = load_vae_suds(z_size=args.model)
    train_loader, test_loader = load_data("mnist")
    vae_size = args.model
    #
    # Create a master list for containers
    #
    print("Mapping inputs to latent variables ...\n")
    master_index = []
    for i, data in enumerate(test_loader, 0):
        covers, labels = data
        idx = torch.randperm(covers.shape[0])
        master_index.append(idx)
    #
    # Generate z data
    #
    z_cover_lsb, z_container_lsb, master_l_lsb = lsb_eval_latent_all(test_loader, master_index, vae_model)
    z_cover_ddh, z_container_ddh, master_l_ddh = ddh_eval_latent_all(test_loader, master_index, HnetD, vae_model)
    z_cover_udh, z_container_udh, master_l_udh = udh_eval_latent_all(test_loader, master_index, Hnet, vae_model)
    #
    # Generate plot
    #
    print("Plotting ...\n")
    get_digit_idx = lambda d, master_l: [i for i in range(10000) if master_l[i].item() == d]
    cap_size = 3 # cap size of error bar
    l_size = 1 # line size of error bars
    
    fig, ax = plt.subplots(2, 2, figsize=(7,7))
    
    for i in range(2):
        for j in range(2):
            if i == 1 and j == 1:
                ax[i, j].errorbar(0, 0, 0, linestyle='None', label='cover', capsize=cap_size, elinewidth=l_size)
                ax[i, j].errorbar(0, 0, 0, linestyle='None', label='lsb container', capsize=cap_size, elinewidth=l_size)
                ax[i, j].errorbar(0, 0, 0, linestyle='None', label='ddh container', capsize=cap_size, elinewidth=l_size)
                ax[i, j].errorbar(0, 0, 0, linestyle='None', label='udh container', capsize=cap_size, elinewidth=l_size)
                ax[i, j].set_axis_off()
                ax[i, j].legend(loc='center')
            else:
                l = (i*2)+j
                index = get_digit_idx(l, master_l_lsb) # all master lists are the same
                mu_cover, e_cover = get_mean_std(z_cover_lsb[index])
                mu_container_lsb, e_container_lsb = get_mean_std(z_container_lsb[index])
                mu_container_ddh, e_container_ddh = get_mean_std(z_container_ddh[index])
                mu_container_udh, e_container_udh = get_mean_std(z_container_udh[index])
                
                ax[i, j].errorbar(np.linspace(1, vae_size, vae_size)-0.2, mu_cover, e_cover, linestyle='None', marker='.', label='cover', capsize=cap_size, elinewidth=l_size)
                ax[i, j].errorbar(np.linspace(1, vae_size, vae_size)-0.1, mu_container_lsb, e_container_lsb, linestyle='None', marker='.', label='lsb container', capsize=cap_size, elinewidth=l_size)
                ax[i, j].errorbar(np.linspace(1, vae_size, vae_size), mu_container_ddh, e_container_ddh, linestyle='None', marker='.', label='ddh container', capsize=cap_size, elinewidth=l_size)
                ax[i, j].errorbar(np.linspace(1, vae_size, vae_size)+0.1, mu_container_udh, e_container_udh, linestyle='None', marker='.', label='udh container', capsize=cap_size, elinewidth=l_size)
                ax[i, j].set_xticks([0,2,4,6, 8])
                ax[i, j].set_yticks([-2,0,2,4])
                ax[i, j].set_xlim([0, 9])
                ax[i, j].set_ylim([-3, 4])
                ax[i, j].set_title(f"Label = {l}", **csfont)
    #
    # Save
    #
    plt.tight_layout()
    print(f"Saving to: {args.savedir}/latent_mappings_plot_compact.pdf")
    if os.path.exists(args.savedir) == 0:
        os.mkdir(args.savedir)
    plt.savefig(f"{args.savedir}/latent_mappings_plot_compact.pdf")
    plt.show()
    #
    # Generate Complete Entire Image
    #
    fig = plt.figure(figsize=(12, 10))
    for i in range(10):
        if i == 9:
            ax = fig.add_subplot(3, 4, i+3)
            ax.errorbar(0, 0, 0, linestyle='None', label='cover', capsize=cap_size, elinewidth=l_size)
            ax.errorbar(0, 0, 0, linestyle='None', label='lsb container', capsize=cap_size, elinewidth=l_size)
            ax.errorbar(0, 0, 0, linestyle='None', label='ddh container', capsize=cap_size, elinewidth=l_size)
            ax.errorbar(0, 0, 0, linestyle='None', label='udh container', capsize=cap_size, elinewidth=l_size)
            ax.set_axis_off()
            ax.legend(loc='center')
            
        elif i < 9:
            if i == 8:
                ax = fig.add_subplot(3, 4, i+2)
            else:
                 ax = fig.add_subplot(3, 4, i+1)
            index = get_digit_idx(i, master_l_lsb) # all master lists are the same
            mu_cover, e_cover = get_mean_std(z_cover_lsb[index])
            mu_container_lsb, e_container_lsb = get_mean_std(z_container_lsb[index])
            mu_container_ddh, e_container_ddh = get_mean_std(z_container_ddh[index])
            mu_container_udh, e_container_udh = get_mean_std(z_container_udh[index])
        
            ax.errorbar(np.linspace(1, vae_size, vae_size)-0.2, mu_cover, e_cover, linestyle='None', marker='.', label='cover', capsize=cap_size, elinewidth=l_size)
            ax.errorbar(np.linspace(1, vae_size, vae_size)-0.1, mu_container_lsb, e_container_lsb, linestyle='None', marker='.', label='lsb container', capsize=cap_size, elinewidth=l_size)
            ax.errorbar(np.linspace(1, vae_size, vae_size), mu_container_ddh, e_container_ddh, linestyle='None', marker='.', label='ddh container', capsize=cap_size, elinewidth=l_size)
            ax.errorbar(np.linspace(1, vae_size, vae_size)+0.1, mu_container_udh, e_container_udh, linestyle='None', marker='.', label='udh container', capsize=cap_size, elinewidth=l_size)
    
            ax.set_xticks([0,2,4,6, 8])
            ax.set_yticks([-2,0,2,4])
            ax.set_xlim([0, 9])
            ax.set_ylim([-3, 4])
            ax.set_title(f"Label = {i}", **csfont)
    
    ax = fig.add_subplot(3, 4, 11)
    index = get_digit_idx(9, master_l_lsb) # all master lists are the same
    mu_cover, e_cover = get_mean_std(z_cover_lsb[index])
    mu_container_lsb, e_container_lsb = get_mean_std(z_container_lsb[index])
    mu_container_ddh, e_container_ddh = get_mean_std(z_container_ddh[index])
    mu_container_udh, e_container_udh = get_mean_std(z_container_udh[index])
    ax.errorbar(np.linspace(1, vae_size, vae_size)-0.2, mu_cover, e_cover, linestyle='None', marker='.', label='cover', capsize=cap_size, elinewidth=l_size)
    ax.errorbar(np.linspace(1, vae_size, vae_size)-0.1, mu_container_lsb, e_container_lsb, linestyle='None', marker='.', label='lsb container', capsize=cap_size, elinewidth=l_size)
    ax.errorbar(np.linspace(1, vae_size, vae_size), mu_container_ddh, e_container_ddh, linestyle='None', marker='.', label='ddh container', capsize=cap_size, elinewidth=l_size)
    ax.errorbar(np.linspace(1, vae_size, vae_size)+0.1, mu_container_udh, e_container_udh, linestyle='None', marker='.', label='udh container', capsize=cap_size, elinewidth=l_size)
    ax.set_xticks([0,2,4,6, 8])
    ax.set_yticks([-2,0,2,4])
    ax.set_xlim([0, 9])
    ax.set_ylim([-3, 4])
    ax.set_title(f"Label = {9}", **csfont)
    #
    # Save plot
    #
    print(f"Saving to: {args.savedir}/latent_mappings_plot.pdf")
    if os.path.exists(args.savedir) == 0:
        os.mkdir(args.savedir)
    plt.tight_layout()
    plt.savefig(f'{args.savedir}/latent_mappings_plot.pdf', bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    args = get_args()
    main(args)
