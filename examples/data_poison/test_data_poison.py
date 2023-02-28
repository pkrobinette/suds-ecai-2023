"""
Test an mnist classifier on poisoned data.
See if SUDS can protect the classifier from malicious
intent.
"""

from utils.utils import\
    load_udh_mnist,\
    load_ddh_mnist,\
    load_data,\
    load_classifier,\
    load_vae_suds

from utils.classifier import Classifier

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import random
import argparse
import torch
from tqdm import tqdm
import os
import copy

np.random.seed(4)
random.seed(4)

def get_args():
    """ 
    Get training arguments 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--hide", type=str, default="ddh", help="The hiding method to use during testing")
    parser.add_argument("--savedir", type=str, default="results/data_poison")
    parser.add_argument("--expr_name", type=str, default="mnist_ddh_marked")
    parser.add_argument("--suds", type=int, default=128, help="the zsize of suds to use.")
    
    args = parser.parse_args()
    
    return args

def main(args):
    """
    main.
    """
    #
    # Load the models and data
    #
    assert (args.hide in ["lsb", "ddh", "udh"]), "Invalid hide method. Try again."
    print("\nLoading Models ...\n")
    class_model=load_classifier()
    suds = load_vae_suds(z_size=args.suds)
    if args.hide == "ddh":
        HnetD, RnetD = load_ddh_mnist()
    elif args.hide == "udh":
        Hnet, Rnet = load_udh_mnist()
    train_loader, test_loader = load_data("mnist")
    #
    # test without SUDS
    #
    print("\nTesting without SUDS ...\n")
    loss_clean = 0
    loss_poison = 0
    poison_cnt = 0
    for i, data in enumerate(tqdm(test_loader), 0):
        inputs, labels = data
        real_labels = copy.deepcopy(labels)
        real_inputs = copy.deepcopy(inputs)
        poison = []
        clean = []
        prob = np.random.random(inputs.shape[0])
        for i, p in enumerate(prob):
            if p < 0.50:
                # update input, label
                poison.append(i)
                idx = np.random.randint(inputs.shape[0])
                while labels[i] == real_labels[idx]:
                    idx = np.random.randint(inputs.shape[0])
                inputs[i] = HnetD(torch.cat((inputs[i].unsqueeze(0), real_inputs[idx].unsqueeze(0)), dim=1))[0]
                labels[i] = real_labels[idx]
            else:
                clean.append(i)
                
        y_pred = class_model.get_predict(inputs)
        
        loss_clean += sum([y_pred[i] == labels[i] for i in clean])
        loss_poison += sum([y_pred[i] == labels[i] for i in poison])
        poison_cnt += len(poison)
        
    model_no_suds = round(loss_clean.item()/(10000-poison_cnt)*100, 2)
    poison_no_suds = round((loss_poison.item()/poison_cnt)*100, 2)
    print(f"Model Accuracy without SUDS: {model_no_suds}%")
    print(f"Poison Accuracy without SUDS: {poison_no_suds}%")
    #
    # Test with SUDS
    #
    print("\nTesting with SUDS ...\n")
    loss_clean = 0
    loss_poison = 0
    poison_cnt = 0
    for i, data in enumerate(tqdm(test_loader), 0):
        inputs, labels = data
        real_labels = copy.deepcopy(labels)
        real_inputs = copy.deepcopy(inputs)
        poison = []
        clean = []
        prob = np.random.random(inputs.shape[0])
        for i, p in enumerate(prob):
            if p < 0.50:
                # update input, label
                poison.append(i)
                idx = np.random.randint(inputs.shape[0])
                while labels[i] == real_labels[idx]:
                    idx = np.random.randint(inputs.shape[0])
                inputs[i] = HnetD(torch.cat((inputs[i].unsqueeze(0), real_inputs[idx].unsqueeze(0)), dim=1))[0]
                labels[i] = real_labels[idx]
            else:
                clean.append(i)
                
        cleaned, _, _ = suds.forward_train(inputs)
        y_pred = class_model.get_predict(cleaned)
        
        loss_clean += sum([y_pred[i] == labels[i] for i in clean])
        loss_poison += sum([y_pred[i] == labels[i] for i in poison])
        poison_cnt += len(poison)
        
    model_w_suds = round(loss_clean.item()/(10000-poison_cnt)*100, 2)
    poison_w_suds = round((loss_poison.item()/poison_cnt)*100, 2)
    print(f"Model Accuracy with SUDS: {model_w_suds}%")
    print(f"Poison Accuracy with SUDS: {poison_w_suds}%")
    #
    # Save the results
    #
    if os.path.exists(args.savedir) == 0:
        os.mkdir(args.savedir)
    with open(f"{args.savedir}/classification_results.txt", "w") as f:
        f.write("Image Type | no SUDS | SUDS\n")
        f.write("------------------------------\n")
        line1 = f"clean (5000) | {model_no_suds} | {model_w_suds}\n"
        line2 = f"containers (5000) | {poison_no_suds} | {poison_w_suds}\n"
        f.write(line1)
        f.write(line2)
    print(f"Results saved to: {args.savedir}/classification_results.txt")
    
if __name__ == "__main__":
    args = get_args()
    main(args)