"""
Train an mnist classifier on poisoned data.
"""

from utils.utils import\
    load_udh_mnist,\
    load_ddh_mnist,\
    load_data

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
    # parser.add_argument("--model", type=int, default=None, help="The model to use during testing.")
    parser.add_argument("--savedir", type=str, default="models/data_poison")
    parser.add_argument("--expr_name", type=str, default="mnist_ddh_marked")
    parser.add_argument("--class_num", type=int, default=10, help="Number of classication classes.")
    parser.add_argument("--channels", type=int, default=1, help="Number of color channels in the training data. 1 if grayscale, 3 if color images.")
    parser.add_argument("--im_size", type=int, default=32, help="Size of the images. 28, 32, etc.")
    parser.add_argument("--k_num", type=int, default=128, help="The number of kernels to use in the CNN of the VAE.")
    parser.add_argument("--epochs", type=int, default=100, help="The number of training epochs used during training.")
    
    args = parser.parse_args()
    
    return args

def main(args):
    #
    # Load models
    #
    assert (args.hide in ["lsb", "ddh", "udh"]), "Not a valid hiding method. Try again."
    train_loader, test_loader = load_data("mnist")
    print(f"\nUsing {args.hide} to hide.\n")
    if args.hide == "ddh":
        HnetD, RnetD = load_ddh_mnist()
    elif args.hide == "udh":
        Hnet, Rnet = load_udh_mnist()
    #
    # Setup training models
    #
    if os.path.exists(args.savedir) == 0:
        os.mkdir(args.savedir)
    if os.path.exists(args.savedir+"/"+args.expr_name) == 0:
        os.mkdir(args.savedir+"/"+args.expr_name)
    trainer = Classifier(c_in=args.channels, k_num=args.k_num, class_num=args.class_num, im_size=args.im_size)
        
    loss_func = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(trainer.parameters(), lr=0.0001)
    #
    # Training
    #
    print("Training ...\n")
    for epoch in range(args.epochs):
        running_loss = 0.0
        running_n = 0
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            real_labels = copy.deepcopy(labels)
            real_inputs = copy.deepcopy(inputs)
            #
            # Create containers
            #
            prob = np.random.random(inputs.shape[0])
            for i, p in enumerate(prob):
                if p < 0.40:
                    # choose idx2 where label1 != label2
                    idx = np.random.randint(real_inputs.shape[0])
                    while real_labels[idx] == labels[i]:
                        idx = np.random.randint(real_inputs.shape[0])
                    # create container
                    inputs[i] = HnetD(torch.cat((inputs[i].unsqueeze(0), real_inputs[idx].unsqueeze(0)), dim=1))[0]
                    labels[i] = real_labels[idx]
            # train like normal
            y_pred = trainer.forward(inputs)
            loss = loss_func(y_pred, labels)
            optimizer.zero_grad()           
            loss.backward()          
            optimizer.step()   
            running_loss += loss
            running_n += 1
        
        print(f"Epoch [{epoch}]: loss = {running_loss/running_n:.4f}")
        
        torch.save(trainer.state_dict(), f'{args.savedir}/{args.expr_name}/{epoch}.pth')
    
    # Save the trained model
    torch.save(trainer.state_dict(), f'{args.savedir}/{args.expr_name}/model.pth')
    # Save the optimization states. This is helpful for continual training.
    torch.save(trainer.state_dict(), f'{args.savedir}/{args.expr_name}/optimizer.pth')
    
    print('Finished Training')
    
if __name__ == "__main__":
    args = get_args()
    main(args)
    