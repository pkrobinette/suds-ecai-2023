from art.attacks.evasion import FastGradientMethod, CarliniL2Method, DeepFool, BasicIterativeMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from utils.utils import load_data, load_vae_suds, load_classifier

import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import argparse
import os
from tqdm import tqdm
import torch

np.random.seed(4)

def get_args():
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(description='Argument parser for ddh, udh, and lsb')
    
    parser.add_argument('--savedir', type=str, default="results/evasion_demo", help='The directory path to save stats.')
    parser.add_argument('-f', '--filename', type=str, default="evasion_stats.csv", help='The name of the file to save.')
    
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
    if os.path.exists(args.savedir) == 0:
        os.mkdir(args.savedir)
    filename = args.filename if ".csv" in args.filename else args.filename + ".csv"
    path = args.savedir + "/"+filename
    # 
    # load necessary models
    # 
    train_loader, test_loader = load_data("mnist", batch_size=2000)
    suds = load_vae_suds()
    class_model = load_classifier()
    classifier = PyTorchClassifier(
        model=class_model,
        clip_values=(0, 1), # image_pixels
        loss=nn.CrossEntropyLoss(),
        # optimizer=optimizer,
        input_shape=(1, 32, 32),
        nb_classes=10,
    )
    #
    # Set up evasion attacks
    #
    fgsm_attack = FastGradientMethod(classifier)
    cw_attack = CarliniL2Method(classifier)
    fool_attack = DeepFool(classifier)
    bim_attack = BasicIterativeMethod(classifier)
    pgd_attack = ProjectedGradientDescent(classifier)
    attacks = [fgsm_attack, fool_attack, bim_attack, cw_attack, pgd_attack]
    attack_n = ["fgsm", "deepfool", "bim", "c&w", "pgd"]
    #
    # Create test set
    #
    print("Generating Adversarial Examples ...")
    # # if you wanted to save the images
    # test_set_imgs = np.empty((10000, 1, 32, 32))
    # test_set_rlabels = np.empty((10000, ))
    # test_set_type = np.empty((10000, ))
    clean_acc = 0
    clean_suds_acc = 0
    res_nosuds = []
    res_wsuds = []
    for i, data in enumerate(tqdm(test_loader), 0):
        print(f"{attack_n[i]} ...")
        inputs, labels = data
        # clean accuracy
        y_pred = classifier.predict(inputs)
        with torch.no_grad():
            sani, _, _ = suds.forward_train(inputs)
        inputs = np.array(inputs).astype(np.float32)
        labels = np.array(labels).astype(np.float32)
        clean_acc += np.sum(np.argmax(y_pred, axis=1) == labels) / len(labels)
        y_pred = classifier.predict(sani)
        clean_suds_acc += np.sum(np.argmax(y_pred, axis=1) == labels) / len(labels)
        # advsarial acc without suds
        adv_examples = attacks[i].generate(inputs)
        y_pred = classifier.predict(adv_examples)
        res_nosuds.append(round(np.sum(np.argmax(y_pred, axis=1) == labels) / len(labels), 4))
        # adversarial acc with suds
        adv_examples = torch.Tensor(adv_examples)
        with torch.no_grad():
            sani, _, _ = suds.forward_train(adv_examples)
        y_pred = classifier.predict(sani)
        res_wsuds.append(round(np.sum(np.argmax(y_pred, axis=1) == labels) / len(labels), 4))
        # ------------------> different approach if want to save images
        # #create adversarial images
        # test_set_imgs[i*2000:(i+1)*2000] = attacks[i].generate(inputs)
        # # create real_labels
        # test_set_rlabels[i*2000:(i+1)*2000] = labels
        # # create type label
        # test_set_type[i*2000:(i+1)*2000] = np.array([i]*2000)

    # 
    # Write out and save stats
    #
    clean_acc = round(clean_acc/len(test_loader), 4)
    clean_suds_acc = round(clean_suds_acc/len(test_loader), 4)
    print("Clean Accuracy: ", clean_acc)
    print("Clean SUDS Accuracy: ", clean_suds_acc)
    print("Adv Accuracy: ", res_nosuds)
    print("Adv Accuracy: ", res_wsuds)

    with open(path, "w") as f:
        # write clean
        f.write(f"clean, {clean_acc}, {clean_suds_acc} \n")
        # write adv results
        for i in range(5):
            f.write(f"{attack_n[i]}, {res_nosuds[i]}, {res_wsuds[i]}\n")

    print("Results saved to: ", path)

    

if __name__ == "__main__":
    main()