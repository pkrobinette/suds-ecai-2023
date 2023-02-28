"""
Utility functions used in the main folder.
"""
import numpy
import os
import glob
import time
import argparse
from PIL import Image
from rawkit import raw
import torchvision
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import yaml
from yaml.loader import SafeLoader
from .HidingUNet import UnetGenerator
from .RevealNet import RevealNet
from .StegoPy import encode_msg, decode_msg, encode_img, decode_img
from .vae import CNN_VAE
from .classifier import Classifier
from dhide_main import weights_init
import itertools
from tqdm import tqdm
from torch.nn.functional import normalize
from skimage.util import random_noise
import numpy as np
import random

np.random.seed(4)
random.seed(4)

TRANSFORMS_GRAY = transforms.Compose([ 
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize([32, 32]), 
                transforms.ToTensor(),
            ])

TRANSFORMS_RGB = transforms.Compose([
                transforms.Resize([32, 32]), 
                transforms.ToTensor(),
            ])  

SUDS_CONFIG_PATH = "configs/" # CHANGE IF DIFFERENT

def load_data(dataset, batch_size=128):
    """
    Load a dataset for training and testing.
    
    Parameters
    ----------
    dataset : str
        Indicate which dataset to load (mnist or cifar)
    batch_size : int
        The number of images in each batch
    
    Returns
    -------
    train_loader : DataLoader
        Training set
    test_loader : DataLoader
        Test set
    """
    assert (dataset in ["mnist", "cifar"]), "Invalid dataset key; mnist or cifar"
    
    if dataset == "mnist":
        trainset = datasets.MNIST(root='../data', train=True,
                                            download=True, transform=TRANSFORMS_GRAY)
        train_loader = DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=False)
        testset = datasets.MNIST(root='../data', train=False,
                                           download=True, transform=TRANSFORMS_GRAY)
        test_loader = DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)
    else:
        trainset = datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=TRANSFORMS_RGB)
        train_loader = DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=False)
        testset = datasets.CIFAR10(root='../data', train=False,
                                           download=True, transform=TRANSFORMS_RGB)
        test_loader = DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)
    
    return train_loader, test_loader


def load_ddh_mnist(config="gray_ddh"):
    """
    Load the trained ddh networks for ddh steganography.
    
    Parameters
    ----------
    config : str
        The name of the config file in the SUDS_CONFIG_PATH dir. 
        If a different dir, change global var above.
    
    Returns
    -------
    ddh model
    """
    #
    # check config
    #
    if ".yml" not in config or ".yaml" not in config:
        config += ".yml"
    path = SUDS_CONFIG_PATH + config
    assert (os.path.exists(path)), "config path does not exist. Try again."
    # load yaml file
    with open(path) as f:
        data = yaml.load(f, Loader=SafeLoader)
    #
    # Initialize models
    #
    norm_layer = nn.BatchNorm2d
    HnetD = UnetGenerator(input_nc=data["channel_secret"]*data["num_secret"]+data["channel_cover"]*data["num_cover"], output_nc=data["channel_cover"]*data["num_cover"], num_downs=data["num_downs"], norm_layer=norm_layer, output_function=nn.Sigmoid)
    RnetD = RevealNet(input_nc=data["channel_cover"]*data["num_cover"], output_nc=data["channel_secret"]*data["num_secret"], nhf=64, norm_layer=norm_layer, output_function=nn.Sigmoid)
    HnetD.apply(weights_init)
    RnetD.apply(weights_init)
    #
    # Apply DDH saved info
    #
    checkpoint_diff = data["test_diff"] + "/checkPoints/" + "checkpoint.pth.tar"
    checkpoint_diff = torch.load(checkpoint_diff, map_location=torch.device('cpu'))
    H_state_diff = {}
    R_state_diff = {}
    for k in checkpoint_diff["H_state_dict"].keys():
        name = k.replace("module.", "")
        H_state_diff[name] = checkpoint_diff["H_state_dict"][k]
    
    for l in checkpoint_diff["R_state_dict"].keys():
        name = l.replace("module.", "")
        R_state_diff[name] = checkpoint_diff["R_state_dict"][l]
        
    HnetD.load_state_dict(H_state_diff)
    RnetD.load_state_dict(R_state_diff)
    
    print(f"Finished loading MNIST DDH Models...")
    
    return HnetD, RnetD


def load_udh_mnist(config="gray_udh"):
    """
    Load the trained udh networks for udh steganography.
    
    Parameters
    ----------
    config : str
        The name of the config file in the SUDS_CONFIG_PATH dir. 
        If a different dir, change global var above.
    
    Returns
    -------
    udh model
    """
    #
    # Set up parameters
    #
    if ".yml" not in config or ".yaml" not in config:
        config += ".yml"
    path = SUDS_CONFIG_PATH + config
    assert (os.path.exists(path)), "config path does not exist. Try again."
    # load yaml file
    with open(path) as f:
        data = yaml.load(f, Loader=SafeLoader)
    #
    # Initialize models
    #
    norm_layer = nn.BatchNorm2d
    Hnet = UnetGenerator(input_nc=data["channel_secret"]*data["num_secret"], output_nc=data["channel_cover"]*data["num_cover"], num_downs=data["num_downs"], norm_layer=norm_layer, output_function=nn.Tanh)
    Rnet = RevealNet(input_nc=data["channel_cover"]*data["num_cover"], output_nc=data["channel_secret"]*data["num_secret"], nhf=64, norm_layer=norm_layer, output_function=nn.Sigmoid)
    Hnet.apply(weights_init)
    Rnet.apply(weights_init)
    
    #
    # Apply saved checkpoint and weights
    #
    checkpoint = data["test"] + "/checkPoints/" + "checkpoint.pth.tar"
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    H_state = {}
    R_state = {}
    for k in checkpoint["H_state_dict"].keys():
        name = k.replace("module.", "")
        H_state[name] = checkpoint["H_state_dict"][k]
    
    for l in checkpoint["R_state_dict"].keys():
        name = l.replace("module.", "")
        R_state[name] = checkpoint["R_state_dict"][l]
    
    Hnet.load_state_dict(H_state)
    Rnet.load_state_dict(R_state)
    
    Hnet.eval()
    Rnet.eval()
    
    print(f"Finished loading MNIST UDH Models...")
    
    return Hnet, Rnet

def load_vae_suds(channels=1, k_num=128, z_size=128, im_size=32, dataset="mnist"):
    """ 
    Load vae sanitization model, SUDS
    
    Parameters
    ----------
    channels : int
        Number of color channels, 3 = rgb, 1 = grayscale
    k_num : int
        kernal number used during training
    z_size : int
        latent space size used during training
    im_size : int
        image size (32x32)
        
    Returns
    -------
    vae_model : vae
    """
    # 
    # Get intended load directory
    # 
    name = "_"+str(z_size)+"/"
    # if z_size == 128:
    #     name = "/"
    path = "models/sanitization/suds_"+dataset+name+"model.pth"
    print(f"VAE load using --> {path}")
    assert (os.path.exists(path)), "Model does not exist. Try again."
    #
    # Load intended model
    #
    vae_model = CNN_VAE(c_in=channels, k_num=k_num, z_size=z_size, im_size=im_size);
    try:
        vae_model.load_state_dict(torch.load(path));
    except:
        vae_model.load_state_dict(torch.load(path, map_location='cpu'));
    vae_model.eval();
    
    return vae_model

def load_classifier(c_in=1, k_num=128, class_num=10, im_size=32, name="mnist_ddh_marked"):
    """
    Load a classifier trained with poisoned data.
    
    Parameters
    ----------
    name : str
        The name of the save directory within models/data_poison/.
    """
    # 
    # Get intended load directory
    # 
    path = "models/data_poison/"+name+"/model.pth"
    print(f"Data Poison load using --> {path}")
    assert (os.path.exists(path)), "Model does not exist. Try again."
    #
    # Load intended model
    #
    model = Classifier(c_in=c_in, k_num=k_num, class_num=class_num, im_size=im_size);
    try:
        model.load_state_dict(torch.load(path));
    except:
        model.load_state_dict(torch.load(path, map_location='cpu'));
    model.eval();
    
    return model

def use_lsb(covers, secrets, sani_model=None):
    """
    Create containers using lsb hide method.
    
    Parameters
    ----------
    covers : tensor
        cover images
    secrets : tensor
        secrets to hide inside covers
    sani_model : vae
        the sanitizer to use if using sanitization
    
    Returns
    -------
    containers : tensor
        secrets hidden in covers
    C_res : tensor
        difference between original cover and the container
    reveal_secret: tensor
        The secret revealed from the container
    S_res : tensor
        difference between original secret and the recovered secret
    
    Add. Returns
    -----------
    chat : tensor
        A sanitized container
    reveal_sani_secret : tensor
        The secret recovered after sanitization. R(chat)
    """
    if covers.max() <= 1:
        covers = covers.clone().detach()*255
        secrets = secrets.clone().detach()*255
    try:
        _, c, h, w = covers.shape
    except:
        c, h, w = covers.shape
    #
    # Steg hide
    #
    containers = encode_img(covers, secrets, train_mode=True) # steg function is on pixels [0, 255]
    C_res = containers - covers
    reveal_secret = decode_img(containers, train_mode=True)
    S_res = abs(reveal_secret - secrets)
    #
    # Sanitize if sanitzer included
    #
    if sani_model != None:
        with torch.no_grad():
            if c == 3:
                chat, _, _ = sani_model.forward_train(containers) # cifar vae model trained on [0, 1]
            else:
                chat, _, _ = sani_model.forward_train(containers/255) # sani model is on pixels [0, 1]
        reveal_sani_secret = decode_img(chat*255, train_mode=True)
        return containers/255, chat, C_res/255, reveal_secret/255, reveal_sani_secret/255, S_res/255
    
    return containers/255, C_res/255, reveal_secret/255, S_res/255


def use_ddh(covers, secrets, HnetD, RnetD, sani_model=None):
    """
    Create containers using ddh hide method.
    
    Parameters
    ----------
    covers : tensor
        cover images
    secrets : tensor
        secrets to hide inside covers
    sani_model : vae
        the sanitizer to use if using sanitization
    HnetD : ddh hide
    RnetD : ddh reveal
    
    
    Returns
    -------
    containers : tensor
        secrets hidden in covers
    C_res : tensor
        difference between original cover and the container
    reveal_secret: tensor
        The secret revealed from the container
    S_res : tensor
        difference between original secret and the recovered secret
    
    Add. Returns
    -----------
    chat : tensor
        A sanitized container
    reveal_sani_secret : tensor
        The secret recovered after sanitization. R(chat)
    """
    if covers.max() > 1:
        covers = covers.clone().detach()/255
    if secrets.max() > 1:
        secrets = secrets.clone().detach()/255
    try:
        _, c, h, w = covers.shape
    except:
        c, h, w = covers.shape
    #
    # Steg Hide
    #
    H_input = torch.cat((covers, secrets), dim=1)
    with torch.no_grad():
        containers = HnetD(H_input)
    
    C_res = containers - covers
    with torch.no_grad():
        reveal_secret = RnetD(containers)
    S_res = reveal_secret - secrets
    #
    # Sanitize if sanitzer included
    #
    if sani_model != None:
        with torch.no_grad():
            if c == 3:
                chat, _, _ = sani_model.forward_train(containers*255) # cifar suds trained on [0, 255]
            else:
                chat, _, _ = sani_model.forward_train(containers) # sani model is on pixels [0, 1]
            reveal_sani_secret = RnetD(chat)
        return containers, chat, C_res, reveal_secret, reveal_sani_secret, S_res
    
    return containers, C_res, reveal_secret, S_res


def use_udh(covers, secrets, Hnet, Rnet, sani_model=None):
    """
    Create containers using udh hide method.
    
    Parameters
    ----------
    covers : tensor
        cover images
    secrets : tensor
        secrets to hide inside covers
    sani_model : vae
        the sanitizer to use if using sanitization
    Hnet : udh hide
    Rnet : udh reveal
    
    
    Returns
    -------
    containers : tensor
        secrets hidden in covers
    C_res : tensor
        difference between original cover and the container
    reveal_secret: tensor
        The secret revealed from the container
    S_res : tensor
        difference between original secret and the recovered secret
    
    Add. Returns
    -----------
    chat : tensor
        A sanitized container
    reveal_sani_secret : tensor
        The secret recovered after sanitization. R(chat)
    """
    if covers.max() > 1:
        covers = covers.clone().detach()/255
    if secrets.max() > 1:
        secrets = secrets.clone().detach()/255
    try:
        _, c, h, w = covers.shape
    except:
        c, h, w = covers.shape
    #
    # Steg Hide
    #
    with torch.no_grad():
        C_res = Hnet(secrets)
        containers = C_res + covers # maybe gohead and normalize here??
        reveal_secret = Rnet(containers)
        S_res = reveal_secret - secrets
        
    #
    # Sanitize if sanitzer included
    #
    if sani_model != None:
        with torch.no_grad():
            if c == 3:
                chat, _, _ = sani_model.forward_train(containers*255) # cifar suds trained on [0, 255]
            else:
                chat, _, _ = sani_model.forward_train(containers) # sani model is on pixels [0, 1]
            reveal_sani_secret = Rnet(chat)
        return containers, chat, C_res, reveal_secret, reveal_sani_secret, S_res
    
    return containers, C_res, reveal_secret, S_res



def lsb_eval_latent_all(test_loader, master_idx, vae_model):
    """
    Evaluate an entire test_loader to get a list of all z variables.
    
    Parameters
    ----------
    test_loader : DataLoader
        Images to be mapped.
    master_idx : list
        Used to make sure all secrets and covers are the same
        across different hiding techniques.
    vae_model : SUDS
        a sanitization model
        
    Returns
    -------
    z_cover_all : tensor
        All z's for input cover images
    z_cont_all : tensor
        All z's for input container images
    master_l : list
        indices used to create containers.
    """
    master_l = []
    z_cont_all = []
    z_cover_all = []
    for i, data in enumerate(tqdm(test_loader), 0):
        covers, labels = data
        labels = labels.clone().detach()
        covers = covers.clone().detach()
        # add labels to master set
        master_l.append(list(labels))
        secrets = covers[master_idx[i]]
        # create containers
        containers = encode_img(covers*255, secrets*255, train_mode=True)
        # get latent vars
        with torch.no_grad():
            z_container = vae_model.encode(containers/255)
            z_cover = vae_model.encode(covers)
        z_cont_all.append(z_container)
        z_cover_all.append(z_cover)
        
    master_l = list(itertools.chain(*master_l))
    z_cont_all = torch.cat(z_cont_all, dim=0)
    z_cover_all = torch.cat(z_cover_all, dim=0)
    
    return z_cover_all, z_cont_all, master_l


def ddh_eval_latent_all(test_loader, master_idx, HnetD, vae_model):
    """
    Evaluate an entire test_loader to get a list of all z variables.
    
    Parameters
    ----------
    test_loader : DataLoader
        Images to be mapped.
    master_idx : list
        Used to make sure all secrets and covers are the same
        across different hiding techniques.
    HnetD : a hide network for ddh
    vae_model : SUDS
        a sanitization model
        
    Returns
    -------
    z_cover_all : tensor
        All z's for input cover images
    z_cont_all : tensor
        All z's for input container images
    master_l : list
        indices used to create containers.
    """
    master_l = []
    z_cont_all = []
    z_cover_all = []
    for i, data in enumerate(tqdm(test_loader), 0):
        covers, labels = data
        labels = labels.clone().detach()
        covers = covers.clone().detach()
        # add labels to master set
        master_l.append(list(labels))
        secrets = covers[master_idx[i]]
        # create containers
        
        # get latent vars
        with torch.no_grad():
            H_input = torch.cat((covers,secrets), dim=1)
            containers = HnetD(H_input)
            z_container = vae_model.encode(containers)
            z_cover = vae_model.encode(covers)
            
        z_cont_all.append(z_container)
        z_cover_all.append(z_cover)
        
    master_l = list(itertools.chain(*master_l))
    z_cont_all = torch.cat(z_cont_all, dim=0)
    z_cover_all = torch.cat(z_cover_all, dim=0)
    
    return z_cover_all, z_cont_all, master_l


def udh_eval_latent_all(test_loader, master_idx, Hnet, vae_model):
    """
    Evaluate an entire test_loader to get a list of all z variables.
    
    Parameters
    ----------
    test_loader : DataLoader
        Images to be mapped.
    master_idx : list
        Used to make sure all secrets and covers are the same
        across different hiding techniques.
    Hnet : Hide network for UDH
    vae_model : SUDS
        a sanitization model
        
    Returns
    -------
    z_cover_all : tensor
        All z's for input cover images
    z_cont_all : tensor
        All z's for input container images
    master_l : list
        indices used to create containers.
    """
    master_l = []
    z_cont_all = []
    z_cover_all = []
    for i, data in enumerate(tqdm(test_loader), 0):
        covers, labels = data
        labels = labels.clone().detach()
        covers = covers.clone().detach()
        # add labels to master set
        master_l.append(list(labels))
        secrets = covers[master_idx[i]]
        # create containers
        
        # get latent vars
        with torch.no_grad():
            containers = Hnet(secrets) + covers
            z_container = vae_model.encode(containers)
            z_cover = vae_model.encode(covers)
            
        z_cont_all.append(z_container)
        z_cover_all.append(z_cover)
        
    master_l = list(itertools.chain(*master_l))
    z_cont_all = torch.cat(z_cont_all, dim=0)
    z_cover_all = torch.cat(z_cover_all, dim=0)
    
    return z_cover_all, z_cont_all, master_l


def add_gauss(imgs, mu=0, sigma=0.01):
    """
    Add gaussian noise to images.
    
    Parameters
    ----------
    imgs : tensor
        tensor of images
    mu : float
        mean
    sigma : float
        std
    """
    dim = list(imgs.shape)
    n = torch.normal(mu, sigma, dim)
    
    return imgs + n

def add_saltnpep(imgs, pep=0.2):
    """
    Add saltnpepper noise to images.
    
    Parameters
    ----------
    imgs : tensor
        tensor of images
    pep : float
        salt to pepper ratio.
    """
    return torch.tensor(random_noise(imgs, 
                              mode='s&p', 
                              salt_vs_pepper=pep).astype(np.float32))

def add_speckle(imgs, mu=0, var=0.01):
    """
    Add speckle noise to images.
    
    Parameters
    ----------
    imgs : tensor
        tensor of images
    mu : float
        mean
    var : float
        variance
    """
    return torch.tensor(random_noise(imgs, 
                              mode='speckle', 
                              mean=mu, 
                              var=var, 
                              clip=True).astype(np.float32))







# # dirs and files
# raw_file_type = ".CR2"
# raw_dir = args.source + '/'
# converted_dir = args.destination + '/'
# raw_images = glob.glob(raw_dir + '*' + raw_file_type)

# converter function which iterates through list of files
def cr2png(src_folder: str, dst_folder: str):
    """ 
    Not sure where this came from!
    converts a source folder of images to png format in destination folder
    
    Parameters
    ----------
    src_folder : str
        The source folder of images to convert
    dst_folder : str
        The destination folder --> where to save the images
    """
    raw_images = os.listdir(src_folder)
    # import pdb; pdb.set_trace()

    for raw_image in raw_images:
        raw_image = src_folder + "/" + raw_image
        print ("Converting the following raw image: " + raw_image + " to PNG")

        # file vars
        file_name = os.path.basename(raw_image)
        file_without_ext = os.path.splitext(file_name)[0]
        file_timestamp = os.path.getmtime(raw_image)

        # parse CR2 image
        raw_image_process = raw.Raw(raw_image)
        buffered_image = numpy.array(raw_image_process.to_buffer())

        # check orientation due to PIL image stretch issue
        if raw_image_process.metadata.orientation == 0:
            png_image_height = raw_image_process.metadata.height
            png_image_width = raw_image_process.metadata.width
        else:
            png_image_height = raw_image_process.metadata.width
            png_image_width = raw_image_process.metadata.height

        # prep PNG details
        png_image_location = dst_folder + "/" + file_without_ext + '.png'
        png_image = Image.frombytes('RGB', (png_image_width, png_image_height), buffered_image)
        png_image.save(png_image_location, format="png")

        # update PNG file timestamp to match CR2
        os.utime(png_image_location, (file_timestamp,file_timestamp))

        # close to prevent too many open files error
        png_image.close()
        raw_image_process.close()