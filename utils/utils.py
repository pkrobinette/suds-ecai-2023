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
from dhide_main import weights_init

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