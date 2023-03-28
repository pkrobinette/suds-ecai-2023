"""
Steganography using LSB technique.

"""

from PIL import Image
import numpy as np
import time
import os
import torch

# ========================================
# Encode Functions
# ========================================

def create_rnd_map(imsize=32, p=0.50, train=False):
    """
    Create a random map.
    """
    m = np.random.rand(imsize, imsize) > p
    
    if train:
        return torch.Tensor(m).unsqueeze(0)
    
    return Image.fromarray(m)


def msg_to_map(message, imsize=32, train_mode=False):
    """
    Create a map from an ascii message.
    """
    # if list, return tensor of combined results.
    if type(message) == list or type(message) == np.ndarray:
        res = torch.empty((len(message), 1, imsize, imsize))
        for i in range(res.shape[0]):
            res[i] = msg_to_map(message[i], imsize, train_mode=True)
        return res
            
    # check message for length
    if len(message)*8 > imsize**2:
        print("shortening message")
        message = message[:((imsize**2)//8)]
        
    # convert string to binary
    bin_m = str_to_bin(message)
    # create image
    b_map = np.full((imsize**2,), False)
    
    # map bits to particular pixels
    cnt = 0
    for char in bin_m:
        for bit in char:
            if bit == "1":
                b_map[cnt] = True
            cnt += 1
            
    #reshape the image (ex: 1024 --> (32, 32))
    b_map = np.reshape(b_map, (imsize, imsize))
    
    if train_mode:
        b_map = np.expand_dims(b_map, axis=2)
        return torch.Tensor(b_map.transpose(2,0,1))
    
    return Image.fromarray(b_map)


def map_to_msg(im, train_mode=False):
    """
    Uncover a plaintxt message from a black and white map.
    """
    # make sure image is in binary format
    if im.max() > 1.0:
        im = torch.where(im/255>0.19, 1.0, 0.0)
    else:
        im = torch.where(im > 0.19, 1.0, 0.0)
    # convert image to numpy array
    data = np.array(im)
    if train_mode:
        data = data.transpose(2, 0, 1)
    
    # flatten the data
    im = data.flatten()

    # list of bytes
    r_map = []

    # create list of bytes from image
    for i in range(im.shape[0]):
        # reset the character
        if i%8 == 0:
            if i != 0:
                if char == "00000000":
                    break
                r_map.append(char)
            char = ""
        char += "1" if im[i] == True else "0"
    
    # convert binary to string and return
    return bin_to_str(r_map)

def encode_msg_bit(cover, msg, bits=4, train_mode=False):
    """
    Embed a message into a cover image using LSB.
    
    Parameters
    ----------
    cover : open image
        image opened with PIL
    msg : str
        message to be embedded in img
    train_mode : bool
        True --> returns tensors [channel, height, width]
        False --> returns carrier image
        
    Returns
    -------
    if train_mode : tensor of image
    if ~train_mode : an image
    
    """
    if train_mode:
        if type(msg) != list: # sometimes a single string sneaks in
            msg = [msg]
        if len(cover.shape) != 4:
            return hide_secret_msgs(cover.unsqueeze(0), msg)
        
        return hide_secret_msgs(cover, msg)
    
    # get pixel values of the image
    img_data = np.array(cover)
    # get img dimensions
    width, height, channels = img_data.shape
    try: # if this doesn't work, the image is a jpeg
        img_data = np.reshape(img_data, width * height * channels)
    except:
        img_data = save_as_png(img_data, save=False) # don't really need to save here
    
    # Check if image is large enough to embed the message
    # Explanation: each character needs 3 pixels.  or 8 bits + 1 flag
    if len(msg) > (width*height*channels)//9:
        end = (width*height*channels)//9
        msg = msg[:127]
        print(len(msg))
        # return "Error: Cover image is not large enough to embed the message."
    img_data = format_msg(img_data, msg)
        
    # reshape image
    img_data = np.reshape(img_data, (width, height, channels))
    
    new_img = Image.fromarray(img_data)
    
    return new_img



def encode_msg(cover, msg, train_mode=False):
    """
    Embed a message into a cover image using LSB.
    
    Parameters
    ----------
    cover : open image
        image opened with PIL
    msg : str
        message to be embedded in img
    train_mode : bool
        True --> returns tensors [channel, height, width]
        False --> returns carrier image
        
    Returns
    -------
    if train_mode : tensor of image
    if ~train_mode : an image
    
    """
    if train_mode:
        if type(msg) != list: # sometimes a single string sneaks in
            msg = [msg]
        if len(cover.shape) != 4:
            return hide_secret_msgs(cover.unsqueeze(0), msg)
        
        return hide_secret_msgs(cover, msg)
    
    t = type(cover) # used later
    
    # get pixel values of the image
    img_data = np.array(cover)
    
    if t == torch.Tensor:
        img_data = img_data.transpose(1,2,0)
    # get img dimensions
    width, height, channels = img_data.shape
    # if t == torch.Tensor():
    #     channels, width, height = img_data.shape
    try:
        img_data = np.reshape(img_data, width * height * channels)
    except:
        img_data = save_as_png(img_data, save=False) # don't really need to save here
    
    # Check if image is large enough to embed the message
    # Explanation: each character needs 3 pixels.  or 8 bits
    if len(msg) > (width*height*channels)//9:
        end = (width*height*channels)//9
        msg = msg[:end-1]
        # return "Error: Cover image is not large enough to embed the message."
    img_data = format_msg(img_data, msg)
        
    # reshape image
    if t == torch.Tensor:
        img_data = np.reshape(img_data, (width, height, channels))
        img_data = img_data.transpose(2, 0, 1)
        return torch.Tensor(img_data)
    else:
        img_data = np.reshape(img_data, (width, height, channels)).astype(np.unit8)
        
    new_img = Image.fromarray(img_data)
    
    return new_img

def hide_secret_msgs(covers, msgs):
    """
    *Helper function. Called by encode_msg during training.*
    This will take a batch from the dataloader and encode msgs
    to create stego images. Need to have image open prior to call.
    
    Parameters
    ----------
    covers : tensor [bs, 3, 32, 32]
        images from cifar-10
    msgs : list [bs]
        msgs to encode.
    
    Returns
    -------
    stegos : tensor [bs, 3, 32, 32]
        stego images in the shape of covers
    """
    stegos = torch.zeros_like(covers)
    
    for i, (cover, msg) in enumerate(zip(covers, msgs)):
        # getting images as 3, 32, 32 --> needs to be 32, 32, 3
        img = cover.permute(1, 2, 0)
        img_data = np.array(img)
        h, w, c = img_data.shape
        img_data = np.reshape(img_data, h*w*c)
        
        # Check if image is large enough to embed the message
        # Explanation: each character needs 3 pixels. 
        if len(msg)*3 > (h*w*c): # image height, width
            msg = msg[:h*w*c-1]
        
        img_data = format_msg(img_data, msg)
            
        # reshape image
        img_data = np.reshape(img_data, (h, w, c))
        stegos[i] = torch.Tensor(img_data.transpose(2, 0, 1))
        
    return stegos # => [bs, c, h, w]


def encode_img(cover, secret, train_mode=False):
    """
    Embed an image into a cover image using LSB.
    
    Parameters
    ----------
    cover : image
        cover image
    secret : image
        secret image
    train_mode : bool
        True = [channel, height, width]. else = [height, width, channel] 
    Returns
    -------
    new_img : str
        path to the saved stego image
    """ 
    if train_mode:
        if len(cover.shape) != 4:
            return hide_secret_imgs(cover.unsqueeze(0), secret.unsqueeze(0))
        
        return hide_secret_imgs(cover, secret)
        
    # if image opened with PIL --> [h, w, c]
    # get pixel values of the image
    cover_data = np.array(cover)
    secret_data = np.array(secret)
    # get img dimensions
    w_c, h_c, c_c = cover_data.shape
    w_s, h_s, c_s = secret_data.shape
    
    # Check if carrier image is larger than embedding image
    if (w_c < w_s) or (h_c < h_s):
        return "Error: Cover image is not large enough to embed the message."
    
    try:
        cover_data = np.reshape(cover_data, w_c * h_c * c_c)
    except:
        cover_data = save_as_png(cover_data, save=False) # don't really need to save here
        
    try:
        secret_data = np.reshape(secret_data, w_s * h_s * c_s)
    except:
        secret_data = save_as_png(secret_data, save=False) # don't really need to save here
        
    # the combined image
    new_data = np.copy(cover_data)
    
    # for each character in the msg
    for i, s_val in enumerate(secret_data):
        rgb1 = int_to_bin(cover_data[i])
        rgb2 = int_to_bin(s_val)
        new_rgb = rgb1[:4] + rgb2[:4]
        new_data[i] = bin_to_int(new_rgb)
        
    # reshape image
    new_data = np.reshape(new_data, (w_c, h_c, c_c))
    return Image.fromarray(new_data)

def hide_secret_imgs(covers, secrets):
    """
    Helper function. Called by encode_img during training.
    
    Parameters
    ----------
    covers : tensor [bs, 3, 32, 32]
        images from cifar-10
    secrets : tensor [bs, 3, 32, 32]
        images from cifar-10
    
    Returns
    -------
    stegos : tensor [bs, 3, 32, 32]
        stego images in the shape of covers
    """
    # [bs, 3, 32, 32]
    stegos = torch.zeros_like(covers)
    
    for i, (cover, secret) in enumerate(zip(covers, secrets)):
        # getting images as 3, 32, 32 --> needs to be 32, 32, 3
        cover = cover.permute(1, 2, 0)
        secret = secret.permute(1, 2, 0)
        
        # get pixel values of the image
        cover_data = np.array(cover).astype(np.uint8)
        secret_data = np.array(secret).astype(np.uint8)
       
        # get img dimensions
        w_c, h_c, c_c = cover_data.shape
        w_s, h_s, c_s = secret_data.shape
        
        # flatten images
        cover_data = np.reshape(cover_data, w_c*h_c*c_c)
        secret_data = np.reshape(secret_data, w_s*h_s*c_s)
        
        # the combined image
        new_data = np.copy(cover_data)
    
        # for each character in the msg
        for k, s_val in enumerate(secret_data):
            rgb1 = int_to_bin(cover_data[k])
            rgb2 = int_to_bin(s_val)
            new_rgb = rgb1[:4] + rgb2[:4]
            new_data[k] = bin_to_int(new_rgb)
        
        # reshape image
        new_data = np.reshape(new_data, (w_c, h_c, c_c))
        stegos[i] = torch.Tensor(new_data.transpose(2, 0, 1)) # go back to [c, h, w]
        
    return stegos # => [bs, c, h, w]


    
# ========================================
# Decode Functions
# ========================================
    
def decode_msg(carrier):
    """
    Decode a secret message from a stego image using LSB.
    
    Parameters
    ----------
    carrier : image
        can be in either tensor format or image format 

    Returns
    -------
    msg : str
        The secret message embedded in the image
    """
    try: # will work if tensor, not work if actual image
        dims = list(carrier.shape)
        
        if len(dims) == 4 and dims[0] == 1:
            carrier = carrier.squeeze(0)
            dims.pop(0)
        
        if dims[2] != 3: # if true, in tensor format [3, h, w]
            carrier = carrier.permute(1, 2, 0) # puts it as [h, w, 3]
        img_data = np.array(carrier) # imgs are 0 to 1 now!!
        if img_data.max() <= 1:
            img_data = img_data*255
        width, height, channels = img_data.shape
        img_data = np.reshape(img_data, width*height*channels) # flatten the array
            
    except: # carrier is an open image
        img_data = np.array(carrier)
        width, height, channels = img_data.shape
        img_data = np.reshape(img_data, width*height*channels)
    
    end = False
    bin_msg = []
    
    ptr = 0
    
    while end == False:
        tmp_bin = []
        
        for i in range(9):
            curr_val = img_data[ptr]
            # value is even
            if curr_val %2 == 0:
                tmp_bin.append('0')
            else:
                tmp_bin.append('1')
            
            # check if we have reached the end of the message
            if i == 8 and tmp_bin[-1] == "1":
                end = True
            
            # increment ptr
            ptr += 1
        
        bin_msg.append("".join(tmp_bin[:-1])) # skip flag
    
    # convert binary message to a string
    scrt_msg = bin_to_str(bin_msg)
    
    return scrt_msg

    
    
def decode_img(carrier, train_mode=False):
    """
    Decode a secret image from a stego image using LSB.
    
     Parameters
    ----------
    carrier : image
        can be in either tensor format or image format 

    Returns
    -------
    secret image : image
        The secret image embedded in the carrier
    """
    if train_mode:
        if len(carrier.shape) != 4:
            return reveal_secret_imgs(carrier.unsqueeze(0))
        return reveal_secret_imgs(carrier)
    
    try: # will work if tensor, not work if actual image
        dims = list(carrier.shape)
        
        if len(dims) == 4 and dims[0] == 1: # only one image
            carrier = carrier.squeeze(0)
            dims.pop(0)
            
        if dims[2] != 3: # needs to be in [h, w, 3]
            carrier = carrier.permute(1, 2, 0)
        
        img_data = np.array(carrier) # imgs are 0 to 1 now!!
        if img_data.max() <= 1:
            img_data = img_data*255
            
        img_data = img_data.astype(np.uint8)
        
        width, height, channels = img_data.shape
        img_data = np.reshape(img_data, width*height*channels) # flatten the array
    
    except:
        # get pixel values of the image
        img_data = np.array(carrier)
        # get img dimensins
        width, height, channels = img_data.shape
        img_data = np.reshape(img_data, width*height*channels)
    
    new_data = np.zeros_like(img_data)
    
    for i, val in enumerate(img_data):
        rgb_b = int_to_bin(val)
        pix = rgb_b[4:] + "0000"
        new_data[i] = bin_to_int(pix)
    
    new_data = np.reshape(new_data, (width, height, channels))
    return new_data
    # return Image.fromarray(new_data) # error making image
    
    
def reveal_secret_imgs(carriers):
    """
    Helper function. Called by encode_img during training.
    
    Parameters
    ----------
    covers : tensor [bs, 3, 32, 32]
        images from cifar-10
    secrets : tensor [bs, 3, 32, 32]
        images from cifar-10
    
    Returns
    -------
    stegos : tensor [bs, 3, 32, 32]
        stego images in the shape of covers
    """
    # [bs, 3, 32, 32]
    secrets = torch.zeros_like(carriers)
    
    for i, carrier in enumerate(carriers):
        # getting images as 3, 32, 32 --> needs to be 32, 32, 3
        secret = torch.tensor(decode_img(carrier, train_mode=False))
        secrets[i] = secret.permute(2, 0, 1)
    
    return secrets # => [bs, c, h, w]

# def decode_img(carrier):
#     """
#     Decode a secret image from a stego image using LSB.
    
#      Parameters
#     ----------
#     carrier : image
#         can be in either tensor format or image format 

#     Returns
#     -------
#     secret image : image
#         The secret image embedded in the carrier
#     """
#     try: # will work if tensor, not work if actual image
#         dims = list(carrier.shape)
        
#         if len(dims) == 4 and dims[0] == 1: # only one image
#             carrier = carrier.squeeze(0)
#             dims.pop(0)
            
#         if dims[2] != 3: # needs to be in [h, w, 3]
#             carrier = carrier.permute(1, 2, 0)
        
#         img_data = np.array(carrier) # imgs are 0 to 1 now!!
#         if img_data.max() <= 1:
#             img_data = img_data*255
            
#         img_data = img_data.astype(np.uint8)
        
#         width, height, channels = img_data.shape
#         img_data = np.reshape(img_data, width*height*channels) # flatten the array
    
#     except:
#         # get pixel values of the image
#         img_data = np.array(carrier)
#         # get img dimensins
#         width, height, channels = img_data.shape
#         img_data = np.reshape(img_data, width*height*channels)
    
#     new_data = np.zeros_like(img_data)
    
#     for i, val in enumerate(img_data):
#         rgb_b = int_to_bin(val)
#         pix = rgb_b[4:] + "0000"
#         new_data[i] = bin_to_int(pix)
    
#     new_data = np.reshape(new_data, (width, height, channels))
#     return new_data
#     # return Image.fromarray(new_data) # error making image




# ========================================
# Helper Functions
# ========================================
    
def str_to_bin(msg):
    """
    Convert ascii to binary
    
    Parameters
    """
    # convert string to array of unicode values
    unicode = [ord(c) for c in msg]
    # array of binary values
    b = [bin(u)[2:].zfill(8) for u in unicode]
    
    return b

    
def bin_to_str(b):
    """
    Convert binary to ascii
    
    Parameters
    ----------
    b : list
        list of binary values making up the str
        
    msg : str
        decoded string
    """
    # convert binary to unicode values, the "2" indicates base 2 --> binary
    unicode = [int(i,2) for i in b]
    # convert unicode to characters
    chars = [chr(u) for u in unicode]
    
    return "".join(chars) # concat chars into a string


def int_to_bin(i):
    """
    Convert integers to binary.
    """
    return f'{i:08b}'


def bin_to_int(b):
    """
    Convert binary to integer.
    """
    return int(b, 2)


def format_msg(img_data, msg):
    """
    Helper function. 
    Used to embed message into a flattened image array.
    
    Parameters
    ----------
    img_data : array
        flattened image data
    msg : string
        the message to embed in the cover image
        
    Returns
    -------
    new_data : array
        flat array with data embedded
    """
    # Convert the message to binary
    bin_msg = str_to_bin(msg)
        
    ptr = 0 # points to current pixel/color channel
    cnt = 1 # used to flag the end of the msg
    
    # for each character in the msg
    for char in bin_msg:
        # for each bit in the character:
        for b in char:
            # change value of last pixel etc.
            curr_val = img_data[ptr]
            if int(b) == 0:
                # the curr_val is odd
                if curr_val %2 != 0:
                    curr_val -= 1
            else:
                if curr_val %2 == 0:
                    curr_val += 1  
            img_data[ptr] = curr_val
            ptr += 1
        
        # add the msg flg if end of the message
        curr_val = img_data[ptr]
        if cnt == len(bin_msg):
            # msg ended, make flag odd (1)
            if curr_val %2 == 0:
                curr_val += 1
        else:
            # msg cong, make flag even (0)
            if curr_val %2 != 0:
                curr_val -= 1
                
        img_data[ptr] = curr_val
        
        #increment ptr and cnt
        ptr += 1
        cnt += 1
        
    return img_data


def save_as_png(img_data, img_path="new_png", save=False):
    """
    Save a jpeg as png image.
    """
    print("reformatting image ...")
    img_data = np.reshape(img_data, width*height*4)
    img_data = np.delete(img_data, np.arange(3, img_data.size, 4))
    if save:
        Image.fromarray(np.reshape(img_data, (height, width, 3))).save(img_path+".png")
        
    return img_data





if __name__ == "__main__":
    while True:
        print("\n* ---------- * *----------- * * ----------- * ")
        print("\t[0]: Steganography Encode")
        print("\t[1]: Steganography Decode")
        print("\t[2]: Exit ")
        print("* ---------- * *----------- * * ----------- * \n")
        
        print("Select an action from the menu above. (0, 1, or 2)")
        key = int(input(" >> "))
    
        if key == 0:
            print("\nWhich file would you like to use as a cover?")
            img_path = input("file path (including ext.): ")
            img_path = str(img_path)
        
            msg = input("\nWrite the message you would like to hide? \n >> ")
            msg = str(msg)
        
            new_path = encode(img_path, msg)
        
            print("\nEncoding complete. The stego image can be found here: '{}'".format(new_path))
            time.sleep(2)
            
            c = input("\nContinue? (y/n) >> ")
            if c.lower() == "n":
                break
            os.system('clear')
        
        elif key == 1:
            print("\nWhich file would you like to decode?")
            img_path = input("file path (including ext.): ")
            img_path = str(img_path)
        
            msg = decode(img_path)
        
            print("\nSECRET >> {}".format(msg))
            time.sleep(2)
            
            c = input("\nContinue? (y/n) >> ")
            if c.lower() == "n":
                break
            os.system('clear')
        
        else:
            print("Exiting ... ")
            break
        
        
        
    
    
    
    
    
    
    
    
# def encode_to_dir(img_path, msg, to_dir):
#     """
#     Embed a message into a cover image using LSB.
    
#     Parameters
#     ----------
#     img_path : str
#         path to the image
#     msg : str
#         message to be embedded in img
#     to_dir : str
#         path to save new image
        
#     Returns
#     -------
#     new_img : str
#         path to the saved stego image
#     """
#     # load the image
#     img = Image.open(img_path)
#     # get img dimensions
#     width, height = img.size
#     # get pixel values of the image
#     img_data = np.array(img)
#     try:
#         img_data = np.reshape(img_data, width * height * 3)
#     except:
#         print("reformatting image ...")
#         img_data = np.reshape(img_data, width*height*4)
#         img_data = np.delete(img_data, np.arange(3, img_data.size, 4))
#         Image.fromarray(np.reshape(img_data, (height, width, 3))).save(img_path)
    
#     # Check if image is large enough to embed the message
#     # Explanation: each character needs 3 pixels. 
#     if len(msg)*3 > (width*height):
#         return "Error: Cover image is not large enough to embed the message."
    
#     # Convert the message to binary
#     bin_msg = str_to_bin(msg)
    
#     ptr = 0 # points to current pixel/color channel
#     cnt = 1 # used to flag the end of the msg
    
#     # for each character in the msg
#     for char in bin_msg:
#         # for each bit in the character:
#         for b in char:
#             # change value of last pixel etc.
#             curr_val = img_data[ptr]
#             if int(b) == 0:
#                 # the curr_val is odd
#                 if curr_val %2 != 0:
#                     curr_val -= 1
#             else:
#                 if curr_val %2 == 0:
#                     curr_val += 1  
#             img_data[ptr] = curr_val
#             ptr += 1
        
#         # add the msg flg if end of the message
#         curr_val = img_data[ptr]
#         if cnt == len(bin_msg):
#             # msg ended, make flag odd (1)
#             if curr_val %2 == 0:
#                 curr_val += 1
#         else:
#             # msg cong, make flag even (0)
#             if curr_val %2 != 0:
#                 curr_val -= 1
                
#         img_data[ptr] = curr_val
        
#         #increment ptr and cnt
#         ptr += 1
#         cnt += 1
        
#     # reshape image
#     img_data = np.reshape(img_data, (height, width, 3))
    
#     new_img = Image.fromarray(img_data)
#     name = img_path.split(".")[0]
#     new_img.save(to_dir+ "/" +name.split("/")[-1]+"_secret.png") ##  must be saved as a png because jpg is lossy compressed (not exact save)