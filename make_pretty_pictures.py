"""
Create a diagram for generated steg images. Used for RQ1, RQ1, and RQ5.

"""

import matplotlib.pyplot as plt
import os
from PIL import Image
import os

# where pictures are held for each im-type
DIRS = {
    "suds": [
        'C',
        'C_prime',
        'C_hat_suds',
        'C_res_suds',
        'S',
        'S_hat_suds'
    ],
    "gauss":[
        'C',
        'C_prime',
        'C_hat_gauss',
        'C_res_gauss',
        'S',
        'S_hat_gauss'
    ],
    "cifar": [
        'C',
        'C_prime',
        'C_hat',
        'C_res',
        'S',
        'S_hat_suds'
    ]}

def pretty_picture(input_path, save_name, noise="suds", imsize=64, dataset="mnist"):
    """
    Make a pretty picture
    """
    # Set the dimensions of each individual image
    width, height = imsize, imsize
    
    # Set the number of images per row and column in the final image
    images_per_row = 6
    images_per_column = 6
    
    # Create a blank image with the combined dimensions
    imtype = "L" if dataset == "mnist" else "RGB"
    combined_image = Image.new(imtype, (width * images_per_row, height * images_per_column))

    if dataset == "cifar":
        noise = "cifar"
        n = lambda j: ["S_hat_lsb", "S_hat_ddh", "S_hat_udh"][j]
        
    directories = DIRS[noise]
    
    hide_type = ["lsb_demo", "ddh_demo", "udh_demo"]
    
    # Iterate through the images and paste them into the combined image
    for i, h in enumerate(hide_type):
        # each type only gets 2
        for j in range(2):
            for col, d in enumerate(directories):
                if col == 5 and dataset == "cifar":
                    p = input_path+h+"/"+n(i)
                else:
                    p = input_path+h+"/"+d

                image_file = str((i*2)+j) + ".jpg"
                
                # print(p, image_file)
                image = Image.open(os.path.join(p, image_file))
                image = image.resize((width, height))
        
                # Calculate the position of the current image
                x = col * width
                y = ((i*2)+j) * height
        
                # Paste the current image into the combined image
                combined_image.paste(image, (x, y))
    
    # Save the combined image
    save_path = input_path + save_name if input_path[-1] == "/" else input_path + "/" + save_name
    combined_image.save(save_path)
    
    print("Saved to: ", save_path)
    
    
if __name__ == "__main__":
    input_path = "results/noise_vs_suds_demo/"
    output_path = "suds-pretty-picture.pdf"
    # suds-mnist
    pretty_picture(input_path, output_path, noise="suds", dataset="mnist")
    # noise-mnist
    output_path = "noise-pretty-picture.pdf"
    pretty_picture(input_path, output_path, noise="gauss", dataset="mnist")
    # suds-cifar
    input_path = "results/sanitize_demo_cifar/"
    output_path = 'cifar-suds-pretty-picture.pdf'
    pretty_picture(input_path, output_path, noise="suds", dataset="cifar")
    