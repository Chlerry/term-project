import numpy as np
from matplotlib import pyplot as plt

import os
from PIL import Image
from skimage.util.shape import view_as_blocks

# Note: Data shape example are only for 240x416 images
#       480x832 images will double the block hight and block width

# Print out a block of patches one by one
#       patch (390, 16, 16, 3): block of patches 
#       block_shape (15, 26)
def print_patches(patch, block_shape):
    n_rows = block_shape[0]
    n_cols = block_shape[1]
    _, axarr = plt.subplots(n_rows, n_cols,figsize=(10, 10), sharey=True)

    for i in range(n_rows):
        for j in range (n_cols):
            axarr[i, j].imshow(patch[i*n_cols + j])

    plt.show()

# Merge a block of patches to one single image
#       patch (15*26, 16, 16, 3): a numpy array stores 15 x 26 patches of 16 x 16 frame
#       merge_block returns a 240 x 416 image
def merge_block(patch, block_shape):

    # (15, 26)
    (block_hight, block_width) = block_shape
    # (16, 16, 3)
    (frame_hight, frame_width, n_channel) = patch.shape[1:]
    
    # Reshape patch: (15*26, 16, 16, 3) - > (15, 26, 16, 16, 3)
    patch = patch.reshape(list(block_shape) + list(patch.shape[1:]))

    # image_shape: (16*15, 16*26, 3)
    image_hight = block_hight * frame_hight
    image_width = block_width * frame_width
    image_shape = (image_hight, image_width, n_channel)

    # Initialize img array
    img = np.empty(image_shape, dtype = np.float32)

    # Assign pixels to img row by row 
    #       block_hight: 15, frame_hight: 16
    for i in range(block_hight):
        for j in range(frame_hight):
            # Combine the jth row of all patches in ith block row to(1, 416, 3)
            img[i*frame_hight + j] = patch[i][:,j].reshape((1, image_width, n_channel))
            
    return img

# Merge multiple images' patch data to an image set (used for test data)
#       block_set for test data:(29250, 16, 16, 3)
#       block_shape: (15, 26)
#       Return img: n_test 240 x 416 images
def merge_all_block(block_set, block_shape):
    n_frame = block_set.shape[0]

    (block_hight, block_width) = block_shape
    block_size = block_hight * block_width
    n_image = n_frame // block_size
    
    (frame_hight, frame_width, n_channel) = block_set.shape[1:]
    image_hight = block_hight * frame_hight
    image_width = block_width * frame_width

    # image_set_shape: (75, 16*15, 16*26, 3)
    image_set_shape = (n_image, image_hight, image_width, n_channel)

    # Initialze image set (array)
    img = np.empty(image_set_shape, dtype = np.float32)

    # Separate patches for each image, and the patches for each image,
    #       then add the image to img set
    for i in range(n_image):
        begin = i * block_size
        end = begin + block_size
        patch = block_set[begin:end]
        img[i] = merge_block(patch, block_shape)

    return img

# Load image data from local path
#       Return normalized image array
def load_data(data_path):
    # Obtain a list of image names from data_path
    image_names = os.listdir(data_path)

    # The number of images in the path
    n_image = len(image_names)
    
    # Obtain the first image 
    input_im = np.array(Image.open(data_path + '/' + image_names[0]))
    
    # Reshape image from (240, 416, 4) -> (240, 416, 3)
    image_shape = input_im[...,:3].shape 

    # Initialize image_data (n_image, 240, 416, 3)
    image_data = np.empty([n_image] + list(image_shape), dtype = np.int32)

    # Load each data from the data_path
    for i in range(n_image):  
        image_path = data_path + '/' + image_names[i]
        image_data[i] =  np.array(Image.open(image_path))[...,:3]

    # Normalize image_data
    image_data = image_data / 255.0

    return image_data

# Return block_shape: 
#       (15, 26) for BlowingBubbles_416x240_50 and RaceHorses_416x240_30
#       (30, 52) for BasketballDrill_832x480_50
def get_block_shape(image_shape, patch_shape):
    # image_data.shape: (n_image, 240, 416, 3)
    (_, image_hight, image_width, _) = image_shape

    # patch_shape: (16, 16, 3)
    (patch_hight, patch_width, _) = patch_shape

    # block_hight: 240 / 16 -> 15
    block_hight = image_hight // patch_hight
    # block_width: 416/ 16 -> 26 
    block_width = image_width // patch_width

    return (block_hight, block_width)

# Convert all images to patches 
#       Return (117000, 16, 16, 3) patch_data for BlowingBubbles_416x240_50 and RaceHorses_416x240_30
#       Return (4*117000, 16, 16, 3) patch_data for BasketballDrill_832x480_50
def get_patch(image_data, patch_shape):

    # image_data.shape: (300, 240, 416, 3)
    n_image = image_data.shape[0]
    
    # block_hight: 15, block_width: 26
    (block_hight, block_width) = get_block_shape(image_data.shape, patch_shape)

    # n_patch: 15 * 16 -> 390
    n_patch = block_hight * block_width

    # Initialize patch_data (390*300, 16, 16, 3)
    patch_data = np.empty([n_patch * n_image] + list(patch_shape) , dtype = np.float32)
    
    # block_reshape: (390, 16, 16, 3)
    block_reshape = [n_patch] + list(patch_shape)
    for i in range(n_image):  
        # Get each patch block of a image, and organize the patch block to (390, 16, 16, 3)
        block = view_as_blocks(image_data[i], block_shape = patch_shape).reshape(block_reshape)

        # Add patch from a patch block to the patch_data one by one
        for j in range(n_patch):
            patch_data[i*n_patch + j] = block[j]

    print(patch_data.shape)
    return patch_data