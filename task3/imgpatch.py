import numpy as np
from matplotlib import pyplot as plt

import os
from PIL import Image
from skimage.util.shape import view_as_blocks

# patch is block of patches (390, 16, 16, 3)
# block_shape: (15, 26)
def print_patches(patch, block_shape):
    n_rows = block_shape[0]
    n_cols = block_shape[1]
    _, axarr = plt.subplots(n_rows, n_cols,figsize=(10, 10), sharey=True)

    for i in range(n_rows):
        for j in range (n_cols):
            axarr[i, j].imshow(patch[i*n_cols + j])

    plt.show()

# --- need fix # block (16, 16, 15, 26, 3) is a numpy array stores 16 x 16 patches of 15 x 26 frame
# --- need fix # merge_block returns a 240 x 416 image
# def merge_block(block):
#     image_shape = (block.shape[0]*block.shape[2], block.shape[1]*block.shape[3], block.shape[4])
#     img = np.empty(image_shape, dtype = np.float32)
#     for i in range(16):
#         for j in range(15):
#             img[i*15 + j] = block[i][:,j].reshape((1, image_shape[1], 3))
            
#     return img

def load_data(data_path):
    image_names = os.listdir(data_path)
    n_image = len(image_names)
    
    input_im = np.array(Image.open(data_path + '/' + image_names[0]))
    
    # image_shape: (240, 416, 3) -> (240, 416, 3)
    image_shape = input_im[...,:3].shape 

    image_data = np.empty([n_image] + list(image_shape), dtype = np.int32)

    for i in range(n_image):  
        image_path = data_path + '/' + image_names[i]
        image_data[i] =  np.array(Image.open(image_path))[...,:3]

    return image_data

# return block_shape: (15, 26) for BlowingBubbles_416x240_50 and RaceHorses_416x240_30
def get_block_shape(image_shape, patch_shape):
    # image_data.shape: (n_image, 240, 416, 3)
    (_, image_hight, image_width, _) = image_shape

    # patch_shape: (16, 16, 3)
    (patch_hight, patch_width, _) = patch_shape

    # block_hight: 240 / 16 -> 15
    block_hight = image_hight // patch_hight
    # block_width: 416/ 16 -> 26 
    block_width = image_width // patch_width

    # print("block_shape: ", (block_hight, block_width))
    return (block_hight, block_width)


# return (117000, 16, 16, 3) patch_data for BlowingBubbles_416x240_50 and RaceHorses_416x240_30
# return (4*117000, 16, 16, 3) patch_data for BasketballDrill_832x480_50

def get_patch(image_data, patch_shape):

    # image_data.shape: (300, 240, 416, 3)
    n_image = image_data.shape[0]
    
    # 15, 26
    (block_hight, block_width) = get_block_shape(image_data.shape, patch_shape)

    # 15 * 16 -> 390
    n_patch = block_hight * block_width

    # patch_data: (390*300, 16, 16, 3)
    patch_data = np.empty([n_patch * n_image] + list(patch_shape) , dtype = np.int32)
    # block_reshape: (390, 16, 16, 3)
    block_reshape = [n_patch] + list(patch_shape)
    for i in range(n_image):  
        block = view_as_blocks(image_data[i], block_shape = patch_shape).reshape(block_reshape)

        for j in range(n_patch):
            patch_data[i*n_patch + j] = block[j]

    print(patch_data.shape)
    return patch_data