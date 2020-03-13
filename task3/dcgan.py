
import numpy as np
from skimage.color import convert_colorspace
from matplotlib import pyplot as plt

def detach_luminance(img):
    # Convert image from RGB color space to YCbCr color space
    #   input array must be dtype=np.uint8 nparray
    img_ycbcr = convert_colorspace(img, 'RGB', 'YCbCr')

    # Detach the luminance from Cb and Cr
    img_y = img_ycbcr[: , :, :, :1]
    img_cbcr = img_ycbcr[: , :, :, 1:]

    # Normalize the image 
    #   Y is scales to a different range of 16 to 235
    img_y = (img_y - 16) / (235 - 16)
    #   CB and CR are scaled to a different range of 16 to 240.
    img_cbcr = (img_cbcr - 26) / (240 - 16)

    return img_y, img_cbcr

