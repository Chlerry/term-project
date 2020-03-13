
import numpy as np
from skimage.color import convert_colorspace
from matplotlib import pyplot as plt

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D


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

# Ratio options are: 1/32, 1/16, 1/8, 1/4, 1/2
def luminance_autocoder(train_data, patch_shape, ratio):
    input_img = Input(shape=patch_shape)

    e = Conv2D(64, (7, 7), activation='relu', padding='same')(input_img)
    if(ratio == '1/2'):
        e = MaxPooling2D((2, 1), padding='same')(e)
    else:
        e = MaxPooling2D((2, 2), padding='same')(e)

    e = Conv2D(32, (5, 5), activation='relu', padding='same')(e)
    if(ratio == '1/8'):
        e = MaxPooling2D((2, 1), padding='same')(e)
    elif (ratio == '1/16' or ratio == '1/32'):
        e = MaxPooling2D((2, 2), padding='same')(e)

    e = Conv2D(16, (1, 1), activation='relu', padding='same')(e)
    if (ratio == '1/32'):
        e = MaxPooling2D((2, 1), padding='same')(e)
        
    e = Conv2D(8, (3, 3), activation='relu', padding='same')(e)
    encoded = Conv2D(4, (3, 3), activation='relu', padding='same')(e)

    d = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
    d = Conv2D(8, (3, 3), activation='relu', padding='same')(d)
    
    if (ratio == '1/32'):
        d = UpSampling2D((2, 1))(d)
    d = Conv2D(16, (1, 1), activation='relu', padding='same')(d)
    
    if(ratio == '1/8'):
        d = UpSampling2D((2, 1))(d)
    elif (ratio == '1/16' or ratio == '1/32'):
        d = UpSampling2D((2, 2))(d)
    d = Conv2D(32, (5, 5), activation='relu', padding='same')(d)
    
    if(ratio == '1/2'):
        d = UpSampling2D((2, 1))(d)
    else:
        d = UpSampling2D((2, 2))(d)
    d = Conv2D(64, (7, 7), activation='relu', padding='same')(d)
    decoded = Conv2D(1, (1, 1), activation='linear')(d)

    autoencoder = Model(input_img, decoded)
    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(train_data, train_data, epochs=1, batch_size=25)

    return autoencoder