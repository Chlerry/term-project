import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D

import imgpatch

# Ratio options are: 1/32, 1/16, 1/8, 1/4, 1/2
def model1(train_data, patch_shape, ratio):
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
    encoded = Conv2D(3, (3, 3), activation='relu', padding='same')(e)

    d = Conv2D(3, (3, 3), activation='relu', padding='same')(encoded)
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
    decoded = Conv2D(3, (1, 1), activation='linear')(d)

    autoencoder = Model(input_img, decoded)
    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(train_data, train_data, epochs=5, batch_size=25)

    return autoencoder

# Ratio options are: 1/32, 1/16, 1/8, 1/4, 1/2
def model2(train_data, patch_shape, ratio):
    input_img = Input(shape=patch_shape)

    
    if(ratio == '1/2'):
        e = Conv2D(64, (7, 7), activation='relu', strides=(1,2), padding='same')(input_img)
    else:
        e = Conv2D(64, (7, 7), activation='relu', strides=(2,2), padding='same')(input_img)

    
    if(ratio == '1/8'):
        e = Conv2D(32, (5, 5), activation='relu', strides=(1,2),padding='same')(e)
    elif (ratio == '1/16' or ratio == '1/32'):
        e = Conv2D(32, (5, 5), activation='relu', strides=(2,2),padding='same')(e)
    else:
        e = Conv2D(32, (5, 5), activation='relu', padding='same')(e)

    
    if (ratio == '1/32'):
        e = Conv2D(16, (1, 1), activation='relu', strides=(1,2),padding='same')(e)
    else:
        e = Conv2D(16, (1, 1), activation='relu', padding='same')(e)
        
    e = Conv2D(8, (3, 3), activation='relu', padding='same')(e)
    encoded = Conv2D(3, (3, 3), activation='relu', padding='same')(e)

    d = Conv2DTranspose(3, (3, 3), activation='relu', padding='same')(encoded)
    d = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(d)
    
    if (ratio == '1/32'):
        d = Conv2DTranspose(16, (1, 1), activation='relu', strides=(1,2),padding='same')(d)
    else:
        d = Conv2DTranspose(16, (1, 1), activation='relu', padding='same')(d)
    
    if(ratio == '1/8'):
        d = Conv2DTranspose(32, (5, 5), activation='relu', strides=(1,2),padding='same')(d)
    elif (ratio == '1/16' or ratio == '1/32'):
        d = Conv2DTranspose(32, (5, 5), activation='relu', strides=(2,2),padding='same')(d)
    else:
        d = Conv2DTranspose(32, (5, 5), activation='relu', padding='same')(d)
    
    
    if(ratio == '1/2'):
        d = Conv2DTranspose(64, (7, 7), activation='relu', strides=(1,2),padding='same')(d)
    else:
        d = Conv2DTranspose(64, (7, 7), activation='relu', strides=(2,2),padding='same')(d)
    
    decoded = Conv2DTranspose(3, (1, 1), activation='linear')(d)

    autoencoder = Model(input_img, decoded)
    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(train_data, train_data, epochs=5, batch_size=25)

    return autoencoder

# Calculate average PSNR value
def get_psnr(test_image, decoded_image):

    PSNR = 0
    n_test = test_image.shape[0]

    for i in range(n_test):
        MSE = tf.keras.losses.MeanSquaredError()
        test_mse = MSE(test_image[i], decoded_image[i])
        PSNR += 10.0 * np.log10(1.0 / test_mse)
    
    PSNR /= n_test

    return PSNR

# Obtain decoded image patches from the CNN model, and merge patches back to normal images
def get_decoded_image(autoencoder, test_data, patch_shape, image_shape):
    # Obtain decoded image for test_data
    decoded_patches = autoencoder.predict(test_data)

    # Limit pixel value range to [0, 1]
    decoded_patches = np.minimum(decoded_patches, np.ones(decoded_patches.shape, dtype = np.float32))
    decoded_patches = np.maximum(decoded_patches, np.zeros(decoded_patches.shape, dtype = np.float32))

    # Merge patches back to normal images
    block_shape = imgpatch.get_block_shape(image_shape, patch_shape)
    decoded_image = imgpatch.merge_all_block(decoded_patches, block_shape)

    return decoded_image