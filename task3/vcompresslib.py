import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D

def test_model(train_data, patch_shape):
    input_img = Input(shape=patch_shape)

    e = Conv2D(64, (7, 7), activation='relu', padding='same')(input_img)
    e = MaxPooling2D((2, 2), padding='same')(e)
    e = Conv2D(32, (5, 5), activation='relu', padding='same')(e)
    e = MaxPooling2D((2, 2), padding='same')(e)
    e = Conv2D(16, (1, 1), activation='relu', padding='same')(e)
    e = Conv2D(8, (3, 3), activation='relu', padding='same')(e)
    encoded = Conv2D(4, (3, 3), activation='relu', padding='same')(e)

    d = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
    d = Conv2D(8, (3, 3), activation='relu', padding='same')(d)
    d = Conv2D(16, (1, 1), activation='relu', padding='same')(d)
    d = UpSampling2D((2, 2))(d)
    d = Conv2D(32, (5, 5), activation='relu', padding='same')(d)
    d = UpSampling2D((2, 2))(d)
    d = Conv2D(64, (7, 7), activation='relu', padding='same')(d)
    decoded = Conv2D(3, (1, 1), activation='linear')(d)

    autoencoder = Model(input_img, decoded)
    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(train_data, train_data, epochs=1, batch_size=25)

    return autoencoder

def get_psnr(test_image, decoded_image):

    PSNR = 0
    n_test = test_image.shape[0]

    for i in range(n_test):
        MSE = tf.keras.losses.MeanSquaredError()
        test_mse = MSE(test_image[i], decoded_image[i])
        PSNR += 10.0 * np.log10(1.0 / test_mse)
    
    PSNR /= n_test

    return PSNR