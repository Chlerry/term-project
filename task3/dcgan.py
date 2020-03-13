
import numpy as np
from skimage.color import convert_colorspace
from matplotlib import pyplot as plt

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D

import numpy as np
import time
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from matplotlib import pyplot as plt

import imgpatch
import vcompresslib

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
    img_cbcr = (img_cbcr - 16) / (240 - 16)

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

latent_dim = 100
def generator_model(): 
    dropout = 0.4
    depth = 256 # 64+64+64+64
    dim = 4

    model = Sequential()
    # In: 100
    # Out: dim x dim x depth
    model.add(Dense(dim*dim*depth, input_dim=latent_dim))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(Reshape((dim, dim, depth)))
    model.add(Dropout(dropout))

    # In: dim x dim x depth
    # Out: 2*dim x 2*dim x depth/2
    model.add(UpSampling2D())
    model.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    model.add(UpSampling2D())
    model.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
    model.add(Conv2DTranspose(2, 4, padding='same'))
    model.add(Activation('sigmoid'))

    return model

img_rows, img_cols = 16, 16
img_channels = 2

def discriminator_model():
    depth = 64
    dropout = 0.4
    input_shape = (img_rows, img_cols, img_channels)
    
    model = Sequential()
    # In: 16 x 16 x 1, depth = 1
    # Out: 14 x 14 x 1, depth=64
    model.add(Conv2D(depth, 5, strides=2, input_shape=input_shape, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout))

    model.add(Conv2D(depth*2, 5, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout))

    model.add(Conv2D(depth*4, 5, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout))

    model.add(Conv2D(depth*8, 5, strides=1, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout))

    # Out: 1-dim probability
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    return model

def adversarial_model(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', 
                  optimizer=RMSprop(lr=0.0001, decay=3e-8), 
                  metrics=['accuracy'])
    discriminator.trainable = True
    return model

discriminator = discriminator_model()
discriminator.compile(loss='binary_crossentropy', 
                      optimizer=RMSprop(lr=0.0002, decay=6e-8), 
                      metrics=['accuracy'])

generator = generator_model()

adversarial = adversarial_model(generator, discriminator)

def plot_images(gan_image_y, gan_image_train, noise=None):

    images = generator.predict(noise)

    patch_shape = (16, 16, 2)
    # Merge patches back to normal images
    image_shape = (300, 240, 416, 3)
    block_shape = imgpatch.get_block_shape(image_shape, patch_shape)
    img = imgpatch.merge_all_block(images, block_shape)
    t_show = img[: , :, :, :1]
#----------------------2----------------------------------
    t_show2 = gan_image_train[: , :, :, :1]
#----------------------3 ---------------------------------
    img3 = img * (240 - 16) + 16
    gan_image_y3 = gan_image_y * (235 - 16) + 16
    t_show3 = np.concatenate((gan_image_y3, img3), axis=3)

    t_show3 = np.array(t_show3)
    t_show3 = convert_colorspace(t_show3, 'YCbCr', 'RGB')
    print(t_show3.shape)
#----------------------------------------------------------
    n_images = 3
    _, axarr = plt.subplots(3, n_images,figsize=(10, 10), sharey=True)

    for i in range(3):
        axarr[0, i].imshow(t_show[i].reshape((240, 416)), cmap='gray', vmin = 0, vmax = 1)
        axarr[1, i].imshow(t_show2[i].reshape((240, 416)), cmap='gray', vmin = 0, vmax = 1)
        axarr[2, i].imshow(t_show3[i])

    plt.tight_layout()
    plt.show()

def train(test_image_y, test_image_cbcr, train_epochs, save_interval):

    n_decoded = 75 # ---- temperary
    sample_size = n_decoded // 25

    noise_input = None
    # noise_input = np.random.uniform(0.0, 1.0, size=[16, latent_dim])
    noise_input = np.random.uniform(0.0, 1.0, size=[1170, latent_dim])

    for epoch in range(train_epochs):
    # ----- Randomly select data 
        
        # select a random half of images
        indices = np.random.randint(0, n_decoded, size = sample_size)

        gan_image_cbcr = test_image_cbcr[indices, :, :, :]
        gan_image_y = test_image_y[indices, :, :, :]

        patch_shape = (16, 16, 2)
        gan_data_train = imgpatch.get_patch(gan_image_cbcr, patch_shape)

        batch_size = gan_data_train.shape[0]
    # ---------------------------------
        
        noise = np.random.uniform(0.0, 1.0, size=[batch_size, latent_dim])
        images_fake = generator.predict(noise)
        print(noise.shape, gan_data_train.shape)


        x = np.concatenate((gan_data_train, images_fake), axis=0)
        y = np.ones([2*batch_size, 1])
        y[batch_size:, :] = 0

        d_loss = discriminator.train_on_batch(x, y)

        # train the generator (wants discriminator to mistake images as real)
        y = np.ones([batch_size, 1])
        a_loss = adversarial.train_on_batch(noise, y)

        log_msg = "%d: [D loss: %f, acc: %f]" % (epoch, d_loss[0], d_loss[1])
        log_msg = "%s  [A loss: %f, acc: %f]" % (log_msg, a_loss[0], a_loss[1])

        
        if save_interval>0:
            # if epoch > train_epochs - 5:
            # if epoch == train_epochs - 1:
            if (epoch+1) % save_interval == 0:
                plot_images(gan_image_y, gan_image_cbcr, noise=noise_input)
                print(log_msg)