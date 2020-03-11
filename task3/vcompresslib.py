from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D

def test_model(train_data, patch_shape):
    input_img = Input(shape=patch_shape)

    e = Conv2D(64, (7, 7), activation='relu', padding='same')(input_img)
    e = MaxPooling2D((2, 2), padding='same')(e)
    e = Conv2D(32, (5, 5), activation='relu', padding='same')(e)
    #e = MaxPooling2D((2, 2), padding='same')(e)
    e = Conv2D(16, (1, 1), activation='relu', padding='same')(e)
    e = Conv2D(8, (3, 3), activation='relu', padding='same')(e)
    encoded = Conv2D(4, (3, 3), activation='relu', padding='same')(e)

    d = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
    d = Conv2D(8, (3, 3), activation='relu', padding='same')(d)
    d = Conv2D(16, (1, 1), activation='relu', padding='same')(d)
    #d = UpSampling2D((2, 2))(d)
    d = Conv2D(32, (5, 5), activation='relu', padding='same')(d)
    d = UpSampling2D((2, 2))(d)
    d = Conv2D(64, (7, 7), activation='relu', padding='same')(d)
    decoded = Conv2D(3, (1, 1), activation='linear')(d)

    autoencoder = Model(input_img, decoded)
    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(train_data, train_data, epochs=1, batch_size=25)

    return autoencoder