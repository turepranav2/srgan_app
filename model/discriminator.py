import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Dense, Flatten, Input
from tensorflow.keras.models import Model

def build_discriminator():
    def d_block(x, filters, strides=1, bn=True):
        x = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(x)
        if bn:
            x = BatchNormalization(momentum=0.8)(x)
        return LeakyReLU(alpha=0.2)(x)

    input_layer = Input(shape=(384, 384, 3))
    x = d_block(input_layer, 64, bn=False)
    x = d_block(x, 64, strides=2)
    x = d_block(x, 128)
    x = d_block(x, 128, strides=2)
    x = d_block(x, 256)
    x = d_block(x, 256, strides=2)
    x = d_block(x, 512)
    x = d_block(x, 512, strides=2)

    x = Flatten()(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    output = Dense(1, activation='sigmoid')(x)

    return Model(inputs=input_layer, outputs=output)