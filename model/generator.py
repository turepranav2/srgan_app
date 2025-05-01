import tensorflow as tf
from tensorflow.keras.layers import Conv2D, PReLU, BatchNormalization, Add, UpSampling2D, Input
from tensorflow.keras.models import Model

def residual_block(x_in):
    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x_in)
    x = BatchNormalization(momentum=0.8)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    return Add()([x_in, x])

def build_generator():
    input_layer = Input(shape=(96, 96, 3))
    x_start = Conv2D(64, kernel_size=9, strides=1, padding='same')(input_layer)
    x_start = PReLU(shared_axes=[1, 2])(x_start)

    r = residual_block(x_start)
    for _ in range(15):
        r = residual_block(r)

    x_mid = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
    x_mid = BatchNormalization(momentum=0.8)(x_mid)
    x = Add()([x_start, x_mid])

    x = UpSampling2D(size=2)(x)
    x = Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
    x = PReLU(shared_axes=[1, 2])(x)

    x = UpSampling2D(size=2)(x)
    x = Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
    x = PReLU(shared_axes=[1, 2])(x)

    output = Conv2D(3, kernel_size=9, strides=1, padding='same', activation='tanh')(x)

    return Model(inputs=input_layer, outputs=output)