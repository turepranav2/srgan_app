import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, PReLU, Add, UpSampling2D

def build_generator():
    def residual_block(x_in):
        x = Conv2D(64, 3, padding='same')(x_in)
        x = BatchNormalization(momentum=0.8)(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(64, 3, padding='same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        return Add()([x_in, x])

    input_layer = Input(shape=(96, 96, 3))
    x = Conv2D(64, 9, padding='same')(input_layer)
    x = x1 = PReLU(shared_axes=[1, 2])(x)

    for _ in range(16):
        x = residual_block(x)

    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Add()([x, x1])

    x = UpSampling2D(size=2)(x)
    x = Conv2D(256, 3, padding='same')(x)
    x = PReLU(shared_axes=[1, 2])(x)

    x = UpSampling2D(size=2)(x)
    x = Conv2D(256, 3, padding='same')(x)
    x = PReLU(shared_axes=[1, 2])(x)

    out = Conv2D(3, 9, padding='same', activation='tanh')(x)

    return Model(inputs=input_layer, outputs=out)

model = build_generator()
model.save("generator_weights.h5")
print("✅ Saved generator_weights.h5")
