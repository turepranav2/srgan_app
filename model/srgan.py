from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from model.generator import build_generator
from model.discriminator import build_discriminator
import tensorflow as tf

def build_srgan():
    generator = build_generator()
    discriminator = build_discriminator()
    discriminator.trainable = False

    img_input = Input(shape=(96, 96, 3))
    generated_img = generator(img_input)
    validity = discriminator(generated_img)

    combined = Model(img_input, [validity, generated_img])
    combined.compile(loss=['binary_crossentropy', 'mse'],
                     loss_weights=[1e-3, 1],
                     optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

    return generator, discriminator, combined