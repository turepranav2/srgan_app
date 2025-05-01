import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load generator model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("generator_weights.h5", compile=False)
    return model

generator = load_model()

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((96, 96))
    img_array = np.array(image).astype(np.float32)
    img_array = img_array / 127.5 - 1.0
    return np.expand_dims(img_array, axis=0), image

def postprocess_image(img):
    img = 0.5 * img + 0.5
    img = np.clip(img, 0, 1)
    return Image.fromarray((img[0] * 255).astype(np.uint8))

st.title("SRGAN Super Resolution")

uploaded_file = st.file_uploader("Upload a low-resolution image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    lr_img, original = preprocess_image(uploaded_file)
    sr_img = generator.predict(lr_img)
    sr_image = postprocess_image(sr_img)

    st.subheader("Original Image")
    st.image(original, use_column_width=True)
    st.subheader("Super-Resolved Image")
    st.image(sr_image, use_column_width=True)