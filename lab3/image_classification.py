import io
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.applications import EfficientNetB0
from keras.utils import load_img, img_to_array
from keras.applications.efficientnet import decode_predictions


@st.cache_resource
def load_model():
    return EfficientNetB0(weights='imagenet')


def preprocess_image(img):
    img = img.resize((224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x


def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для распознавания')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def print_predictions(preds):
    classes = decode_predictions(preds, top=3)[0]
    for cl in classes:
        st.write(f"{cl[1]} — {cl[2]:.4f}")


model = load_model()

st.title('Новая улучшенная классификация изображений в облаке Streamlit')

img = load_image()

if st.button('Распознать изображение') and img is not None:
    x = preprocess_image(img)
    preds = model.predict(x)
    st.write('**Результаты распознавания:**')
    print_predictions(preds)
