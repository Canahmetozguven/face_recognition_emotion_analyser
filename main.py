import streamlit as st
import fastai
from fastai.vision.all import *
import pathlib

posix_backup = pathlib.PosixPath
try:
    pathlib.PosixPath = pathlib.WindowsPath
    learn_inf = load_learner("model.pkl")
finally:
    pathlib.PosixPath = posix_backup


def is_cat(x): return x[0].isupper()

st.title('Happy or sad?')


def classify(img):
    pred, pred_idx, probs = learn_inf.predict(img)
    return pred


uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    img = PILImage.create(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = classify(img)
    st.write(f"This is a {label} person!")
