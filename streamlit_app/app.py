import streamlit as st
from class_names import class_names
from model import get_model
from predict import predict_image
import torch
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
st.set_page_config(
    page_title="FoodVision",
    page_icon="🥟",
    layout="wide"
)
st.title("FoodVision")

st.markdown(
    "Upload a food image and the model will predict the class."
)
col1, col2 = st.columns(2)
allowed_types = ["png", "jpg", "jpeg"]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# st.write("Hello World")
# st.write(device)
# st.write(get_model())
with col1:
    upload_image = st.file_uploader("Upload Image",type=allowed_types)

    if upload_image is not None:
        # st.write("Uploaded....")
        # st.write(Image.open(upload_image))
        with col2:
            with st.spinner("Predicting..."):
                model_pred = predict_image(upload_image,get_model().to(device),device)
                # st.write(model_pred)
                # st.write(model_pred["pred"])
                for i in range(len(model_pred["name"])):
                    st.write(f"{model_pred['name'][i]} - {model_pred['pred'][i]}%")
                df = pd.DataFrame(model_pred)
                st.bar_chart(df, x="name", y="pred",sort=True)
                
st.sidebar.title("About")

st.sidebar.write(
    "EfficientNet-B0 trained on Food101 subset (chicken_curry, donuts, dumplings, fried_rice)."
)
## fix for "FileNotFoundError: [Errno 2] No such file or directory: 'best_model_0.pth'"
# import os

# filename = 'streamlit_app/best_model_0.pth'

# st.write(f"Current Working Directory: {os.getcwd()}")
# st.write(f"Does the file exist here? {os.path.exists(filename)}")

# # List all files in the current folder to check for typos
# st.write("Files in this directory:", os.listdir('.'))