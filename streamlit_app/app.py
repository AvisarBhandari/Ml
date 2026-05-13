from inspect import getmodule

import streamlit as st
from Grad_CAM_Visualization import generate_gradcam
from class_names import class_names
from model import get_model
from predict import predict_image
import torch
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="FoodVision", page_icon="🥟", layout="wide")

st.title("FoodVision")

st.markdown(
    "Upload a food image (chicken_curry, donuts, dumplings, fried_rice) and the model will predict the class."
)


col1, col2 = st.columns(2)
allowed_types = ["png", "jpg", "jpeg"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def model_pred_result(image, model: torch.nn.Module, device):
    """
    show the pred result
    """
    model_pred = predict_image(image, get_model().to(device), device)
    df = pd.DataFrame(model_pred)
    # st.write(model_pred)
    # st.write(model_pred["pred"])
    for i in range(len(model_pred["name"])):
        st.write(f"{model_pred['name'][i]} - {model_pred['pred'][i]}%")
    st.bar_chart(df, x="name", y="pred", sort=True)
    cam_image = generate_gradcam(
        image_path=image, model=model.to(device), device=device
    )

    st.image(cam_image)

    return df


with col1:
    upload_image = st.file_uploader("Upload Image", type=allowed_types)

    if upload_image is not None:
        st.write(Image.open(upload_image))
        with col2:
            with st.spinner("Predicting...", show_time=True):
                model_pred = model_pred_result(
                    image=upload_image, model=get_model().to(device), device=device
                )
    else:
        example_dir = Path("streamlit_app/images.jpg")
        if example_dir.is_file():
            st.write("Example Image:")
            st.write(Image.open(example_dir))
            with col2:
                model_pred = model_pred_result(
                    image=example_dir, model=get_model().to(device), device=device
                )


st.sidebar.title("About")

st.sidebar.write(
    "EfficientNet-B0 trained on Food101 subset (chicken_curry, donuts, dumplings, fried_rice)."
)
