import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def process_image(image_path, device):
    weights = EfficientNet_B0_Weights.DEFAULT
    transform = weights.transforms()

    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))

    # tensor for model
    img_tensor = transform(img).unsqueeze(0).to(device)

    # numpy image for visualization
    rgb_img = np.array(img).astype(np.float32) / 255

    return img_tensor, rgb_img


def generate_gradcam(model, image_path, device):

    model.eval()

    # IMPORTANT FOR GRAD-CAM
    for param in model.features.parameters():
        param.requires_grad = True

    input_tensor, rgb_img = process_image(image_path=image_path, device=device)

    target_layers = [model.features[-2]]

    cam = GradCAM(model=model, target_layers=target_layers)

    # generate heatmap
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    # resize heatmap
    grayscale_cam = cv2.resize(grayscale_cam, (rgb_img.shape[1], rgb_img.shape[0]))

    # overlay heatmap
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return visualization
