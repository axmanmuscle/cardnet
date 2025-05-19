import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def imshow(img_tensor, title=None):
    img = img_tensor.permute(1, 2, 0).numpy()
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


def show_val_samples(model, val_dataset, class_names, device, num_images=4):
    model.eval()
    indices = random.sample(range(len(val_dataset)), num_images)

    with torch.no_grad():
        for idx in indices:
            img, label = val_dataset[idx]
            input_tensor = img.unsqueeze(0).to(device)
            output = model(input_tensor)
            pred = output.argmax(dim=1).item()

            true_class = class_names[label]
            pred_class = class_names[pred]
            title = f"True: {true_class} | Pred: {pred_class}"
            imshow(img, title)

def show_grad_cam(model, val_dataset, class_names, device, num_images=4):
    model.eval()
    target_layer = model.layer4[-1]  # works for ResNet18

    cam = GradCAM(model=model, target_layers=[target_layer])

    indices = random.sample(range(len(val_dataset)), num_images)
    for idx in indices:
        img_tensor, label = val_dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)
        rgb_img = img_tensor.permute(1, 2, 0).numpy()
        rgb_img = np.clip(rgb_img, 0, 1)

        target = [ClassifierOutputTarget(label)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=target)[0]  # [0] = first image

        cam_overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        plt.imshow(cam_overlay)
        plt.title(f"True: {class_names[label]}")
        plt.axis('off')
        plt.show()

def load_class_names(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]