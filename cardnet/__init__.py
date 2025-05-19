"""
cardnet: Real-time playing card classification from image or webcam input.

Modules:
- model: CNN and pretrained ResNet architectures
- data: Dataset loaders and transforms
- infer: Inference utilities for single image or batch
- utils: Grad-CAM, metrics, and visualization tools
"""

__version__ = "0.1.0"

from .model import CardCNN, DeeperCardCNN, build_resnet18
# from .data import make_data_loaders
# from .infer import load_model_and_predict
from .utils import show_val_samples, show_grad_cam
