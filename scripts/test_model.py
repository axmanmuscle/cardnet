## want a quick script to test whether the model is working on clean images before we try to put it on a webcam
from PIL import Image
from torchvision import transforms
from cardnet import build_resnet18
from cardnet.utils import load_class_names
import torch

def main():
  device = torch.device("cpu")
  model_path = 'models/resnet_front.pth'
  class_names_path = 'models/class_names.txt'

  class_names = load_class_names(class_names_path)
  num_classes = len(class_names)
  model = build_resnet18(num_classes=num_classes, freeze_backbone=False)
  model.load_state_dict(torch.load(model_path, map_location='cpu'))
  model.eval()

  img = Image.open("data/test_images/7_h/7_h_20250518_211347.jpg")
  val_transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
  ])  # same val transform as training
  input_tensor = val_transform(img).unsqueeze(0).to(device)
  output = model(input_tensor)
  pred = output.argmax(dim=1).item()
  print(class_names[pred])

if __name__ == "__main__":
  main()