import os
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from cardnet import CardCNN, DeeperCardCNN, build_resnet18
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from cardnet.utils import show_val_samples, show_grad_cam

from collections import Counter
from torch.utils.data import Subset

class CardDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.png')])
        self.labels = [f.split('.')[0] for f in self.image_files]  # "k_s.png" → "k_s"
        self.classes = sorted(list(set(self.labels)))              # unique class names
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        label_name = self.labels[idx]
        label = self.class_to_idx[label_name]

        if self.transform:
            image = self.transform(image)
        return image, label

def make_data(data_dir, train_frac = 0.8):
  # Augmentations for training
  train_transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.RandomRotation(15),
      transforms.ColorJitter(brightness=0.2, contrast=0.2),
      transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
  ])

#   train_transform = transforms.Compose([
#     transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),  # zoom out more
#     transforms.RandomRotation(30),
#     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
#     transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
# ])
  # Simpler transforms for validation
  val_transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
  ])

  # Load full dataset
  # full_dataset = CardDataset(root_dir=data_dir, transform=None)
  full_dataset = ImageFolder(root=data_dir, transform=None)
  print(full_dataset.class_to_idx)
  class_names = full_dataset.classes
  num_classes = len(class_names)
  
  # Print dataset statistics
  print(f"Total images: {len(full_dataset)}")
  print(f"Number of unique classes: {num_classes}")
  
  # Count examples per class
  label_counts = Counter([full_dataset.targets[i] for i in range(len(full_dataset))])
  label_map = {i: class_names[i] for i in range(num_classes)}
  print("Class label counts:")
  for idx, count in label_counts.items():
    print(f"  {label_map[idx]}: {count}")

  # Create train/val indices manually
  dataset_size = len(full_dataset)
  indices = torch.randperm(dataset_size).tolist()
  train_size = int(train_frac * dataset_size)
  train_indices, val_indices = indices[:train_size], indices[train_size:]

  # Create two datasets with different transforms
  train_dataset = ImageFolder(root=data_dir, transform=train_transform)
  val_dataset = ImageFolder(root=data_dir, transform=val_transform)

  # Subset them
  train_data = Subset(train_dataset, train_indices)
  val_data = Subset(val_dataset, val_indices)

  # DataLoaders
  train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
  val_loader = DataLoader(val_data, batch_size=32)

  return train_loader, val_loader, num_classes, class_names, val_data

def train_model(model, train_loader, val_loader, val_data, device, num_epochs=10, lr=1e-3, class_names=None):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    for epoch in range(num_epochs):
        # --- TRAINING ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)                 # logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = 100 * train_correct / train_total
        train_loss /= train_total

        # --- VALIDATION ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())              

        # if epoch % 25 == 0 or epoch == num_epochs - 1:
        #   print("Visualizing predictions on val set:")
        #   show_val_samples(model, val_data, class_names, device, num_images=4)

        #   print("🔥 Grad-CAM Visualizations:")
        #   show_grad_cam(model, val_data, class_names, device, num_images=4)

        # if epoch == num_epochs - 1:
        #   cm = confusion_matrix(all_labels, all_preds)
        #   sns.heatmap(cm, annot=True, xticklabels=class_names, yticklabels=class_names)
        #   plt.xlabel("Predicted")
        #   plt.ylabel("True")
        #   plt.title("Confusion Matrix")
        #   plt.show()

        val_acc = 100 * val_correct / val_total
        val_loss /= val_total

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%\n")


    # Save model weights
    model_path = "models/resnet_front.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")

    class_name_path = "models/class_names.txt"
    with open(class_name_path, "w") as f:
      for cls in class_names:
        f.write(cls + "\n")

    print(f"✅ Class names saved to {class_name_path}")


def main():
  # data_dir = "data/imgs_front/color"
  data_dir = "data/front_images"
  train_frac = 0.8

  train_loader, val_loader, num_classes, class_names, val_data = make_data(data_dir, train_frac)
  # num_classes = len(full_dataset.classes)  # automatically inferred
  # model = CardCNN(num_classes=num_classes)
  # model = DeeperCardCNN(num_classes=num_classes)
  model = build_resnet18(num_classes=num_classes, freeze_backbone=True)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  train_model(model, train_loader, val_loader, val_data, device, num_epochs = 75, class_names = class_names)
  

if __name__ == "__main__":
  main()