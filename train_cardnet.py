import os
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
from cardnet import CardCNN
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

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

  # Simpler transforms for validation
  val_transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
  ])

  # Load full dataset
  full_dataset = CardDataset(root_dir=data_dir, transform=None)
  num_classes = len(full_dataset.classes)
  # Print dataset statistics
  print(f"Total images: {len(full_dataset)}")
  print(f"Number of unique classes: {len(full_dataset.classes)}")
  print("Class label counts:", Counter(full_dataset.labels))

  # Create train/val indices manually
  dataset_size = len(full_dataset)
  indices = torch.randperm(dataset_size).tolist()
  train_size = int(train_frac * dataset_size)
  train_indices, val_indices = indices[:train_size], indices[train_size:]

  # Build two *independent* dataset objects
  train_dataset = CardDataset(root_dir=data_dir, transform=train_transform)
  val_dataset = CardDataset(root_dir=data_dir, transform=val_transform)

  # Use Subset to apply the splits
  train_data = Subset(train_dataset, train_indices)
  val_data = Subset(val_dataset, val_indices)

  # DataLoaders
  train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
  val_loader = DataLoader(val_data, batch_size=32)


  # # Automatic split (e.g. 80% train, 20% val)
  # train_data = torch.utils.data.Subset(CardDataset(root_dir, transform=train_transform), train_indices)
  # val_data = torch.utils.data.Subset(CardDataset(root_dir, transform=val_transform), val_indices)

  # train_size = int(train_frac * len(full_dataset))
  # val_size = len(full_dataset) - train_size
  # train_data, val_data = random_split(full_dataset, [train_size, val_size])

  # # Assign transforms to each split
  # train_data.dataset.transform = train_transform
  # val_data.dataset.transform = val_transform

  # # DataLoaders
  # train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
  # val_loader = DataLoader(val_data, batch_size=32)

  return train_loader, val_loader, num_classes

def train_model(model, train_loader, val_loader, device, num_epochs=10, lr=1e-3):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = 100 * train_correct / train_total
        train_loss /= train_total

        # --- VALIDATION ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        val_loss /= val_total

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%\n")


def main():
  data_dir = "data/imgs_front/color"
  train_frac = 0.8

  train_loader, val_loader, num_classes = make_data(data_dir, train_frac)
  # num_classes = len(full_dataset.classes)  # automatically inferred
  model = CardCNN(num_classes=num_classes)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  train_model(model, train_loader, val_loader, device)
  



if __name__ == "__main__":
  main()