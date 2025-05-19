import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

def build_resnet18(num_classes, freeze_backbone=True):
    model = resnet18(pretrained=True)

    # for param in model.parameters():
    #     param.requires_grad = True  # or freeze some layers if needed

    # Optionally freeze most layers
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model

class CardCNN(nn.Module):
    def __init__(self, num_classes):
        super(CardCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)
        # self.fc1 = nn.Linear(128 * 28 * 28, 256)
        # self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32 x 112 x 112
        x = self.pool(F.relu(self.conv2(x)))  # 64 x 56 x 56
        x = self.pool(F.relu(self.conv3(x)))  # 128 x 28 x 28
        x = x.view(-1, 64 * 28 * 28)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class DeeperCardCNN(nn.Module):
    def __init__(self, num_classes):
        super(DeeperCardCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 x 112 x 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 x 56 x 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128 x 28 x 28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 256 x 14 x 14
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # 256 * 14 * 14 = 50176
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # slightly lower dropout
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x