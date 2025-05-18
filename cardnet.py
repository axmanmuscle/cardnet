import torch.nn as nn
import torch.nn.functional as F

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