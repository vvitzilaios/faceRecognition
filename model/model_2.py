import torch.nn as nn
import torch.nn.functional as F


class ModelTwo(nn.Module):
    def __init__(self, num_classes):
        super(ModelTwo, self).__init__()

        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 30 * 30, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 30 * 30)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
