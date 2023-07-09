from torch import nn
import torch.nn.functional as F


class ModelThree(nn.Module):
    def __init__(self, num_classes):
        super(ModelThree, self).__init__()

        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(256)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 29 * 29, 120)
        self.bn4 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn5 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, self.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 256 * 29 * 29)
        x = self.dropout(F.relu(self.bn4(self.fc1(x))))
        x = self.dropout(F.relu(self.bn5(self.fc2(x))))
        x = self.fc3(x)

        return x
