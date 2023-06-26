from torch import nn
import torch.nn.functional as F


class ModelThree(nn.Module):
    def __init__(self, num_classes):
        super(ModelThree, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, stride=2, padding=3)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=7, stride=2, padding=3)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=7, stride=2, padding=3)

        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer
        self.fc1 = nn.Linear(in_features=512 * 1 * 1, out_features=num_classes)

    def forward(self, x):
        # Convolutional layers with ReLU and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # Flattening the output of the convolutional layers
        x = x.view(-1, 512 * 1 * 1)

        # Fully connected layer
        x = self.fc1(x)

        return x
