import torch.nn as nn
import torch.nn.functional as F


class ModelOne(nn.Module):
    def __init__(self, num_classes):
        super(ModelOne, self).__init__()

        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)                  # 2x2 max pooling
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)    # 3 channels, 64 filters, 3x3 kernel
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)  # 64 channels, 128 filters, 3x3 kernel
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 62 * 62, 120)        # 128 * 61 * 61 input features, 120 output features
        self.fc2 = nn.Linear(120, 84)                   # 120 input features, 84 output features
        self.fc3 = nn.Linear(84, self.num_classes)      # 84 input features, num_classes output features

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))            # Convolution 1, ReLU activation, max pooling
        x = self.pool(F.relu(self.conv2(x)))            # Convolution 2, ReLU activation, max pooling
        x = x.view(-1, 128 * 62 * 62)                   # Flatten the output of the previous layer
        x = F.relu(self.fc1(x))                         # Fully connected layer, ReLU activation
        x = F.relu(self.fc2(x))                         # Fully connected layer, ReLU activation
        x = self.fc3(x)                                 # Fully connected layer

        return x                                        # Return the output
