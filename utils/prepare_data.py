import json

import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from models.model_one import ModelOne
from models.model_two import ModelTwo
from utils.dataset_info import DatasetInfo
from utils.test_args import TestArgs


def prepare_train_data(selected_model: str):
    # Preprocess the data
    data_transform = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.ToTensor(),
    ])

    # Load the training & validation data
    train_data = datasets.ImageFolder(root='data/train', transform=data_transform)
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

    val_data = datasets.ImageFolder(root='data/val', transform=data_transform)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=True)

    num_classes = __calculate_num_classes(train_data.class_to_idx)

    # Get the device : GPU or CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Return the DatasetInfo object
    return DatasetInfo(selected_model, num_classes, train_loader, val_loader, device)


def prepare_test_args(selected_model: str):
    # Get the device : GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the number of classes from the saved index to class mapping
    with open('class_to_label.json', 'r') as f:
        class_to_label = json.load(f)

    # Number of classes is the length of the dictionary
    num_classes = len(class_to_label)

    # Load the haar cascades for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    return TestArgs(selected_model, num_classes, class_to_label, device, face_cascade)


def define_model(selected_model: str, num_classes: int):
    models = {
        'model_one': ModelOne(num_classes),
        'model_two': ModelTwo(num_classes)
    }

    if selected_model in models:
        return models[selected_model]
    else:
        raise ValueError(f'Invalid model name, choose from {list(models.keys())}')


def __calculate_num_classes(class_to_idx: dict):
    # Invert the dictionary to get index to class mapping
    idx_to_class = {idx: class_ for class_, idx in class_to_idx.items()}

    # Save the mapping to a file
    with open('class_to_label.json', 'w') as f:
        json.dump(idx_to_class, f)

    return len(class_to_idx)