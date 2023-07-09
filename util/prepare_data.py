import json

import cv2
import torch
import splitfolders
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from model.model_1 import ModelOne
from model.model_2 import ModelTwo
from model.model_3 import ModelThree
from util.dataset_info import DatasetInfo
from util.logger import logger
from util.test_args import TestArgs


def prepare_train_data(selected_model: str, dataset_path: str):
    logger.info('Preparing training data...')
    __split_data(dataset_path, 'data', ratio=(0.8, 0.2))

    # Preprocess and normalize the data
    transform_train_data = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_val_data = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load the training & validation data
    train_data = datasets.ImageFolder(root='data/train', transform=transform_train_data)
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

    val_data = datasets.ImageFolder(root='data/val', transform=transform_val_data)
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


def define_model(selected_model: int, num_classes: int):
    models = {
        1: ModelOne(num_classes),
        2: ModelTwo(num_classes),
        3: ModelThree(num_classes)
    }

    if selected_model in models:
        return models[selected_model]
    else:
        raise ValueError(f'Invalid model name, choose from {list(models.keys())}')


def __split_data(input_folder, output_folder, ratio):
    logger.info(f'Splitting data into train and validation sets using \'{input_folder}\' as the source directory......')
    splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=ratio, group_prefix=None)


def __calculate_num_classes(class_to_idx: dict):
    # Invert the dictionary to get index to class mapping
    idx_to_class = {idx: class_ for class_, idx in class_to_idx.items()}

    # Save the mapping to a file
    with open('data/class_to_label.json', 'w') as f:
        json.dump(idx_to_class, f, indent=4)

    return len(class_to_idx)
