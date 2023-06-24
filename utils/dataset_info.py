from typing import NamedTuple

import torch
from torch.utils.data import DataLoader


class DatasetInfo(NamedTuple):
    model: str
    num_classes: int
    train_loader: DataLoader
    val_loader: DataLoader
    device: torch.device
