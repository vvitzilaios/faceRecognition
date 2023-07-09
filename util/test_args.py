from typing import NamedTuple

import cv2
import torch


class TestArgs(NamedTuple):
    selected_model: str
    num_classes: int
    class_to_label: dict
    device: torch.device
    face_cascade: cv2.CascadeClassifier

