"""
dataset.py

PyTorch Dataset that reads (image, steering_angle) pairs from the image
folder plus labels.csv produced by generate_dataset.py.

Images are loaded as RGB and resized to 64x48 (W x H) to match the CNN
input and the inference pipeline. Angle targets are normalized by
ANGLE_SCALE so MSE loss stays well-behaved.
"""

import csv
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 48
ANGLE_SCALE = 90.0


class RoadAngleDataset(Dataset):
    def __init__(self, images_dir, labels_csv, augment=False):
        self.images_dir = images_dir
        self.augment = augment
        self.samples = []  # list of (filename, angle_deg)

        with open(labels_csv, newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                self.samples.append((row['filename'], float(row['angle_deg'])))

        if not self.samples:
            raise ValueError(f'No labeled samples found in {labels_csv}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, angle = self.samples[idx]
        path = os.path.join(self.images_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise FileNotFoundError(path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != IMAGE_HEIGHT or img.shape[1] != IMAGE_WIDTH:
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

        if self.augment:
            # horizontal flip mirrors left/right, so the angle negates
            if np.random.rand() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1, :])
                angle = -angle
            # brightness jitter
            if np.random.rand() < 0.5:
                factor = np.random.uniform(0.85, 1.15)
                img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        x = torch.from_numpy(img).float() / 255.0  # [H,W,3] in [0,1]
        x = x.permute(2, 0, 1)                      # [3,H,W]
        y = torch.tensor(angle / ANGLE_SCALE, dtype=torch.float32)
        return x, y
