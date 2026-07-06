import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPTS_DIR, '..', '..', '..'))
sys.path.insert(0, _SCRIPTS_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import RoadAngleDataset, ANGLE_SCALE
from model import LineFollowerCNN

DATASET_DIR = os.path.join(_PROJECT_ROOT, 'dataset')
IMAGES_DIR = os.path.join(DATASET_DIR, 'images')
LABELS_CSV = os.path.join(DATASET_DIR, 'labels.csv')
MODEL_OUTPUT = os.path.join(_PROJECT_ROOT, 'line_follower_model.pth')
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2


def main():
    full_dataset = RoadAngleDataset(IMAGES_DIR, LABELS_CSV, augment=True)
    print(f'Total labeled images: {len(full_dataset)}')

    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f'Training: {train_size}, Validation: {val_size}')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on: {device}')

    model = LineFollowerCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4)

    best_val_mae = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        n_seen = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            n_seen += images.size(0)

        train_loss = running_loss / n_seen

        model.eval()
        val_loss_sum, val_abs_sum, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                val_loss_sum += criterion(outputs, targets).item() * images.size(0)
                val_abs_sum += (outputs - targets).abs().sum().item() * ANGLE_SCALE
                n_val += images.size(0)

        val_loss = val_loss_sum / n_val
        val_mae_deg = val_abs_sum / n_val
        val_rmse_deg = (val_loss ** 0.5) * ANGLE_SCALE
        scheduler.step(val_loss)

        print(f'Epoch {epoch+1}/{EPOCHS} | '
              f'Train MSE: {train_loss:.5f} | '
              f'Val MSE: {val_loss:.5f} | '
              f'Val MAE: {val_mae_deg:.2f} deg | '
              f'Val RMSE: {val_rmse_deg:.2f} deg')

        if val_mae_deg < best_val_mae:
            best_val_mae = val_mae_deg
            torch.save(model.state_dict(), MODEL_OUTPUT)
            print(f'  -> new best (Val MAE {best_val_mae:.2f} deg), saved to {MODEL_OUTPUT}')

    print(f'Training done. Best Val MAE: {best_val_mae:.2f} deg. Model at {MODEL_OUTPUT}')


if __name__ == '__main__':
    main()
