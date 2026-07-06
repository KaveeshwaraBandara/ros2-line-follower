import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPTS_DIR, '..', '..', '..'))
sys.path.insert(0, _SCRIPTS_DIR)

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from dataset import RoadAngleDataset, ANGLE_SCALE
from model import LineFollowerCNN

DATASET_DIR = os.path.join(_PROJECT_ROOT, 'dataset')
IMAGES_DIR = os.path.join(DATASET_DIR, 'images')
LABELS_CSV = os.path.join(DATASET_DIR, 'labels.csv')
MODEL_PATH = os.path.join(_PROJECT_ROOT, 'line_follower_model.pth')
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2


def main():
    # Reproduce the same train/val split used in train.py so we evaluate on
    # held-out samples (augment=False for a clean, deterministic read).
    full_dataset = RoadAngleDataset(IMAGES_DIR, LABELS_CSV, augment=False)

    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    torch.manual_seed(42)
    _, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = LineFollowerCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
    model.eval()

    preds, truths = [], []
    with torch.no_grad():
        for images, targets in val_loader:
            outputs = model(images)
            preds.append(outputs.numpy() * ANGLE_SCALE)
            truths.append(targets.numpy() * ANGLE_SCALE)

    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    errors = preds - truths

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    max_err = np.max(np.abs(errors))

    print(f'\nEvaluated {len(preds)} validation images')
    print(f'MAE  = {mae:.2f} deg')
    print(f'RMSE = {rmse:.2f} deg')
    print(f'Max abs error = {max_err:.2f} deg')

    worst_idx = np.argsort(-np.abs(errors))[:10]
    print('\nWorst 10 predictions:')
    print(f"{'idx':>6} {'true':>8} {'pred':>8} {'err':>8}")
    for i in worst_idx:
        print(f'{i:6d} {truths[i]:8.2f} {preds[i]:8.2f} {errors[i]:8.2f}')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        plt.scatter(truths, preds, s=6, alpha=0.4)
        lims = [min(truths.min(), preds.min()), max(truths.max(), preds.max())]
        plt.plot(lims, lims, 'r--', linewidth=1)
        plt.xlabel('true angle (deg)')
        plt.ylabel('predicted angle (deg)')
        plt.title(f'MAE={mae:.2f} deg  RMSE={rmse:.2f} deg')
        plt.tight_layout()
        out = os.path.join(_PROJECT_ROOT, 'eval_scatter.png')
        plt.savefig(out, dpi=120)
        print(f'\nSaved {out}')
    except ImportError:
        pass


if __name__ == '__main__':
    main()
