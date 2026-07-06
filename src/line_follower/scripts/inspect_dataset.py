import os
import csv

import cv2
import numpy as np

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPTS_DIR, '..', '..', '..'))

DATASET = os.path.join(_PROJECT_ROOT, 'dataset')
IMAGES_DIR = os.path.join(DATASET, 'images')
LABELS_CSV = os.path.join(DATASET, 'labels.csv')


def main():
    samples = []
    with open(LABELS_CSV, newline='') as fh:
        for row in csv.DictReader(fh):
            samples.append((row['filename'], float(row['angle_deg'])))

    # sort by angle and pick evenly spaced samples across the range
    samples.sort(key=lambda s: s[1])
    n_show = min(6, len(samples))
    picks = [samples[int(i * (len(samples) - 1) / (n_show - 1))] for i in range(n_show)]

    tiles = []
    for fname, angle in picks:
        img = cv2.imread(os.path.join(IMAGES_DIR, fname))
        img = cv2.resize(img, (160, 120))
        cv2.putText(img, f'{angle:+.1f}', (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        tiles.append(img)

    montage = np.hstack(tiles)
    out = os.path.join(_PROJECT_ROOT, 'dataset_preview.png')
    cv2.imwrite(out, montage)
    print(f'Saved preview across angle range ({picks[0][1]:+.1f} .. {picks[-1][1]:+.1f} deg): {out}')


if __name__ == '__main__':
    main()
