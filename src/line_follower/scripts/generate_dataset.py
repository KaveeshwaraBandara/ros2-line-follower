import os
import sys
import csv

import numpy as np
import cv2

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPTS_DIR)

from angle_detector import find_line_angle

_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPTS_DIR, '..', '..', '..'))

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
NUM_SAMPLES = 3000
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, 'dataset')
IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images')
LABELS_CSV = os.path.join(OUTPUT_DIR, 'labels.csv')


def add_realism(image, rng):
    brightness = rng.uniform(0.6, 1.0)
    image = (image * brightness).astype(np.uint8)

    noise = rng.normal(0, 15, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if rng.random() > 0.5:
        ksize = rng.choice([3, 5])
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)

    return image


def generate_frame(line_x, angle_deg, rng):
    bg_gray = rng.integers(200, 256)
    image = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), bg_gray, dtype=np.uint8)

    texture = rng.normal(0, 8, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + texture, 0, 255).astype(np.uint8)

    line_width = rng.integers(30, 55)
    line_darkness = int(rng.integers(0, 60))

    angle_rad = np.deg2rad(angle_deg)

    x_bottom = int(line_x)
    y_bottom = IMAGE_HEIGHT - 1

    dx = int(np.tan(angle_rad) * IMAGE_HEIGHT)
    x_top = x_bottom - dx
    y_top = 0

    cv2.line(
        image,
        (x_bottom, y_bottom),
        (x_top, y_top),
        (line_darkness, line_darkness, line_darkness),
        thickness=int(line_width)
    )

    image = add_realism(image, rng)
    return image


def main():
    rng = np.random.default_rng(42)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    kept, skipped = 0, 0
    with open(LABELS_CSV, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['filename', 'angle_deg'])

        for i in range(NUM_SAMPLES):
            # randomize both the line's bottom position and its tilt so the
            # regression target (look-ahead steering angle) covers the full range
            line_x = rng.uniform(IMAGE_WIDTH * 0.15, IMAGE_WIDTH * 0.85)
            draw_angle = rng.uniform(-35, 35)

            image = generate_frame(line_x, draw_angle, rng)
            filename = f'frame_{i:04d}.png'
            path = os.path.join(IMAGES_DIR, filename)
            cv2.imwrite(path, image)

            # Label with the classical look-ahead detector so the target is the
            # steering angle toward a point ahead on the line (accounts for both
            # lateral offset and tilt), not just the raw drawing angle.
            result = find_line_angle(path)
            if result['found']:
                writer.writerow([filename, f"{result['angle_deg']:.4f}"])
                kept += 1
            else:
                os.remove(path)
                skipped += 1

            if (i + 1) % 500 == 0:
                print(f'...generated {i + 1}/{NUM_SAMPLES}  (kept {kept}, skipped {skipped})')

    print(f'Dataset complete: {kept} labeled images in {IMAGES_DIR}')
    print(f'Labels written to {LABELS_CSV} (skipped {skipped} with no detectable line)')


if __name__ == '__main__':
    main()
