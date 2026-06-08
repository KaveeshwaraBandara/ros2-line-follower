import os
import numpy as np
import cv2

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
SAMPLES_PER_CLASS = 1000
OUTPUT_DIR = '/workspaces/ros2-line-follower/dataset'

LEFT_THRESHOLD = IMAGE_WIDTH * 0.4
RIGHT_THRESHOLD = IMAGE_WIDTH * 0.6


def add_realism(image, rng):
    brightness = rng.uniform(0.6, 1.0)
    image = (image * brightness).astype(np.uint8)

    noise = rng.normal(0, 15, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if rng.random() > 0.5:
        ksize = rng.choice([3, 5])
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)

    return image


def generate_frame(line_x, rng):
    bg_gray = rng.integers(200, 256)
    image = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), bg_gray, dtype=np.uint8)

    texture = rng.normal(0, 8, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + texture, 0, 255).astype(np.uint8)

    line_width = rng.integers(30, 55)
    line_darkness = rng.integers(0, 60)

    x_start = max(0, int(line_x - line_width / 2))
    x_end = min(IMAGE_WIDTH, int(line_x + line_width / 2))
    image[:, x_start:x_end] = line_darkness

    image = add_realism(image, rng)
    return image


def main():
    rng = np.random.default_rng(42)

    classes = {
        'left':   (0, LEFT_THRESHOLD),
        'center': (LEFT_THRESHOLD, RIGHT_THRESHOLD),
        'right':  (RIGHT_THRESHOLD, IMAGE_WIDTH),
    }

    for class_name, (x_min, x_max) in classes.items():
        class_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)

        for i in range(SAMPLES_PER_CLASS):
            line_x = rng.uniform(x_min, x_max)
            image = generate_frame(line_x, rng)
            filename = os.path.join(class_dir, f'frame_{i:04d}.png')
            cv2.imwrite(filename, image)

        print(f'Generated {SAMPLES_PER_CLASS} images for class "{class_name}"')

    print(f'Dataset complete at {OUTPUT_DIR}')


if __name__ == '__main__':
    main()