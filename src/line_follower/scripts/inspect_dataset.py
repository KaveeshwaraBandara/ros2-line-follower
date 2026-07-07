import os
import cv2
import numpy as np

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPTS_DIR, '..', '..', '..'))

DATASET = os.path.join(_PROJECT_ROOT, 'dataset')
classes = ['left', 'center', 'right']

rows = []
for class_name in classes:
    class_dir = os.path.join(DATASET, class_name)
    files = sorted(os.listdir(class_dir))[:5]
    images = []
    for f in files:
        img = cv2.imread(os.path.join(class_dir, f))
        img = cv2.resize(img, (160, 120))
        images.append(img)
    row = np.hstack(images)
    rows.append(row)

montage = np.vstack(rows)
cv2.imwrite(os.path.join(_PROJECT_ROOT, 'dataset_preview.png'), montage)
print('Saved preview: 3 rows (left, center, right), 5 samples each')