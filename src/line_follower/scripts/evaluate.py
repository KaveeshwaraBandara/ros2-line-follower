import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import LineFollowerCNN

DATASET_DIR = '/workspaces/ros2-line-follower/dataset'
MODEL_PATH = '/workspaces/ros2-line-follower/line_follower_model.pth'
BATCH_SIZE = 32
IMAGE_SIZE = (48, 64)
VALIDATION_SPLIT = 0.2


def main():
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    full_dataset = datasets.ImageFolder(root=DATASET_DIR, transform=transform)
    class_names = full_dataset.classes

    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size

    torch.manual_seed(42)
    _, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = LineFollowerCNN(num_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()


    num_classes = len(class_names)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for true_label, pred_label in zip(labels, predicted):
                confusion[true_label][pred_label] += 1

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total


    print(f'\nOverall accuracy: {accuracy:.2f}% ({correct}/{total})\n')
    print('Confusion matrix (rows = true, cols = predicted):')
    print('             ' + '  '.join(f'{name:>7}' for name in class_names))
    for i, name in enumerate(class_names):
        row = '  '.join(f'{confusion[i][j].item():>7}' for j in range(num_classes))
        print(f'{name:>10}   {row}')

    print('\nPer-class accuracy:')
    for i, name in enumerate(class_names):
        class_total = confusion[i].sum().item()
        class_correct = confusion[i][i].item()
        class_acc = 100 * class_correct / class_total if class_total > 0 else 0
        print(f'  {name:>7}: {class_acc:.2f}% ({class_correct}/{class_total})')


if __name__ == '__main__':
    main()