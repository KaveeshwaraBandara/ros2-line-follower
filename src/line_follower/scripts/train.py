import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPTS_DIR, '..', '..', '..'))
sys.path.insert(0, _SCRIPTS_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import LineFollowerCNN

DATASET_DIR = os.path.join(_PROJECT_ROOT, 'dataset')
MODEL_OUTPUT = os.path.join(_PROJECT_ROOT, 'line_follower_model.pth')
BATCH_SIZE = 32
EPOCHS = 12
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
IMAGE_SIZE = (48, 64)

def main():
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    full_dataset = datasets.ImageFolder(root=DATASET_DIR, transform=transform)
    print(f'Classes found: {full_dataset.classes}')
    print(f'Total images: {len(full_dataset)}')

    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )
    print(f'Training: {train_size}, Validation: {val_size}')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on: {device}')

    model = LineFollowerCNN(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)


        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f'Epoch {epoch+1}/{EPOCHS} | '
              f'Train loss: {train_loss:.4f} | '
              f'Train acc: {train_acc:.2f}% | '
              f'Val acc: {val_acc:.2f}%')

    torch.save(model.state_dict(), MODEL_OUTPUT)
    print(f'Model saved to {MODEL_OUTPUT}')


if __name__ == '__main__':
    main()