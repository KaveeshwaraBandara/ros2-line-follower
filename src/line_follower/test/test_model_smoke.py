import torch
import sys
import os

sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), '..', 'scripts'))

from model import LineFollowerCNN


def test_model_output_shape():
    """Model takes a 48x64x3 image and returns 3 class scores."""
    model = LineFollowerCNN(num_classes=3)
    model.eval()

    dummy_input = torch.randn(1, 3, 48, 64)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, 3), \
        f'Expected shape (1, 3), got {output.shape}'


def test_model_batch():
    """Model handles a batch of images, not just one."""
    model = LineFollowerCNN(num_classes=3)
    model.eval()

    batch = torch.randn(8, 3, 48, 64)
    with torch.no_grad():
        output = model(batch)

    assert output.shape == (8, 3), \
        f'Expected shape (8, 3), got {output.shape}'


def test_model_trains_one_step():
    """One training step runs and changes the weights."""
    model = LineFollowerCNN(num_classes=3)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    images = torch.randn(4, 3, 48, 64)
    labels = torch.tensor([0, 1, 2, 0])

    weight_before = model.conv1.weight.clone()

    optimizer.zero_grad()
    output = model(images)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    weight_after = model.conv1.weight

    assert not torch.equal(weight_before, weight_after), \
        'Weights did not change after a training step'