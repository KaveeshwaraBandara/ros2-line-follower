import torch
import sys
import os

sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), '..', 'scripts'))

from model import LineFollowerCNN


def test_model_output_shape():
    """Model takes a 48x64x3 image and regresses a single angle."""
    model = LineFollowerCNN()
    model.eval()

    dummy_input = torch.randn(1, 3, 48, 64)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1,), \
        f'Expected shape (1,), got {output.shape}'


def test_model_batch():
    """Model handles a batch of images, not just one."""
    model = LineFollowerCNN()
    model.eval()

    batch = torch.randn(8, 3, 48, 64)
    with torch.no_grad():
        output = model(batch)

    assert output.shape == (8,), \
        f'Expected shape (8,), got {output.shape}'


def test_model_trains_one_step():
    """One regression training step runs and changes the weights."""
    model = LineFollowerCNN()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    images = torch.randn(4, 3, 48, 64)
    targets = torch.tensor([0.1, -0.2, 0.3, 0.0])

    weight_before = model.conv1.weight.clone()

    optimizer.zero_grad()
    output = model(images)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    weight_after = model.conv1.weight

    assert not torch.equal(weight_before, weight_after), \
        'Weights did not change after a training step'
