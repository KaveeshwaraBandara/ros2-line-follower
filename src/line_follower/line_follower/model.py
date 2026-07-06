import torch
import torch.nn as nn
import torch.nn.functional as F


class LineFollowerCNN(nn.Module):
    """CNN that regresses a single continuous steering angle.

    The output is a normalized angle in ~[-1, 1]; multiply by ANGLE_SCALE
    (see ``angle_scale``) to recover degrees (0 = straight, + = right,
    - = left). Training targets are normalized the same way so MSE loss
    stays well-behaved.
    """

    ANGLE_SCALE = 90.0

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        """        Input:        3 × 48 × 64
        conv1 + pool: 16 × 24 × 32   (channels 3→16, size halved by pool)
        conv2 + pool: 32 × 12 × 16   (channels 16→32, size halved again)
        conv3 + pool: 64 × 6 × 8     (channels 32→64, size halved again)
        """

        self.fc1 = nn.Linear(64 * 6 * 8, 128)
        self.fc2 = nn.Linear(128, 1)

        # kept for backward-compatible attribute access
        self.angle_scale = self.ANGLE_SCALE

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x.squeeze(-1)  # normalized angle; * angle_scale for degrees


if __name__ == '__main__':
    model = LineFollowerCNN()
    dummy_input = torch.randn(4, 3, 48, 64)
    output = model(dummy_input)
    print(f'Input shape:  {dummy_input.shape}')
    print(f'Output shape: {output.shape}')
    print(f'Output values: {output}')

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total trainable parameters: {total_params:,}')
