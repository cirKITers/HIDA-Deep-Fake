import torch.nn as nn

class Discriminator(nn.Module):
    """Naive discriminator"""

    def __init__(self, image_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.image_size = image_size
        self.model = nn.Sequential(
            # Inputs to first hidden layer (num_input_features -> 64)
            nn.Linear(self.image_size * self.image_size, 32),
            nn.ReLU(),
            # First hidden layer (64 -> 16)
            nn.Linear(32, 16),
            nn.ReLU(),
            # Second hidden layer (16 -> output)
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, self.image_size * self.image_size)
        return self.model(x)