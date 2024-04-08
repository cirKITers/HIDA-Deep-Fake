import torch.nn as nn


class Discriminator(nn.Module):
    """Naive discriminator"""

    def __init__(self, image_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.image_size = image_size
        self.main = nn.Sequential(
            nn.Linear(self.image_size * self.image_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.main(x)
