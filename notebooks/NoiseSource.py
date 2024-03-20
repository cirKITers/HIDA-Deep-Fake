import torch
import torch.nn as nn
from typing import Tuple

class NoiseSource(nn.Module):
    """Noise Source sampling from normal distribution with zero mean and unit variance weighted and shifted accordingly"""

    def __init__(
        self,
        seed: int,
        noise_gain: torch.Tensor,
        noise_offset: torch.Tensor,
        **kwargs,
    ) -> None:
        """Noise Source sampling from normal distribution with zero mean and unit variance weighted and shifted accordingly.

        Args:
            seed (int): Seed for generating random numbers
            noise_gain (torch.Tensor): Weight tensor of the noise
            noise_offset (torch.Tensor): Offset tensor of the noise
        """

        super().__init__(**kwargs)

        self.rng = torch.Generator().manual_seed(seed)
        self.noise_gain = noise_gain
        self.noise_offset = noise_offset

    def forward(self, output_shape: Tuple[int, ...]):
        return (
            self.noise_offset
            + torch.randn(output_shape, generator=self.rng) * self.noise_gain
        )
