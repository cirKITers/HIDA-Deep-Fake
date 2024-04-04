import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_size, image_size):
        super(Generator, self).__init__()
        self.image_size = image_size
        # FIXME: I removed nz and as we only use n_qubits for the noise source, this migth cause a problem in training the classical generator
        self.main = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.image_size * self.image_size),
            nn.Tanh(),
        )

    def forward(self, p_hat, x):
        return self.main(p_hat)
