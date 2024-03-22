# %%
import pennylane as qml
import torchvision
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import mlflow
from typing import List, Tuple

from .QGenerator import Generator

# To get this running:
# - make sure to install [pytorch correctly](https://pytorch.org/get-started/locally/)
# - launch the mlflow tracking server: `mlflow server`
# - run the notebook


# %%
# Initialization of some global parameters
image_size = 12  # MNIST image size. This will increase computation time quadratically
batch_size = 20  # Batch Size
epochs = 100  # Epochs the model (gen + disc) are being trained
n_figures = 10  # Number of figures to plot
selected_label = 0  # Label to select from 0-9
n_samples = 400  # Size of the (reduced) dataset
n_qubits = 8  # Number of qubits
n_layers = 22  # Number of layers
noise_gain = torch.pi / 6  # Noise gain. Amplifies Z rotation
noise_offset = 0  # Noise offset. Offsets Z rotation
seed = 100  # Seed for generating random numbers

rng = torch.Generator().manual_seed(seed)

# %%
# Load MNIST dataset
dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose(
        [torchvision.transforms.Resize(image_size), torchvision.transforms.ToTensor()]
    ),
)


# %%
# Discriminator definition
class Discriminator(nn.Module):
    """Naive discriminator"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # num_input_features = image_size * image_size
        self.model = nn.Sequential(
            # Inputs to first hidden layer (num_input_features -> 32)
            nn.Linear(image_size * image_size, 32),
            nn.LeakyReLU(),
            # First hidden layer (32 -> 16)
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            # Second hidden layer (16 -> output)
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# %%
# Noise source definition
class NoiseSource(nn.Module):
    """Noise Source sampling from normal distribution with zero mean and unit variance weighted and shifted accordingly"""

    def __init__(
        self,
        output_shape: Tuple[int, ...],
        rng: int,
        noise_gain: torch.Tensor,
        noise_offset: torch.Tensor,
        **kwargs,
    ) -> None:
        """Noise Source sampling from normal distribution with zero mean and unit variance weighted and shifted accordingly.

        Args:
            output_shape (Tuple[int, ...]): Output shape of tensor
            seed (int): Seed for generating random numbers
            noise_gain (torch.Tensor): Weight tensor of the noise
            noise_offset (torch.Tensor): Offset tensor of the noise
        """

        super().__init__(**kwargs)

        self.rng = rng
        self.output_shape = output_shape
        self.noise_gain = noise_gain
        self.noise_offset = noise_offset

    def forward(self):
        return (
            self.noise_offset
            + torch.randn(self.output_shape, generator=self.rng) * self.noise_gain
        )


# %%
# Generator definition


# %%
# Determine if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Generate the individual models
noise_source = NoiseSource(
    (n_qubits), rng=rng, noise_gain=noise_gain, noise_offset=noise_offset
).to(device)
generator = Generator(n_qubits=n_qubits, n_layers=n_layers).to(device)
discriminator = Discriminator().to(device)

# %%
# Initialize the optimizers for the generator and discriminator separately
# We use individual optimizers because it allows us to have different learning rates
# and further separate the parameters of the models as seen later in training
opt_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.001)
opt_generator = torch.optim.Adam(generator.parameters(), lr=0.01)

loss = nn.BCELoss()  # BCELoss with mean so that we don't depend on the batch size


# %%
class Dataset(torch.utils.data.Dataset):
    """Custom Dataset that allows us to get the coordinates an the reference images.
    It also includes noise samples.

    Args:
        torch (_type_): _description_
    """

    def __init__(self, images: List[int], n_samples: int) -> None:
        """Initialize the dataset.

        Args:
            images (List[int]): A list of indices into the dataset.
            n_samples (int): The number of samples to include in the dataset.

        Returns:
            None
        """
        self.z = torch.stack(
            [dataset[zidx][0][0] for zidx in images[:n_samples]]
        )  # first [0] discards label, second [0] discards color channel
        self.x = torch.stack(
            [
                self.generate_grid([-torch.pi / 2, torch.pi / 2], image_size)
                for _ in range(self.z.shape[0])
            ]
        )
        self.p_hat = torch.stack(
            [noise_source() for _ in range(self.z.shape[0])]
        )  # noise_source()

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Arguments:
            idx (int): The index of the item to retrieve

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The coordinates, image, and noise sample
        """

        return self.z[idx], self.x[idx], self.p_hat[idx]

    def generate_grid(self, domain: Tuple[float, float], samples: int) -> torch.Tensor:
        """
        Generates a meshgrid for the given domain and number of samples.

        Args:
            domain (Tuple[float, float]): The domain of the grid, as a tuple of (min, max).
            samples (int): The number of samples to generate in each dimension.

        Returns:
            torch.Tensor: The generated meshgrid, with shape [samples, samples, 2].
        """
        tensors = tuple(
            2 * [torch.linspace(domain[0], domain[1], steps=samples, device=device)]
        )
        # indexing 'ij' means the dimensions are in the same order as the cardinality of the input
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
        return mgrid


# Generate the dataset using the selected label and the number of samples
gan_dataset = Dataset(
    (dataset.targets == selected_label).nonzero().flatten(), n_samples
)
# Configure dataloader to shuffle, drop last incomplete batch and use the rng
dataloader = torch.utils.data.DataLoader(
    gan_dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=rng
)


# %%
# Will help us to detect e.g. when parts of the model don't receive gradients
torch.autograd.set_detect_anomaly(True)

# Do some mlflow setup
tracking_uri = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("QGAN Test")

print(f"Trying to connect to mlflow server. Make sure it runs at {tracking_uri}")
with mlflow.start_run() as run:
    print("Connected to mlflow server")
    mlflow.log_param("image_size", image_size)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("selected_label", selected_label)
    mlflow.log_param("n_samples", n_samples)
    mlflow.log_param("n_qubits", n_qubits)
    mlflow.log_param("n_layers", n_layers)
    mlflow.log_param("noise_gain", noise_gain)
    mlflow.log_param("noise_offset", noise_offset)
    mlflow.log_param("seed", seed)

    fig = plt.figure(figsize=(n_figures, 1))
    for i in range(min(n_figures, batch_size)):
        plt.subplot(1, n_figures, i + 1)
        plt.axis("off")
        plt.imshow(gan_dataset[i][0].cpu().numpy(), cmap="gray")
    mlflow.log_figure(fig, f"reference_images.png")
    plt.close()

    print(
        f"Training started. Navigate to the MLflow UI at http://localhost:5000/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"
    )

    total_step = 0
    for epoch in range(epochs):

        disc_epoch_loss = 0
        gen_epoch_loss = 0
        for i, (z, x, p_hat) in enumerate(dataloader):
            # sample the images using the generated noise
            z_hat = generator(p_hat, x)

            # train the discriminator
            y_hat = discriminator(
                z_hat.detach()
            )  # detach the generator output from the graph
            y = discriminator(z.view(-1, image_size * image_size))

            disc_loss_real = loss(y, torch.ones_like(y))  # 1s: real images
            disc_loss_fake = loss(y_hat, torch.zeros_like(y_hat))  # 0s: fake images
            # Wasserstein discriminator loss
            disc_loss_combined = disc_loss_fake + disc_loss_real

            # if epoch == 0 or epoch % 2 == 0:
            opt_discriminator.zero_grad()
            disc_loss_combined.backward()
            opt_discriminator.step()

            # train the generator
            opt_generator.zero_grad()
            y_hat = discriminator(
                z_hat
            )  # run discriminator again, but with attached generator output
            gen_loss = loss(y_hat, torch.ones_like(y_hat))  # all-real labels
            gen_loss.backward()
            opt_generator.step()  # this will cause the update of parameters only in the generator

            disc_epoch_loss += disc_loss_combined.item()
            gen_epoch_loss += gen_loss.item()

            mlflow.log_metric(
                "discriminator_abs_mean_gradients",
                torch.stack([p.grad.abs().mean() for p in discriminator.parameters()])
                .mean()
                .item(),
                step=total_step,
            )
            mlflow.log_metric(
                "generator_abs_mean_gradients",
                torch.stack([p.grad.abs().mean() for p in generator.parameters()])
                .mean()
                .item(),
                step=total_step,
            )

            mlflow.log_metric(
                "discriminator_loss_step", disc_loss_combined.item(), step=total_step
            )
            mlflow.log_metric("generator_loss_step", gen_loss.item(), step=total_step)

            total_step += 1

        mlflow.log_metric(
            "discriminator_loss", disc_epoch_loss / len(dataloader), step=epoch
        )
        mlflow.log_metric(
            "generator_loss", gen_epoch_loss / len(dataloader), step=epoch
        )

        preds = z_hat.reshape(-1, image_size, image_size).detach().cpu().numpy()
        fig = plt.figure(figsize=(n_figures, 1))
        for i in range(min(n_figures, batch_size)):
            plt.subplot(1, n_figures, i + 1)
            plt.axis("off")
            plt.imshow(preds[i], cmap="gray")

        mlflow.log_figure(fig, f"generated_epoch_{epoch}.png")
        plt.close()
