# %%
import pennylane as qml
import torchvision
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import mlflow
from typing import List, Tuple


# To get this running:
# - make sure to install [pytorch correctly](https://pytorch.org/get-started/locally/)
# - launch the mlflow tracking server: `mlflow server`
# - run the notebook


# %%
# Initialization of some global parameters
image_size = 12
batch_size = 20
epochs = 100
n_figures = 8
selected_label = 0
n_samples = 200
n_qubits = 6
n_layers = 22
noise_gain = torch.pi / 6
noise_offset = torch.pi / 2
seed = 100

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

        self.model = nn.Sequential(
            # Inputs to first hidden layer (num_input_features -> 64)
            nn.Linear(image_size * image_size, 64),
            nn.ReLU(),
            # First hidden layer (64 -> 16)
            nn.Linear(64, 16),
            nn.ReLU(),
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
        seed: int,
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

        self.rng = torch.Generator().manual_seed(seed)
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
class Generator(nn.Module):
    """Quantum Generator Model
    This model takes a samples from the noise source and coordinates and outputs the image
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        **kwargs,
    ) -> None:
        """Construct a quantum circuit as a TorchLayer.

        Args:
            n_qubits (int): Number of qubits in the circuit
            n_layers (int): Number of layers in the circuit (excluding the last one)

        Returns:
            None
        """
        super().__init__(**kwargs)

        self.n_qubits = n_qubits
        self.n_layers = n_layers + 1

        dev = qml.device("default.qubit", wires=self.n_qubits)
        self.qnode = qml.QNode(self.circuit, dev, interface="torch")
        self.qlayer = qml.qnn.TorchLayer(
            self.qnode,
            {"weights": [self.n_layers, self.n_qubits, self.vqc(None)]},
        )

    def circuit(self, weights: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """
        The Quantum Generator Model quantum circuit.

        Args:
            weights (torch.Tensor): The weights for the trainable circuit.
            inputs (torch.Tensor): The input coordinates and noise states.

        Returns:
            torch.Tensor: The expectation value of the PauliZ observable.
        """
        x = inputs[:, :2]  # [B*IS*IS, NQ+2] -> [B*IS*IS, 2]
        p = inputs[:, 2:]  # [B*IS*IS, NQ+2] -> [B*IS*IS, NQ]

        # build the trainable circuit
        for layer in range(self.n_layers - 1):
            self.nec(p)  # prepare random states
            self.vqc(weights[layer])
            self.iec(x)

        # add a last vqc layer
        self.vqc(weights[-1])

        return qml.expval(qml.PauliZ(0))

    def nec(
        self,
        p: torch.Tensor,  # [B*IS*IS, NQ]
    ) -> None:
        """Prepares the random states in the quantum circuit.

        Args:
            p (torch.Tensor): The input noise states.
        """
        for qubit in range(self.n_qubits):
            qml.RZ(p[:, qubit], wires=qubit)

    def iec(self, x: torch.Tensor) -> None:
        """Encodes the input coordinates onto the quantum circuit.

        Args:
            x (torch.Tensor): The input coordinates. Shape = [B*IS*IS, 2]
        """
        for qubit in range(self.n_qubits):
            qml.RX(x[:, 0], wires=qubit)
            qml.RY(x[:, 1], wires=qubit)

    def vqc(self, weights: torch.Tensor) -> None:
        r"""Applies the variational quantum circuit to the qubits.

        Args:
            weights (torch.Tensor): The weights for the quantum circuit.
                Shape = [n_layers, n_qubits, n_params_per_layer].
        """
        if weights is None:
            return 3  # used to get the number of required params per layer

        for qubit, qubit_weights in enumerate(weights):
            qml.RX(qubit_weights[0], wires=qubit)
            qml.RZ(qubit_weights[1], wires=qubit)

        for qubit, qubit_weights in enumerate(weights):
            qml.CRX(
                qubit_weights[2],
                wires=[
                    weights.shape[0] - qubit - 1,
                    (weights.shape[0] - qubit) % weights.shape[0],
                ],
            )

    def forward(self, p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Generate images from the quantum generative model.

        Args:
            p (torch.Tensor): The input noise states. Shape = [B, NQ].
            x (torch.Tensor): The known image values. Shape = [B, IS, IS, 2].

        Returns:
            torch.Tensor: The generated images. Shape = [B, IS, IS, 2].
        """
        # get the known variables
        batch_size = x.shape[0]
        image_sidelength = x.shape[1]

        x_in = x.reshape(batch_size, -1, 2)  # [B, IS, IS, 2] -> [B, IS*IS, 2]
        p_in = p.repeat(
            1, image_sidelength * image_sidelength
        )  # [B, NQ] -> [B, IS*IS*NQ]
        p_in = p_in.reshape(
            batch_size, image_sidelength * image_sidelength, self.n_qubits
        )  # [B, IS*IS*NQ] -> [B, IS*IS, NQ]
        combined = torch.cat(
            (x_in, p_in), dim=2
        )  # [B, IS*IS, 2] + [B, IS*IS, NQ] -> [B, IS*IS, NQ+2]
        z = self.qlayer(combined)
        z = (z + 1) / 2  # move into range [0,1]
        return z


# %%
# Generate the individual models
noise_source = NoiseSource(
    (n_qubits), seed=seed, noise_gain=noise_gain, noise_offset=noise_offset
)
generator = Generator(n_qubits=n_qubits, n_layers=n_layers)
discriminator = Discriminator()

# %%
# Initialize the optimizers for the generator and discriminator separately
# We use individual optimizers because it allows us to have different learning rates
# and further separate the parameters of the models as seen later in training
opt_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.01)
opt_generator = torch.optim.Adam(generator.parameters(), lr=0.05)

loss = nn.BCELoss(
    reduction="mean"
)  # BCELoss with mean so that we don't depend on the batch size


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
        tensors = tuple(2 * [torch.linspace(domain[0], domain[1], steps=samples)])
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        return mgrid


gan_dataset = Dataset(dataset[(dataset.train_labels == selected_label)], n_samples)
dataloader = torch.utils.data.DataLoader(
    gan_dataset, batch_size=batch_size, shuffle=True
)


# %%
torch.autograd.set_detect_anomaly(True)

# Do some mlflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("QGAN Test")

with mlflow.start_run() as run:
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

    print(
        f"Training started. Navigate to the MLflow UI at http://localhost:5000/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"
    )

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
            disc_loss_combined = disc_loss_real + disc_loss_fake

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
            opt_generator.step()

            disc_epoch_loss += disc_loss_combined.item()
            gen_epoch_loss += gen_loss.item()

            mlflow.log_metric("discriminator_loss_step", disc_loss_combined.item())
            mlflow.log_metric("generator_loss_step", gen_loss.item())

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
