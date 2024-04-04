import torchvision
import torch
from typing import List, Tuple


class Dataset(torch.utils.data.Dataset):
    """Custom Dataset that allows us to get the coordinates an the reference images.
    It also includes noise samples.

    Args:
        torch (_type_): _description_
    """

    def __init__(
        self,
        selected_label: int = 0,
        n_samples: int = 200,
        image_size: int = 12,
        data_dir: str = "./data",
        device="cpu",
    ) -> None:
        """Initialize the dataset.

        Args:
            selected_label (int): The label to select from 0-9.
            n_samples (int): The number of samples to include in the dataset.
            image_size (int): The size of the MNIST images to load.
            data_dir (str): The directory to load the MNIST dataset from.

        Returns:
            None
        """
        self.device = device

        # self.noise_source = noise_source
        self.image_size = image_size

        # Load MNIST dataset
        self.dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(image_size),
                    torchvision.transforms.ToTensor(),
                ]
            ),
        )

        self.prepare(
            images=(self.dataset.targets == selected_label).nonzero().flatten(),
            n_samples=n_samples,
        )

    def prepare(self, images: List[int], n_samples: int):
        self.z = torch.stack(
            [self.dataset[zidx][0][0] for zidx in images[:n_samples]]
        )  # first [0] discards label, second [0] discards color channel
        self.x = torch.stack(
            [
                self.generate_grid([-torch.pi / 2, torch.pi / 2], self.image_size)
                for _ in range(self.z.shape[0])
            ]
        )
        # self.p_hat = torch.stack(
        #     [self.noise_source() for _ in range(self.z.shape[0])]
        # )  # noise_source()

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Arguments:
            idx (int): The index of the item to retrieve

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The coordinates, image, and noise sample
        """

        return self.z[idx], self.x[idx]

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
            2
            * [torch.linspace(domain[0], domain[1], steps=samples, device=self.device)]
        )
        # indexing 'ij' means the dimensions are in the same order as the cardinality of the input
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
        return mgrid
