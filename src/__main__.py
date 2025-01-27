import torch
import torch.nn as nn
import mlflow
from mlflow.models import infer_signature
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import Namespace


from log import create_logger, set_level
from discriminator import Discriminator
from classical_generator import Generator as CGenenerator
from classical_noise_source import NoiseSource as CNoiseSource
from quantum_generator import Generator as QGenerator
from quantum_noise_source import NoiseSource as QNoiseSource
from dataset import Dataset
from config_parser import ConfigParser

log = create_logger(__name__)


class Trainer:
    def __init__(self, params: Namespace) -> None:
        """
        Initializes the trainer with the given configuration parameters.

        Parameters
        ----------
        params : Namespace
            The configuration parameters for the experiment.

        Returns
        -------
        None
        """
        self.params = params
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.autograd.set_detect_anomaly(True)
        # Seed rng for the dataloader
        self.rng = torch.Generator().manual_seed(params.seed)
        

        set_level(log, params.log_level)

        log.info("Intializing dataset")
        dataset = Dataset(
            selected_label=params.selected_label,
            n_samples=params.n_samples,
            image_size=params.image_size,
            data_dir=params.data_dir,
            device=self.device,
        )
        log.info("Setting up dataloader")
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=params.batch_size,
            shuffle=True,
            drop_last=True,
            generator=self.rng,
        )

        self.loss = nn.BCELoss()

        self.setup_mlflow()

    def setup_mlflow(self) -> None:
        """
        Setup mlflow experiment using the given server URL.

        Parameters
        ----------
        self : Trainer
            The trainer instance.

        Returns
        -------
        None
        """
        log.info(
            f"Setting up mlflow experiment using server at {self.params.tracking_uri}"
        )
        mlflow.set_tracking_uri(str(self.params.tracking_uri))
        mlflow.set_experiment("QGAN Test")

    def run_mlflow(self):
        """
        Start an mlflow run and return the run object.

        Returns
        -------
        mlflow.ActiveRun
            The run object representing the new experiment run.
        """
        log.info("Trying to connect to mlflow server.")
        run = mlflow.start_run()
        log.info("Connected!")
        return run

    def train_general_gan(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        opt_generator: torch.optim.Optimizer,
        opt_discriminator: torch.optim.Optimizer,
        noiseSource: nn.Module,
    ) -> None:
        """
        Train the general GAN model.

        Parameters
        ----------
        generator : Generator
            Generator model.
        discriminator : Discriminator
            Discriminator model.
        opt_generator : optim.Optimizer
            Optimizer for the generator.
        opt_discriminator : optim.Optimizer
            Optimizer for the discriminator.
        noiseSource : NoiseSource
            Noise source model.

        Returns
        -------
        None
        """
        with self.run_mlflow() as run:
            log.info(
                f"Results will be logged at http://localhost:5000/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"  # noqa
            )
            mlflow.log_params(self.params.__dict__)

            # Main Training Loop
            total_step = 0
            for epoch in tqdm(range(self.params.epochs), desc="Training (Epochs)"):
                disc_epoch_loss = 0
                gen_epoch_loss = 0
                for i, (z, x) in enumerate(self.dataloader):
                    p_hat = torch.stack([noiseSource() for _ in range(x.shape[0])])

                    # sample the images using the generated noise
                    z_hat = generator(p_hat, x)

                    for _ in range(self.params.generator_update_step):
                        # train the discriminator
                        y_hat = discriminator(
                            z_hat.detach()
                        )  # detach the generator output from the graph
                        y = discriminator(z)

                        disc_loss_real = self.loss(
                            y, torch.ones_like(y)
                        )  # 1s: real images
                        disc_loss_fake = self.loss(
                            y_hat, torch.zeros_like(y_hat)
                        )  # 0s: fake images
                        # Wasserstein discriminator loss
                        disc_loss_combined = disc_loss_fake + disc_loss_real

                        opt_discriminator.zero_grad()
                        disc_loss_combined.backward()
                        opt_discriminator.step()

                    # train the generator
                    opt_generator.zero_grad()
                    y_hat = discriminator(
                        z_hat
                    )  # run discriminator again, but with attached generator output
                    gen_loss = self.loss(
                        y_hat, torch.ones_like(y_hat)
                    )  # all-real labels
                    gen_loss.backward()
                    opt_generator.step()  # this will cause the update of parameters only in the generator

                    disc_epoch_loss += disc_loss_combined.item()
                    gen_epoch_loss += gen_loss.item()

                    mlflow.log_metric(
                        "discriminator_loss_step",
                        disc_loss_combined.item(),
                        step=total_step,
                    )
                    mlflow.log_metric(
                        "generator_loss_step", gen_loss.item(), step=total_step
                    )

                    total_step += 1

                mlflow.log_metric(
                    "discriminator_loss",
                    disc_epoch_loss / len(self.dataloader),
                    step=epoch,
                )
                mlflow.log_metric(
                    "generator_loss", gen_epoch_loss / len(self.dataloader), step=epoch
                )

                preds = (
                    z_hat.reshape(-1, self.params.image_size, self.params.image_size)
                    .detach()
                    .cpu()
                    .numpy()
                )
                fig = plt.figure(figsize=(self.params.n_figures, 1))
                for i in range(min(self.params.n_figures, self.params.batch_size)):
                    plt.subplot(1, self.params.n_figures, i + 1)
                    plt.axis("off")
                    plt.imshow(preds[i], cmap="gray")

                mlflow.log_figure(fig, f"generated_epoch_{epoch}.png")
                plt.close()

            log.info("Saving models")

            mlflow.pytorch.log_model(
                generator,
                artifact_path=f"{generator.__class__.__module__}",
                registered_model_name=generator.__class__.__module__.replace(
                    "_", " "
                ).title(),
                signature=infer_signature(
                    {"p": p_hat.numpy(), "x": x.numpy()}, {"z": z_hat.detach().numpy()}
                ),
            )
            mlflow.pytorch.log_model(
                discriminator,
                artifact_path=f"{discriminator.__class__.__module__}",
                registered_model_name=discriminator.__class__.__module__.replace(
                    "_", " "
                ).title(),
                signature=infer_signature(
                    {"z": z.detach().numpy()}, {"y": y.detach().numpy()}
                ),
            )

        log.info("Training finished")

    def train_cc_gan(
        self,
    ) -> None:
        """Training of Classical Generator and Classical Discriminator GAN

        Args:
            None

        Returns:
            None
        """
        log.info("Instantiating classical noise source")
        noise_source = CNoiseSource(
            output_shape=(self.params.latent_size),
            rng=self.rng,
            noise_gain=self.params.noise_gain,
            noise_offset=self.params.noise_offset,
        )

        log.info("Instantiating classical generator")
        # Re-Seed for EACH model initialisation
        torch.manual_seed(self.params.seed)
        generator = CGenenerator(
            latent_size=self.params.latent_size, image_size=self.params.image_size
        ).to(self.device)

        log.info("Instantiating discriminator")
        # Re-Seed for EACH model initialisation
        torch.manual_seed(self.params.seed)
        discriminator = Discriminator(image_size=self.params.image_size).to(self.device)

        opt_generator = torch.optim.Adam(generator.parameters(), lr=self.params.gen_lr)
        opt_discriminator = torch.optim.Adam(
            discriminator.parameters(), lr=self.params.disc_lr
        )

        log.info("Starting training of CC-GAN")
        self.train_general_gan(
            generator=generator,
            discriminator=discriminator,
            opt_generator=opt_generator,
            opt_discriminator=opt_discriminator,
            noiseSource=noise_source,
        )

    def train_cq_gan(
        self,  # type: Trainer
    ) -> None:
        """Training of Classical Generator and Quantum Discriminator GAN

        Args:
            None

        Returns:
            None
        """
        log.info("Instantiating classical noise source")
        log.warning(
            f"Using latent space dimension {self.params.n_qubits} instead of {self.params.latent_size}"
        )
        noise_source = CNoiseSource(
            output_shape=(self.params.n_qubits),  # type: ignore
            rng=self.rng,
            noise_gain=self.params.noise_gain,
            noise_offset=self.params.noise_offset,
        )

        log.info("Instantiating quantum generator")
        # Re-Seed for EACH model initialisation
        torch.manual_seed(self.params.seed)
        generator = QGenerator(
            n_qubits=self.params.n_qubits,
            n_layers=self.params.n_layers,
        ).to(self.device)

        log.info("Instantiating discriminator")
        # Re-Seed for EACH model initialisation
        torch.manual_seed(self.params.seed)
        discriminator = Discriminator(image_size=self.params.image_size).to(self.device)

        opt_generator = torch.optim.Adam(generator.parameters(), lr=self.params.gen_lr)
        opt_discriminator = torch.optim.Adam(
            discriminator.parameters(), lr=self.params.disc_lr
        )

        log.info("Starting training of CQ-GAN")
        self.train_general_gan(
            generator=generator,
            discriminator=discriminator,
            opt_generator=opt_generator,
            opt_discriminator=opt_discriminator,
            noiseSource=noise_source,
        )

    def train_qc_gan(self):
        """Training of Quantum Generator and Classical Discriminator GAN

        Args:
            None

        Returns:
            None
        """
        raise NotImplementedError()

    def train_qq_gan(self):
        """Training of Quantum Generator and Quantum Discriminator GAN

        Args:
            None

        Returns:
            None
        """
        raise NotImplementedError()

    def train_final_discriminator(self, params):
        raise NotImplementedError()


def main():
    configParser = ConfigParser()
    params = configParser.get_config()

    trainer = Trainer(params)
    if params.model == "all":
        trainer.train_cc_gan()
        trainer.train_cq_gan()
    elif params.model == "cc":
        trainer.train_cc_gan()
    elif params.model == "cq":
        trainer.train_cq_gan()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
