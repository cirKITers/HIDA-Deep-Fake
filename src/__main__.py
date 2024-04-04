import torch
import torch.nn as nn
import mlflow
import matplotlib.pyplot as plt

from log import create_logger, set_level

log = create_logger(__name__)

from discriminator import Discriminator
from classical_generator import Generator as CGenenerator
from classical_noise_source import NoiseSource as CNoiseSource
from quantum_generator import Generator as QGenerator
from quantum_noise_source import NoiseSource as QNoiseSource
from gan import GenerativeAdversarialNetwork as GAN
from dataset import Dataset
from config_parser import ConfigParser


class Trainer:
    def __init__(self, params) -> None:
        self.params = params
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.autograd.set_detect_anomaly(True)
        self.rng = torch.Generator().manual_seed(params.seed)
        set_level(log, params.log_level)

        dataset = Dataset(
            selected_label=params.selected_label,
            n_samples=params.n_samples,
            image_size=params.image_size,
            data_dir=params.data_dir,
            device=self.device,
        )
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=params.batch_size,
            shuffle=True,
            drop_last=True,
            generator=self.rng,
        )

        self.loss = nn.BCELoss()

        self.setup_mlflow()

    def setup_mlflow(self):
        mlflow.set_tracking_uri(self.params.tracking_uri)
        mlflow.set_experiment("QGAN Test")

    def run_mlflow(self):
        log.info(
            f"Trying to connect to mlflow server. Make sure it runs at {mlflow.get_tracking_uri()}"
        )
        return mlflow.start_run()

    def train_general_gan(
        self,
        generator,
        discriminator,
        opt_generator,
        opt_discriminator,
        noiseSource,
        loss,
    ):
        with self.run_mlflow() as run:
            mlflow.log_params(self.params.__dict__)

            # Plot figures

            # Main Training Loop
            total_step = 0
            for epoch in range(self.params.epochs):
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

                        disc_loss_real = loss(y, torch.ones_like(y))  # 1s: real images
                        disc_loss_fake = loss(
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
                    gen_loss = loss(y_hat, torch.ones_like(y_hat))  # all-real labels
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

                    log.debug(f"Finished step {total_step}")

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

                log.info(f"Finished epoch {epoch}")

    def train_cc_gan(self):
        noise_source = CNoiseSource(
            output_shape=(self.params.latent_size),
            rng=self.rng,
            noise_gain=self.params.noise_gain,
            noise_offset=self.params.noise_offset,
        )

        generator = CGenenerator(
            latent_size=self.params.latent_size, image_size=self.params.image_size
        ).to(self.device)
        discriminator = Discriminator(image_size=self.params.image_size).to(self.device)

        opt_generator = torch.optim.Adam(generator.parameters(), lr=self.params.c_lr)
        opt_discriminator = torch.optim.Adam(
            discriminator.parameters(), lr=self.params.d_lr
        )
        loss = nn.BCELoss()

        self.train_general_gan(
            generator=generator,
            discriminator=discriminator,
            opt_generator=opt_generator,
            opt_discriminator=opt_discriminator,
            noiseSource=noise_source,
            loss=loss,
        )

    def train_cq_gan(self):
        raise NotImplementedError()

    def train_qc_gan(self):
        raise NotImplementedError()

    def train_qq_gan(self):
        raise NotImplementedError()

    def train_final_discriminator(self, params):
        raise NotImplementedError()


def train_all(params):
    trainer = Trainer(params)
    trainer.train_cc_gan()
    pass


def main():
    configParser = ConfigParser()
    params = configParser.get_config()

    train_all(params)


if __name__ == "__main__":
    main()
