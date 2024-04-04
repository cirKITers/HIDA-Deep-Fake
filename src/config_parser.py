from argparse import ArgumentParser
import yaml
import os
from log import get_logger

log = get_logger(__name__)


class ConfigParser:

    def __init__(self):
        parser = ArgumentParser()

        parser.add_argument("--config_dir", type=str, default="config.yaml")
        parser.add_argument("--data_dir", type=str, default="./data")
        parser.add_argument("--seed", type=int, default=100)
        parser.add_argument("--image_size", type=int, default=12)
        parser.add_argument("--batch_size", type=int, default=20)
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--n_figures", type=int, default=10)
        parser.add_argument("--n_samples", type=int, default=400)
        parser.add_argument("--n_qubits", type=int, default=8)
        parser.add_argument("--latent_size", type=int, default=100)
        parser.add_argument("--n_layers", type=int, default=22)
        parser.add_argument("--d_lr", type=float, default=0.0002)
        parser.add_argument("--c_lr", type=float, default=0.0002)
        parser.add_argument("--q_lr", type=float, default=0.01)
        parser.add_argument("--noise_gain", type=float, default=0.5)
        parser.add_argument("--noise_offset", type=float, default=0.0)
        parser.add_argument("--selected_label", type=int, default=0)
        parser.add_argument("--tracking_uri", type=str, default="http://127.0.0.1:5000")
        parser.add_argument("--log_level", type=str, default="INFO")

        self.args = parser.parse_args()

    def get_config(self):
        if os.path.exists(self.args.config_dir):
            with open(self.args.config_dir, "r") as stream:
                try:
                    config = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)

        else:
            log.warning(
                f"No config file {self.args.config_dir} found. Using default values."
            )
            config = {}
        return config or self.args
