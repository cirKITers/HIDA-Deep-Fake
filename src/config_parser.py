from argparse import ArgumentParser
import yaml
import os
from log import get_logger

log = get_logger(__name__)


class ConfigParser:

    def __init__(self):
        parser = ArgumentParser()

        parser.add_argument("--config_file", type=str, default="config.yaml")
        parser.add_argument("--seed", type=int)
        parser.add_argument("--image_size", type=int)
        parser.add_argument("--batch_size", type=int)
        parser.add_argument("--epochs", type=int)
        parser.add_argument("--n_figures", type=int)
        parser.add_argument("--n_samples", type=int)
        parser.add_argument("--n_qubits", type=int)
        parser.add_argument("--latent_size", type=int)
        parser.add_argument("--n_layers", type=int)
        parser.add_argument("--disc_lr", type=float)
        parser.add_argument("--gen_lr", type=float)
        parser.add_argument("--generator_update_step", type=int)
        parser.add_argument("--noise_gain", type=float)
        parser.add_argument("--noise_offset", type=float)
        parser.add_argument("--selected_label", type=int)
        parser.add_argument("--tracking_uri", type=str)
        parser.add_argument("--log_level", type=str)
        parser.add_argument("--model", type=str)

        self.args = parser.parse_args()

    def get_config(self):
        if os.path.exists(self.args.config_dir):
            log.info(f"Reading config from {self.args.config_dir}")
            with open(self.args.config_dir, "r") as stream:
                try:
                    config = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    log.error(f"Error while loading config file")
                    raise yaml.YAMLError(exc)

        else:
            log.warning(
                f"No config file {self.args.config_dir} found. Using default values."
            )
            config = {}

        # Iterate all arguments
        for k, v in self.args.__dict__.items():
            # check if the argument is provided
            if v is None:
                # check if the config file has the argument
                if k not in config.keys():
                    raise KeyError(f"Key {k} not found in config file.")
                # replace the value with the one from the config file
                v = config[k]
            else:
                log.debug(f"Using provided argument {k}: {v}")
            # set the value to the namespace
            setattr(self.args, k, v)
        return self.args
