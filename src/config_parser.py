from argparse import ArgumentParser
import yaml
import os
from log import get_logger

log = get_logger(__name__)


class ConfigParser:

    def __init__(self):
        parser = ArgumentParser()

        parser.add_argument("--default_config", type=str, default="config/default.yaml")
        parser.add_argument("--overwrite_config", type=str)
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
        if os.path.exists(self.args.default_config):
            log.info(f"Reading default config from {self.args.default_config}")
            with open(self.args.default_config, "r") as stream:
                try:
                    config = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    log.error(f"Error while loading config file")
                    raise yaml.YAMLError(exc)

        else:
            log.warning(
                f"No config file {self.args.default_config} found. Using command line values."
            )
            config = {}

        if os.path.exists(self.args.overwrite_config):
            log.info(f"Reading overwrite config from {self.args.overwrite_config}")
            with open(self.args.overwrite_config, "r") as stream:
                try:
                    ow_config = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    log.error(f"Error while loading config file")
                    raise yaml.YAMLError(exc)

        else:
            log.warning(
                f"No config file {self.args.overwrite_config} found. Using default values."
            )
            ow_config = {}

        # Iterate all arguments
        for k, v in self.args.__dict__.items():
            # check if the argument is provided
            if v is None:
                # check if the overwrite config file has the argument
                if k in ow_config.keys():
                    v = ow_config[k]
                elif k in config.keys():
                    v = config[k]
                else:
                    log.error(f"Key {k} not provided as argument")
                    raise KeyError(
                        f"Key {k} not found neither in config nor overwrite config file."
                    )
                # replace the value with the one from the config file
            else:
                log.info(f"Using provided argument {k}: {v}")
            # set the value to the namespace
            setattr(self.args, k, v)
        return self.args
