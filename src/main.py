import torch
import argparse
import random
import os
import numpy as np
from global_utils import seed_everything, get_config, init_wandb, init_logger
import logging


def main(args):
    cfg = get_config(args.config_dir)
    seed_everything(cfg.hyperparam.seed)
    if cfg.hyperparam.use_wandb:
        init_wandb(cfg)
    init_logger(cfg)
    logger = logging.getLogger("my_log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_dir",
        type=str,
        default="./config/ConfigBartBase.yml",
        help="the config file you want to use",
    )
    parser.add_argument(
        "--trial_name", type=str, default=None, help="what name is this trial in wandb"
    )
    args = parser.parse_args()
    main(args)
    pass
