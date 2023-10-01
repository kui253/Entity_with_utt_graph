import torch
import argparse
import random
import os
import numpy as np
from data_utils import get_dataloader, baseline_collate_fn, gtbart_collate_fn
from global_utils import seed_everything, get_config, init_wandb, init_logger
import logging
from pytorch_lightning import Trainer
from model import plModel


def main(args):
    cfg = get_config(args.config_dir)
    seed_everything(cfg.hyperparam.seed)
    if cfg.hyperparam.use_wandb:
        init_wandb(cfg)
    init_logger(cfg)
    cfg.hyperparam.mode = args.mode
    logger = logging.getLogger(__name__)
    pl_model = plModel(cfg)
    pl_trainer = Trainer(
        devices=1,
        accelerator="gpu",
        precision=16,
        check_val_every_n_epoch=cfg.hyperparam.evaluate_n_times_per_epoch,
        max_epochs=cfg.hyperparam.train_epochs,
        logger=False,
    )
    if cfg.model.model_type == "baseline":
        collate_function = baseline_collate_fn
    else:
        collate_function = gtbart_collate_fn

    train_dl = get_dataloader(cfg=cfg, collate_fn=collate_function, mode="train")
    val_dl = get_dataloader(cfg=cfg, collate_fn=collate_function, mode="validation")
    test_dl = get_dataloader(cfg=cfg, collate_fn=collate_function, mode="test")
    pl_trainer.fit(model=pl_model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    pl_trainer.test(model=pl_model, dataloaders=test_dl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_dir",
        type=str,
        default="./config/ConfigBartBase.yml",
        help="the config file you want to use",
    )
    parser.add_argument(
        "--mode", type=str, default="train", help="what name is this trial in wandb"
    )
    args = parser.parse_args()
    main(args)
    pass
