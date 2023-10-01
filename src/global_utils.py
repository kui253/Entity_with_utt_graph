import yaml
from types import SimpleNamespace
import random
import os
import numpy as np
import torch
import wandb
import time
import logging
from pytorch_lightning import LightningDataModule


def get_config(config_dir):
    with open(config_dir, encoding="utf-8") as fp:
        data = yaml.load(fp, Loader=yaml.FullLoader)
    data = dictionary_to_namespace(data)
    return data


def dictionary_to_namespace(data):
    if type(data) is list:
        return list(map(dictionary_to_namespace, data))
    elif type(data) is dict:
        sns = SimpleNamespace()
        for key, value in data.items():
            setattr(sns, key, dictionary_to_namespace(value))
        return sns
    else:
        return data


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def init_wandb(cfg):
    os.system("wandb online")
    wandb.login(
        key="033031a3151e80aac253f56b432cbf4c66ab3f0f", host="https://api.wandb.ai"
    )
    res = "{}-{}-{}-{}".format(
        time.localtime().tm_year,
        time.localtime().tm_mon,
        time.localtime().tm_mday,
        time.localtime().tm_min,
    )
    wandb.init(
        # set the wandb project where this run will be logged
        project="Evaluate_Student_Summaries_whd",
        # track hyperparameters and run metadata
        config={
            "learning_rate": cfg.optimizer.lr,
            "architecture": cfg.model.backbone_type,
            "epochs": cfg.hyperparam.train_epochs,
        },
        name=cfg.hyperparam.config_name + res,
    )


def init_logger(cfg):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=cfg.hyperparam.train_log_dir
        + cfg.dataset.train_dataset_name.split(".")[0]
        + "{}-{}-{}-{}.log".format(
            time.localtime().tm_year,
            time.localtime().tm_mon,
            time.localtime().tm_mday,
            time.localtime().tm_min,
        ),
    )


def get_rouge_score(hyps, refs):
    import rouge

    evaluator = rouge.Rouge(
        metrics=["rouge-n", "rouge-l"],
        max_n=2,
        limit_length=True,
        length_limit=100,
        length_limit_type="words",
        apply_avg=True,
        apply_best=False,
        alpha=0.5,  # Default F1_score
        weight_factor=1.2,
        stemming=True,
    )
    py_rouge_scores = evaluator.get_scores(hyps, refs)
    return py_rouge_scores
