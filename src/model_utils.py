from typing import Tuple, Optional
from pytorch_lightning import Callback
import pytorch_lightning as pl
import torch
import os
from transformers.file_utils import ModelOutput
from dataclasses import dataclass


@dataclass
class DualBaseModelOutput(ModelOutput):
    """
    This is DualBaseModelOutput for dual encoder outputs: low_encoder and high_encoder
    The original member for BaseModelOutput is still the same
    1.last_hidden_state
    2.hidden_states
    3.attentions
    We add additional members:
    1.speaker_hidden_states
    2.speaker_attentions
    3.speaker_attention_mask(for generation)
    """

    low_encoder_last_hidden_state: torch.FloatTensor = None
    low_encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    low_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    low_encoder_attention_mask: torch.LongTensor = None

    high_encoder_last_hidden_state: torch.FloatTensor = None
    high_encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    high_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    high_encoder_attention_mask: torch.LongTensor = None


class saveCallBack(Callback):
    def __init__(self, cfg, save_ckpt_name: str, mode: str = "train"):
        super().__init__()
        self.cfg = cfg
        self.save_ckpt_name = save_ckpt_name
        self.mode = mode
        if not os.path.exists(cfg.hyperparam.checkpoint_dir):
            os.mkdir(cfg.hyperparam.checkpoint_dir)

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        if (
            trainer.current_epoch + 1
        ) % self.cfg.hyperparam.evaluate_n_times_per_epoch == 0:
            if self.mode == "train":
                torch.save(
                    pl_module.model.state_dict(),
                    os.path.join(
                        self.cfg.hyperparam.checkpoint_dir,
                        "config_v_{}_on_step_{}_{}".format(
                            self.cfg.hyperparam.config_name,
                            trainer.global_step,
                            self.save_ckpt_name,
                        ),
                    ),
                )

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        torch.save(
            pl_module.model.state_dict(),
            os.path.join(
                self.cfg.hyperparam.checkpoint_dir,
                "config_v_{}_on_fitEnd_{}".format(
                    self.cfg.hyperparam.config_name, self.save_ckpt_name
                ),
            ),
        )

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        pl_module.model.load_state_dict(
            torch.load(
                os.path.join(
                    self.cfg.hyperparam.checkpoint_dir,
                    "config_v_{}_on_testStart_{}".format(
                        self.cfg.hyperparam.config_name, self.save_ckpt_name
                    ),
                ),
            )
        )
