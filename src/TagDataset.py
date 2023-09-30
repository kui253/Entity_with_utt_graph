from typing import Any
from torch.utils.data import Dataset
import torch


class SamSumDataset(Dataset):
    def __init__(self, dia, tokenizer, cfg):
        self.dia = dia
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __getitem__(self, index):
        # 只能在batchsize = 1的情况下
        text_in = self.dia[index]["unit_utts"]
        summary = self.dia[index]["summary"]
        utts = self.dia[index]["utterances"]
        speaker_names_set = self.dia[index]["nameSetInorder"]
        speaker_names_seq = self.dia[index]["names"]
        admat = self.dia[index]["map_mat"]
        model_inputs = {}

        if "#" not in self.tokenizer.additional_special_tokens:
            special_tokens_dict = {"additional_special_tokens": ["#"]}
            self.tokenizer.add_special_tokens(special_tokens_dict)

        baseline_input_ids = self.tokenizer(
            text_in,
            max_length=self.cfg.hyperparam.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        model_inputs["input_ids"] = baseline_input_ids["input_ids"]
        model_inputs["attention_mask"] = baseline_input_ids["attention_mask"]

        if self.cfg.model.model_type == "gtbart":
            tokenized_utts = self.tokenizer(utts, padding=False)
            model_inputs["gt_input_ids"] = tokenized_utts["input_ids"]
            model_inputs["gt_attention_mask"] = tokenized_utts["attention_mask"]
            model_inputs["utts_map"] = torch.tensor(admat)
            model_inputs["utts_nums"], model_inputs["ents_nums"] = admat.shape
        # 准备labels数据集
        labels = self.tokenizer(
            summary,
            max_length=self.cfg.hyperparam.max_sum_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        model_inputs["target_input_ids"] = labels["input_ids"]
        model_inputs["target_attention_ids"] = labels["attention_mask"]
        return model_inputs

    def __len__(self):
        return len(self.dia)
