from datasets import load_dataset
import spacy
from tqdm import tqdm
import re
from transformers import AutoTokenizer
import numpy as np
import pickle
from collections import OrderedDict
import os
from global_utils import get_config
from TagDataset import SamSumDataset
from torch.utils.data import DataLoader
import torch


def remove_duplicates(lst):
    return list(OrderedDict.fromkeys(lst))


def rawdataset_processing(dataset, mode="train", resume=0):
    Dia = []
    nlp = spacy.load("en_core_web_sm")

    if mode == "train":
        print("processing the {} dataset".format(mode))
    elif mode == "valid":
        print("processing the {} dataset".format(mode))
    elif mode == "test":
        print("processing the {} dataset".format(mode))

    pbar = tqdm(dataset[mode], desc="processing the example for train")
    for ids, example in enumerate(pbar):
        if ids < resume:
            continue
        if resume != 0 and ids == resume:
            Dia = pickle.load(open("./data/samsum_{}{}.pkl".format(mode, resume), "rb"))
        utterances_features = {}
        temp_str = ""
        utterances = re.split("\r\n|\n", example["dialogue"])  # 这里得到的是每一个句子

        utterances_features["names"] = [
            utterance.split(":", 1)[0] for utterance in utterances if len(utterance) > 0
        ]
        utterances_features["nameSetInorder"] = remove_duplicates(
            utterances_features["names"]
        )  # list
        utterances_features["utterances"] = [
            utterance.split(":", 1)[1] for utterance in utterances if len(utterance) > 0
        ]
        neat_utterances = " ".join(utterances_features["utterances"])
        utterances_features["unit_utts"] = " # ".join(utterances)
        inline_entities = [
            {"label": str(en.label_), "text": str(en.text)}
            for en in nlp(neat_utterances).ents
        ]

        if len(inline_entities) > 1:
            merged_entities = {}
            for item in inline_entities:
                if merged_entities.get(item["label"], None):
                    merged_entities[item["label"]] += [item["text"]]
                else:
                    merged_entities[item["label"]] = [item["text"]]
            result_data = {
                label: " | ".join(values) for label, values in merged_entities.items()
            }
            has_person = result_data.get("PERSON", None)
            if has_person:
                utterances_features["nameSetInorder"] = remove_duplicates(
                    utterances_features["nameSetInorder"] + [has_person]
                )
            for label, value in result_data.items():
                if label == "PERSON":
                    continue
                temp_str += "{" + label + ": " + value + "} "
            entities = remove_duplicates(
                utterances_features["nameSetInorder"]
                + [ent[0] for ent in merged_entities.values()]
            )
        elif len(inline_entities) == 1:
            label = inline_entities[0]["label"]
            value = inline_entities[0]["text"]
            entities = remove_duplicates(
                utterances_features["nameSetInorder"] + [str(value)]
            )
            if label == "PERSON":
                utterances_features["nameSetInorder"] = remove_duplicates(
                    utterances_features["nameSetInorder"] + [str(value)]
                )
            else:
                temp_str += "{" + label + ": " + value + "} "
        else:
            entities = utterances_features["nameSetInorder"]
        if temp_str == "":
            entities_chain = (
                "{PERSON: " + " | ".join(utterances_features["nameSetInorder"]) + "} "
            )
        else:
            entities_chain = (
                "{PERSON: "
                + " | ".join(utterances_features["nameSetInorder"])
                + "} "
                + temp_str
            )
        utterances_features["unit_utts"] = (
            entities_chain + " # " + utterances_features["unit_utts"]
        )
        map_mat = np.zeros((len(utterances), len(entities)))
        for id, utt in enumerate(utterances_features["utterances"]):
            map_mat[id, entities.index(utterances_features["names"][id])] = 1
            if len(nlp(utt).ents):
                utt_ents = [
                    entities.index(str(en))
                    for en in nlp(utt).ents
                    if str(en) in entities
                ]
                map_mat[id, utt_ents] = 1
        utterances_features["map_mat"] = map_mat
        utterances_features["summary"] = example["summary"]
        Dia.append(utterances_features)
        if (ids + 1) % 1000 == 0:
            print("saving the {} example".format(ids + 1))
            pickle.dump(Dia, open("./data/samsum_{}{}.pkl".format(mode, ids + 1), "wb"))

    pickle.dump(Dia, open("./data/samsum_{}.pkl".format(mode), "wb"))


def get_data(cfg, mode="train"):  # done
    if mode == "train":
        path = os.path.join(cfg.hyperparam.input_dir, cfg.dataset.train_dataset_name)
    elif mode == "validation":
        path = os.path.join(cfg.hyperparam.input_dir, cfg.dataset.val_dataset_name)

    else:
        path = os.path.join(cfg.hyperparam.input_dir, cfg.dataset.test_dataset_name)
    Dia = pickle.load(open(path, "rb"))
    return Dia


def get_dataset(cfg, mode="train"):  # done
    processed_data = get_data(cfg, mode)
    path = os.path.join(cfg.hyperparam.models_ckpt_dir, cfg.model.backbone_type)
    tokenizer = AutoTokenizer.from_pretrained(path)
    mydataset = SamSumDataset(dia=processed_data, tokenizer=tokenizer, cfg=cfg)
    return mydataset


def get_dataloader(cfg, collate_fn, mode="train"):
    mydataset = get_dataset(cfg, mode)
    if mode == "train":
        batch_size = cfg.hyperparam.train_batch_size
    elif mode == "validation":
        batch_size = cfg.hyperparam.val_batch_size

    else:
        batch_size = cfg.hyperparam.val_batch_size
    dataloader = DataLoader(
        mydataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.hyperparam.n_workers,
        collate_fn=collate_fn,
    )
    return dataloader


def gtbart_collate_fn(batch):  # 支持batch操作
    max_utt_num = max([item["utts_nums"] for item in batch])
    max_single_utt_num = max(
        [len(utt) for item in batch for utt in item["gt_input_ids"]]
    )
    max_ent_num = max([item["ents_nums"] for item in batch])
    pad_id = 1
    batch_utt_lens = []
    utts_ents_mat = []
    batched_features = {}
    for batch_item in batch:
        # if len(batch_item["gt_input_ids"]) > max_utt_num:
        #     batch_item["gt_input_ids"] = batch_item["gt_input_ids"][:max_utt_num]
        #     batch_item["gt_attention_mask"] = batch_item["gt_attention_mask"][
        #         :max_utt_num
        #     ]
        batch_utt_lens.append(len(batch_item["gt_input_ids"]))
        padded_mat = torch.zeros((max_utt_num, max_ent_num))
        padded_mat[: batch_item["utts_nums"], : batch_item["ents_nums"]] = batch_item[
            "utts_map"
        ]
        utts_ents_mat.append(padded_mat)
        for utt, utt_attn in zip(
            batch_item["gt_input_ids"], batch_item["gt_attention_mask"]
        ):
            diff = max_single_utt_num - len(utt)
            utt += [pad_id] * diff
            utt_attn += [0] * diff
    batched_features["gt_input_ids"] = torch.cat(
        [torch.tensor(x["gt_input_ids"]) for x in batch], dim=0
    )
    batched_features["gt_attention_ids"] = torch.cat(
        [torch.tensor(x["gt_attention_mask"]) for x in batch], dim=0
    )
    batched_features["batch_utt_lens"] = torch.tensor(batch_utt_lens)
    batched_features["uttts_ents_mat"] = torch.cat(utts_ents_mat, dim=0).view(
        -1, max_utt_num, max_ent_num
    )  # batch * max_utt_num * max_ent_num
    batched_features["input_ids"] = torch.cat([x["input_ids"] for x in batch], dim=0)
    batched_features["attention_mask"] = torch.cat(
        [x["attention_mask"] for x in batch], dim=0
    )
    batched_features["decoder_input_ids"] = torch.cat(
        [x["target_input_ids"] for x in batch], dim=0
    )
    batched_features["decoder_attention_mask"] = torch.cat(
        [x["target_attention_ids"] for x in batch], dim=0
    )
    return batched_features


def baseline_collate_fn(batch):
    batched_features = {}
    batched_features["input_ids"] = torch.cat([x["input_ids"] for x in batch], dim=0)
    batched_features["attention_mask"] = torch.cat(
        [x["attention_mask"] for x in batch], dim=0
    )
    batched_features["decoder_input_ids"] = torch.cat(
        [x["target_input_ids"] for x in batch], dim=0
    )
    batched_features["decoder_attention_mask"] = torch.cat(
        [x["target_attention_ids"] for x in batch], dim=0
    )
    return batched_features


# if __name__ == "__main__":
#     # dataset = load_dataset("/data1/whd/diaResearch/DST_tag_method/samsum/samsum.py")
#     # rawdataset_processing(dataset=dataset, mode="train")
#     # rawdataset_processing(dataset=dataset, mode="validation")
#     # rawdataset_processing(dataset=dataset, mode="test")
#     cfg = get_config("./config/ConfigBartBase.yml")
#     if cfg.model.model_type == "gtbart":
#         train_dl = get_dataloader(cfg, gtbart_collate_fn, mode="train")
#     else:
#         train_dl = get_dataloader(cfg, baseline_collate_fn, mode="train")
#     # pl_dataset = plDataset(cfg, collate_fn)
#     # train_dl = pl_dataset.train_dataloader()
#     for i in train_dl:
#         print(i)
#         break
#     # pass
