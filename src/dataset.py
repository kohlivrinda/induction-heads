from torch.utils.data import DataLoader, Dataset
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from functools import partial
from torch.nn.utils.rnn import pad_sequence
import random

class TextDataset(Dataset):
    def __init__(self, tokenized_dataset):
        super().__init__()
        # print(tokenized_dataset[0].keys())
        self.input_ids = [
            torch.tensor(x["input_ids"])
            for x in tokenized_dataset
            if len(x["input_ids"]) > 1
        ]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_ids = self.input_ids[index]
        x = input_ids[:-1].clone()
        y = input_ids[1:].clone()

        # print(len(x), len(y))
        return x, y


def tokenize_function(datapoints, config, tokenizer):
    return tokenizer(
        datapoints["text"], truncation=True, max_length=config["max_seq_len"]
    )


def collate_fn(batch):
    x, y = zip(*batch)
    x_padded = pad_sequence(x, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y, batch_first=True, padding_value=0)
    return x_padded, y_padded


def prepare_dataset(config):
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train[:5%]")
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2",
        model_max_length=config["max_seq_len"],
        # vocab_size=config["vocab_size"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    wrapped_tokenize_fn = partial(tokenize_function, tokenizer=tokenizer, config=config)

    tokenized_dataset = dataset.map(
        wrapped_tokenize_fn,
        batched=True,
        remove_columns=["text"],
    )
    text_dataset = TextDataset(tokenized_dataset)

    dataloader = DataLoader(
        text_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    print("LOG : dataset processing done successfully.")
    return dataloader, tokenizer

