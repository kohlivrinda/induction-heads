from torch.utils.data import DataLoader, Dataset
import torch
from datasets import load_dataset
from transformers import AutoTokenizer


class TextDataset(Dataset):
    def __init__(self, tokenized_dataset):
        super().__init__()
        self.input_ids = [torch.tensor(x) for x in tokenized_dataset]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_ids = self.input_ids(index)
        x = input_ids[:-1].clone()
        y = input_ids[1:].clone()
        return x, y


def tokenize_function(tokenizer, datapoints, config):
    return tokenizer(
        datapoints["text"], truncation=True, max_length=config["max_seq_len"]
    )


def prepare_dataset(config):
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train[:5%]")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2", model_max_length=config["max_seq_len"], vocab_size=config["vocab_size"]
    )

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    text_dataset = TextDataset(tokenized_dataset)

    dataloader = DataLoader(
        text_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )
    return dataloader, tokenizer
