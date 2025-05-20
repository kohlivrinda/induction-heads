from torch.utils.data import DataLoader, Dataset
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from functools import partial
from torch.nn.utils.rnn import pad_sequence
import random


class InductionDataset(Dataset):
    def __init__(self, num_samples = 10000, seq_len = 40, vocab_size = 100):
        self.samples = []
        for _ in range(num_samples):
            half_len = seq_len // 2
            prefix = random.choice(range(vocab_size), k=half_len)
            full_seq = prefix + prefix
            self.samples.append(full_seq)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx][:-1])
        y = torch.tensor(self.samples[idx][1:])
        return x, y
    

def create_dataloader(config):
    dataset = InductionDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    print("LOG : dataset processing done successfully.")
    return dataloader


def generate_induction_sample(seq_len=40, vocab_size=100):
    half_len = seq_len // 2
    prefix = random.choices(range(vocab_size), k=half_len)
    full_seq = prefix + prefix
    return torch.tensor(full_seq[:seq_len], dtype=torch.long)  # crop if needed
