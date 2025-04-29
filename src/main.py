import torch
import torch.nn as nn
import torch.nn.functional as F


import utils.file_utils as futils
from dataset import prepare_dataset
from train import train_epoch
from analysis import visualize_attention
from model import Transformer


def run_experiment(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader, tokenizer = prepare_dataset(config)
    model = Transformer(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )

    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            config=config,
            privacy_engine=None,
            scheduler=scheduler,
        )

        if epoch % 3 == 0:
            sample_text = "The president of the United States is the head of the state. The president leads the executive branch."

            for layer in range(config["n_layers"]):
                visualize_attention(model, tokenizer, sample_text, layer=layer, head=1)

    return model, tokenizer


def main():
    config = futils.load_config()
    run_experiment(config)


if __name__ == "__main__":
    main()
