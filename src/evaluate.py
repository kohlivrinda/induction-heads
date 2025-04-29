import torch
import torch.nn as nn
import torch.nn.functional as F


def evaluate(model, dataloader, config, device):
    model.eval()

    total_loss = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            logits = model(x)

            logits = logits.view(-1, config["vocab_size"])
            y = y.reshape(-1)

            loss = F.cross_entropy(logits, y)
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Evaluation | Avg loss: {avg_loss:.4f}")

        return avg_loss
