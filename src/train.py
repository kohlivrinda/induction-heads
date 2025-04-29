import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train(model:nn.Module, dataloader:DataLoader, optimizer:optim.Optimizer, criterion, config, privacy_engine, scheduler, epoch):
    total_loss = 0
    model.train()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        logits = logits.view(-1, config["vocab_size"])
        y = y.reshape(-1)

        loss = criterion(logits, y)
        loss.backward()

        if not privacy_engine:
            nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()

        if batch % 50 == 0:
            print(f"Epoch - {epoch} || Loss- {loss.item():.4f} || Batch - {batch}/{len(dataloader)}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch-{epoch} || Loss - {avg_loss}")
    return avg_loss


