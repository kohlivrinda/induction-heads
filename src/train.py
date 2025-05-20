import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion,
    config,
    privacy_engine,
    scheduler,
    epoch,
):
    scaler = GradScaler() # for mixed precision training (AMP)
    total_loss = 0
    model.train()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    accum_steps = 8 #for gradient accumulation
    for batch, (x, y) in enumerate(dataloader):
        # print('------------------')
        # print(x.shape, y.shape)
        x, y = x.to(device), y.to(device)
    
        optimizer.zero_grad()
        with autocast(device_type="cuda"): #amp
            
            logits, _ = model(x)
            logits = logits.view(-1, config["vocab_size"]) #assuming  cross entropy
            y = y.reshape(-1)

            loss = criterion(logits, y)
            
        scaler.scale(loss).backward()

        if not privacy_engine:
            nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])

        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        if batch % 50 == 0:
            print(
                f"Epoch - {epoch} || Loss- {loss.item():.4f} || Batch - {batch}/{len(dataloader)}"
            )
        del x, y, loss, logits
        torch.cuda.empty_cache()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch-{epoch} || Loss - {avg_loss}")
    return avg_loss
