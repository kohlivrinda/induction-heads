from dataclasses import dataclass
import torch.nn as nn
import torch
import torch.functional as F

@dataclass
class AttentionHead(nn.Module):
    emb_dim: int
    head_dim: int # d_k

    def __post_init__(self):
        super().__init__()
        self.k_proj = nn.Linear(self.emb_dim, self.head_dim, bias=False)
        self.q_proj = nn.Linear(self.emb_dim, self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.emb_dim, self.head_dim, bias=False)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

    def forward(self, x):
        """
        Args:
            x : batch_size, seq_len, emb_dim
        Output:
            output: batch_size, seq_len, head_dim

        """
        k = self.k_proj(x) # batch, seq_len, head_dim
        q = self.q_proj(x)
        v = self.v_proj(x)

        attn_scores = torch.bmm(q, k.transpose(1, 2)) / self.scale # [B, seq, seq]
        attn_weights = F.softmax(attn_scores, -1) # [B, seq, seq]
        output = torch.bmm(attn_weights, v) # [B, seq, seq] @ [B, seq, head]
        return output



    pass

@dataclass
class MultiHeadAttention:
    pass

@dataclass
class FNN:
    pass

@dataclass
class Block:
    batch_size: int
    seq_len: int

    pass

@dataclass
class Transformer:
    num_layers: int # blocks
    pass
