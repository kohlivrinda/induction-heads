from dataclasses import dataclass
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Any


class AttentionHead(nn.Module):
    """
    dot product attention for one head
    Args:
        emb_dim: embedding dimension
        head_dim: dimension of a singular attention head
    Input:
        x : batch_size, seq_len, emb_dim
    Output:
        attn: batch_size, seq_len, head_dim
    """

    def __init__(self, emb_dim, head_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.head_dim = head_dim

        self.k_proj = nn.Linear(self.emb_dim, self.head_dim, bias=False)
        self.q_proj = nn.Linear(self.emb_dim, self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.emb_dim, self.head_dim, bias=False)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

    def forward(self, x):
        k = self.k_proj(x)  # batch, seq_len, head_dim
        q = self.q_proj(x)
        v = self.v_proj(x)

        attn_scores = torch.bmm(q, k.transpose(1, 2)) / self.scale  # [B, seq, seq]
        attn_weights = F.softmax(attn_scores, -1)  # [B, seq, seq]
        attn = torch.bmm(attn_weights, v)  # [B, seq, seq] @ [B, seq, head]
        return attn


class VectorizedAttentionHead(nn.Module):
    """
    performs scaled dot product attention across multiple heads
    Args:
        emb_dim: embedding dimension
        num_heads: number of attention heads
        mask_input: True to apply causal_mask
        max_seq_len: maximum sequence length
        qk_attn: True if return attn pre val multiplication (to study qk circuit)
    Input:
        x : batch_size, seq_len, emb_dim
    Output:
        attn: batch_size, num_heads, seq_len, head_dim
    """

    def __init__(
        self, emb_dim, num_heads, max_seq_len, mask_input, qk_attn, dropout_ratio
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.mask_input = mask_input
        self.qk_attn = qk_attn
        self.dropout_ratio = dropout_ratio

        self.head_dim = self.emb_dim // self.num_heads
        self.k_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.q_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.v_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        self.dropout = nn.Dropout(self.dropout_ratio)
        causal_mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len))
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        k = self.k_proj(x)  # batch, seq_len, head_dim
        q = self.q_proj(x)
        v = self.v_proj(x)

        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        k = k.permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)  # [b, num_heads, seq, head_dim]

        attn_scores = (
            torch.matmul(q, k.transpose(-2, -1)) / self.scale
        )  # [B, num_heads, seq, seq]
        if self.mask_input:
            mask = self.causal_mask[:seq_len, :seq_len]
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, -1)  # [B, num_heads, seq, seq]
        attn_weights = self.dropout(attn_weights)
        attn = torch.matmul(
            attn_weights, v
        )  # [B, num_heads, seq, seq] @ [B, num_heads, seq, head] -> [B, num_heads, seq, head]
        if self.qk_attn:
            return attn_weights, attn
        return attn


class MultiHeadAttention(nn.Module):
    """
    implenets MHA, ie: combines outputs from multiple attention heads.
    Args:
        num_heads: number of attention heads
        emb_dim: embedding dimensions
        mask_input: True to apply causal_mask
        max_seq_len: maximum sequence length
    Input:
        x: [batch_size, seq_len, emb_dim]
    Output:
        attn: [batch_size, seq_len, emb_dim]

    """

    def __post_init__(
        self, num_heads, emb_dim, mask_input, max_seq_len, qk_attn, dropout_ratio
    ):
        super().__init__()
        self.num_heads = num_heads
        self.mask_input = mask_input
        self.max_seq_len = max_seq_len
        self.emb_dim = emb_dim
        self.dropout_ratio = dropout_ratio
        self.qk_attn = qk_attn

        self.attention_head = VectorizedAttentionHead(
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            max_seq_len=self.max_seq_len,
            mask_input=self.mask_input,
            qk_attn=self.qk_attn,
            dropout_ratio=self.dropout_ratio,
        )
        self.linear_layer = nn.Linear(self.emb_dim, self.emb_dim, bias=True)

    def forward(self, x):
        batch_size, seq_len, emb_dim = x.shape

        if self.qk_attn:
            qk_attn, attn = self.attention_head(x)
        else:
            attn = self.attention_head(x)
        attn = attn.permute(0, 2, 1, 3)
        attn = attn.view(batch_size, seq_len, emb_dim)
        attn = self.linear_layer(attn)

        return (qk_attn, attn) if self.qk_attn else attn


class FFN(nn.Module):
    """
    2 layer feedforward network with gelu activation ez.
    Input:
        x: batch_size, seq_len, emb_dim
    Output:
        x: batch_size, seq_len, emb_dim
    """

    def __init__(self, input_dim, output_dim, hidden_dim, dropout_ratio):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias=True)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class Block(nn.Module):
    """
    Transformer block (pre-norm) with layernorm, MHA and FFN

    Input:
        x: batch_size, seq_len, emb_dim
    Output:
        res_stream: batch_size, seq_len, emb_dim
    """

    num_heads: int
    emb_dim: int
    hidden_dim: int
    mask_input: bool
    max_seq_len: int
    qk_attn: bool
    dropout_ratio: float

    def __post_init__(
        self,
        num_heads,
        emb_dim,
        hidden_dim,
        mask_input,
        max_seq_len,
        qk_attn,
        dropout_ratio,
    ):
        super().__init__()

        self.qk_attn = qk_attn
        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            emb_dim=emb_dim,
            mask_input=mask_input,
            max_seq_len=max_seq_len,
            qk_attn=qk_attn,
            dropout_ratio=dropout_ratio,
        )

        self.ffn = FFN(
            input_dim=emb_dim,
            hidden_dim=hidden_dim,
            output_dim=emb_dim,
            dropout_ratio=dropout_ratio,
        )
        self.lnorm = nn.LayerNorm(normalized_shape=emb_dim)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        res_stream = x
        x = self.lnorm(x)  # Pre-norm transformer

        if self.qk_attn:
            qk_attn, attn = self.mha(x)
        else:
            attn = self.mha(x)
        res_stream = res_stream + self.dropout(attn)

        y = self.lnorm(res_stream)
        ffn_out = self.ffn(y)
        res_stream = res_stream + self.dropout(ffn_out)

        return (res_stream, qk_attn) if self.qk_attn else res_stream


class Transformer(nn.Module):
    """
    Entire transformer model (pre-norm style)
    x -> token embeddings + pos embeddings -> attn blocks -> unembedding layer (logits)

    Input:
        x: batch_size, seq_len
    Output:
        logits: batch_size, vocab_size
    """

    def __init__(self, config):
        super().__init__()

        for key, value in self.config.items():
            setattr(self, key, value)

        self.embeddings = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.emb_dim
        )
        self.positional_embeddings = nn.Embedding(
            num_embeddings=self.max_seq_len, embedding_dim=self.emb_dim
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    num_heads=self.num_heads,
                    emb_dim=self.emb_dim,
                    hidden_dim=self.hidden_dim,
                    max_seq_len=self.max_seq_len,
                    mask_input=self.mask_input,
                    qk_attn=self.qk_attn,
                    dropout_ratio=self.dropout_ratio,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.lnorm = nn.LayerNorm(self.emb_dim)
        self.output_layer = nn.Linear(self.emb_dim, self.vocab_size)
        self.apply(self.__init_weights)
        self.dropout = nn.Dropout(self.dropout_ratio)

    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=self.emb_dim**-0.5)
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        _, seq_len = x.shape
        token_emb = self.embeddings(x)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.positional_embeddings(pos)

        x = token_emb + pos_emb
        x = self.dropout(x)
        attentions = []
        for block in self.blocks:
            if self.qk_attn:
                x, attn = block(x)
                attentions.append(attn)
            else:
                x = block(x)

        x = self.lnorm(x)  # pre-norm transformer architecture
        logits = self.output_layer(x)
        return (logits, attentions) if self.qk_attn else logits

    def get_attention_patterns(self, x):
        og_qk_attn = self.qk_attn
        with torch.no_grad():
            self.qk_attn = True
            logits, attentions = self.forward(x)
        y_hat = nn.Softmax(logits, dim=-1)
        self.qk_attn = og_qk_attn
        return y_hat, attentions
