from dataclasses import dataclass
import torch.nn as nn
import torch
import torch.functional as F

@dataclass
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
    emb_dim: int
    head_dim: int # d_k

    def __post_init__(self):
        super().__init__()
        self.k_proj = nn.Linear(self.emb_dim, self.head_dim, bias=False)
        self.q_proj = nn.Linear(self.emb_dim, self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.emb_dim, self.head_dim, bias=False)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

    def forward(self, x):
        k = self.k_proj(x) # batch, seq_len, head_dim
        q = self.q_proj(x)
        v = self.v_proj(x)

        attn_scores = torch.bmm(q, k.transpose(1, 2)) / self.scale # [B, seq, seq]
        attn_weights = F.softmax(attn_scores, -1) # [B, seq, seq]
        attn = torch.bmm(attn_weights, v) # [B, seq, seq] @ [B, seq, head]
        return attn
    
@dataclass
class VectorizedAttentionHead(nn.Module):
    """
    performs scaled dot product attention across multiple heads
    Args:
        emb_dim: embedding dimension
        num_heads: number of attention heads
    Input:
        x : batch_size, seq_len, emb_dim
    Output:
        attn: batch_size, num_heads, seq_len, head_dim 
    """
    emb_dim: int
    num_heads: int # d_k
    mask_input: bool
    max_seq_len: bool

    def __post_init__(self):
        super().__init__()

        self.head_dim = self.emb_dim // self.num_heads
        self.k_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.q_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.v_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        causal_mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len))
        self.register_buffer("causal_mask", causal_mask)


    def forward(self, x):
        batch_size , seq_len, _ = x.shape
        k = self.k_proj(x) # batch, seq_len, head_dim
        q = self.q_proj(x)
        v = self.v_proj(x)

        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        k = k.permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3) # [b, num_heads, seq, head_dim]


        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale # [B, num_heads, seq, seq]
        if self.mask_input:
            mask = self.causal_mask[:seq_len, :seq_len]
            attn_scores = attn_scores.masked_fill(mask==0, float("-inf"))


        attn_weights = F.softmax(attn_scores, -1) # [B, num_heads, seq, seq]
        attn = torch.matmul(attn_weights, v) # [B, num_heads, seq, seq] @ [B, num_heads, seq, head] -> [B, num_heads, seq, head]
        return attn


@dataclass
class MultiHeadAttention(nn.Module):
    """ 
    implenets MHA, ie: combines outputs from multiple attention heads.
    Args:
        num_heads: number of attention heads
        emb_dim: embedding dimensions
    Input:
        x: [batch_size, seq_len, emb_dim]
    Output:
        attn: [batch_size, seq_len, emb_dim] 

    """
    num_heads: int
    emb_dim: int
    mask_input: bool
    max_seq_len: int

    def __post_init__(self):
        super().__init__()

        self.attention_head = VectorizedAttentionHead(emb_dim = self.emb_dim, 
                                                      num_heads= self.num_heads,
                                                      max_seq_len= self.max_seq_len,
                                                      mask_input= self.mask_input)
        self.linear_layer = nn.Linear(self.emb_dim, self.emb_dim, bias=True)

    def forward(self, x):
        batch_size, seq_len, emb_dim = x.shape

        attn = self.attention_head(x)
        attn = attn.permute(0, 2, 1, 3)
        attn = attn.view(batch_size, seq_len, emb_dim)
        attn = self.linear_layer(attn)
        return attn

@dataclass
class FFN(nn.Module):
    input_dim: int 
    hidden_dim: int
    output_dim: int

    def __post_init__(self):
        super().__init__()
        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.linear2 = nn.Linear(self.hidden_dim, self.output_dim, bias=True)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

@dataclass
class Block(nn.Module):
    num_heads: int
    emb_dim : int
    hidden_dim: int
    mask_input:bool
    max_seq_len:int

    def __post_init__(self):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads= self.num_heads,
                                      emb_dim= self.emb_dim,
                                      mask_input=self.mask_input,
                                      max_seq_len=self.max_seq_len
                                      )
        
        self.ffn = FFN(input_dim= self.emb_dim, 
                       hidden_dim= self.hidden_dim,
                       output_dim= self.emb_dim)
        self.lnorm = nn.LayerNorm(normalized_shape=self.emb_dim)

    def forward(self, x):
        res_stream = x
        x = self.lnorm(x) # Pre-norm transformer
        attn = self.mha(x)
        res_stream = res_stream + attn

        y = self.lnorm(res_stream)
        ffn_out = self.ffn(y)
        res_stream = res_stream + ffn_out

        return res_stream 


@dataclass
class Transformer(nn.Module):
    num_layers: int # blocks
    num_heads: int
    hidden_dim: int
    emb_dim: int
    vocab_size: int
    max_seq_len: int
    mask_input: bool

    def __post_init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim= self.emb_dim)
        self.positional_embeddings = nn.Embedding(num_embeddings=self.max_seq_len, embedding_dim=self.emb_dim)
        self.blocks = nn.ModuleList([
            Block(num_heads= self.num_heads,
                  emb_dim= self.emb_dim,
                  hidden_dim= self.hidden_dim,
                  max_seq_len= self.max_seq_len,
                  mask_input= self.mask_input
                  ) for _ in range(self.num_layers)
        ])
        self.lnorm = nn.LayerNorm(self.emb_dim)
        self.output_layer = nn.Linear(self.emb_dim, self.vocab_size)
    
    def forward(self, x):
        _, seq_len = x.shape
        token_emb = self.embeddings(x)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.positional_embeddings(pos)

        x = token_emb + pos_emb

        for block in self.blocks:
            x = block(x)
        
        x = self.lnorm(x) # pre-norm transformer architecture
        logits = self.output_layer(x)
        return logits


        