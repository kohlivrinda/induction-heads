Plan of Action:
    - build 2 layer transformer
        - attention block ✅
        - multihead attention ✅
        - feedforward layer ✅
        - putting it all together ✅
        - causal masking ✅
        - try outt RoPE???? -> if i do, make it an optional parameter
        - add dropouts ✅
        - attention extraction ✅
        - weights initialisation ✅
        - tie embeddigs ???
    - implement dp training with opacus
        - prepare dataset
        - train loop
        - metrics
    - setup circuit analysis tools
    - add observability

Layer | Initialization | Why
Linear layers (nn.Linear) | Xavier Uniform (glorot_uniform) | Balances input/output variance; good for activations like GELU.
Embedding layers (nn.Embedding) | Normal(0, 1/sqrt(emb_dim)) | Prevents large early logits.
LayerNorm (nn.LayerNorm) | Weight = 1, Bias = 0 | Standard for stability.
Output layer | Zero init bias, Xavier Uniform for weights | Predict logits safely.