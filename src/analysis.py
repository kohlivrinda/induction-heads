import torch
import matplotlib.pyplot as plt
import os


def visualize_attention(model, x_input, config, device, layer=0, head=0):
    model.eval()

    token_ids = x_input.squeeze(0).tolist()
    token_strs = [f"t{tok}" for tok in token_ids] 

    with torch.no_grad():
        _, attention_patterns = model(x_input)
    attn_matx = attention_patterns[layer][0, head].cpu().numpy()

    plt.figure(figsize=(12, 10))
    plt.imshow(attn_matx, cmap="viridis")
    plt.colorbar()

    plt.xticks(range(len(token_strs)), token_strs, rotation=90)
    plt.yticks(range(len(token_strs)), token_strs)

    plt.title(f"Attention Pattern: Layer {layer}, Head {head}")
    plt.tight_layout()

    output_dir = "../plots/attn"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{output_dir}/attn_layer{layer}_head{head}.png" #TODO: add epoch level saving
    plt.savefig(filename)

    return attn_matx
