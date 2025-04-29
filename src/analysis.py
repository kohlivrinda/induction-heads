import torch
import matplotlib.pyplot as plt


def visualize_attention(model, tokenizer, text_input, config, device, layer=0, head=0):
    model.eval()

    tokens = tokenizer.encode(text_input)
    tokens = tokens[: config["max_seq_len"]]
    token_strs = [tokenizer.decode([t]) for t in tokens]

    x = torch.tensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        y_hat, attention_patterns = model(x, return_attention=True)
    attn_matx = attention_patterns[layer][0, head].cpu().numpy()

    plt.figure(figsize=(12, 10))
    plt.imshow(attn_matx, cmap="viridis")
    plt.colorbar()

    plt.xticks(range(len(token_strs)), token_strs, rotation=90)
    plt.yticks(range(len(token_strs)), token_strs)

    plt.title(f"Attention Pattern: Layer {layer}, Head {head}")
    plt.tight_layout()

    filename = f"../plots/attn/attn_layer{layer}_head{head}.png"
    plt.savefig(filename)

    return attn_matx
