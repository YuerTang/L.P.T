import torch
from LTM import SimpleLatentPipeline
from dataset import load_tinyshakespeare, build_vocab

def generate_text(model, idx2char, start_token, length=100):
    model.eval()
    z = torch.randn(1, model.z_embed_dim)
    generated_text = [start_token]

    for _ in range(length):
        z_latent = z.unsqueeze(1)
        logits = model.px_model(z_latent)
        next_char_idx = torch.argmax(logits, dim=-1).item()
        generated_text.append(idx2char[next_char_idx])

    return "".join(generated_text)

if __name__ == "__main__":
    # Load trained model
    model = SimpleLatentPipeline(vocab_size=65, z_embed_dim=32)
    model.load_state_dict(torch.load("trained_model.pth"))  # Load trained weights

    _, _, idx2char = build_vocab(load_tinyshakespeare("data/input.txt"))
    print(generate_text(model, idx2char, start_token="H"))
