import torch
import torch.optim as optim
from dataset import download_tinyshakespeare, load_tinyshakespeare
from LTM import SimpleLatentPipeline

# Tokenize text data
def build_vocab(text):
    vocab = sorted(set(text))
    char2idx = {ch: idx for idx, ch in enumerate(vocab)}
    idx2char = {idx: ch for ch, idx in char2idx.items()}
    return vocab, char2idx, idx2char

def encode_text(text, char2idx):
    return torch.tensor([char2idx[c] for c in text], dtype=torch.long)

def train():
    # Load dataset
    file_path = download_tinyshakespeare()
    text = load_tinyshakespeare(file_path)
    
    # Tokenization
    vocab, char2idx, idx2char = build_vocab(text)
    x_data = encode_text(text[:10000], char2idx)  # Take first 10k characters
    vocab_size = len(vocab)

    # Hyperparameters
    z_dim = 32
    batch_size = 32
    seq_len = 50  # Use small sequences for training
    lr = 1e-3
    n_epochs = 5

    # Initialize model
    model = SimpleLatentPipeline(vocab_size, z_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(n_epochs):
        for i in range(0, len(x_data) - seq_len, batch_size):
            x_batch = [x_data[i:i+seq_len] for i in range(i, i+batch_size)]
            x_batch = torch.stack(x_batch)  # (batch_size, seq_len)

            logits = model(x_batch)  # Predicted logits
            loss = F.cross_entropy(logits.view(-1, vocab_size), x_batch.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
