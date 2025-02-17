import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from dataset import load_tinyshakespeare
from LTM_paper import SimpleLatentPipeline
import sys
import os
import wandb


# Get the root directory and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


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
    file_path = "data/input.txt" 
    text = load_tinyshakespeare(file_path)
    
    # Tokenization
    vocab, char2idx, idx2char = build_vocab(text)
    x_data = encode_text(text[:10000], char2idx)  # Take first 10k characters
    vocab_size = len(vocab)

    # Hyperparameters
    z_dim = 32
    batch_size = 32
    seq_len = 50  # Use small sequences for training
    lr = 1e-4
    n_epochs = 5

    wandb.init(
        project="LTM", 
        entity="yuertang",
        name=f"training_run_{int(time.time())}",  # Generates unique run names
        config={
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": n_epochs,
            "seq_len": seq_len,
            "z_dim": z_dim,
        },
    )

    # Initialize model and move to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleLatentPipeline(vocab_size, z_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(n_epochs):
        epoch_loss = 0
        num_batches = 0

        for i in range(0, len(x_data) - seq_len, batch_size):
            x_batch = [x_data[i:i+seq_len] for i in range(i, min(i+batch_size, len(x_data) - seq_len))]
            x_batch = [torch.clone(seq).detach().to(torch.long) for seq in x_batch]
            x_batch = torch.nn.utils.rnn.pad_sequence(x_batch, batch_first=True, padding_value=0).to(device)

            logits = model(x_batch)
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), x_batch.view(-1), ignore_index=0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        wandb.log({ "loss": avg_loss, "epoch": epoch+1 })


    model_save_path = "trained_model.pth"
    torch.save(model.state_dict(), model_save_path)
    wandb.save(model_save_path)  # Upload to WandB
    print(f"Model saved at {model_save_path}")

if __name__ == "__main__":
    train()
 