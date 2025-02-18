import torch
from torch.utils.data import DataLoader, TensorDataset
import wandb
from train import train, eval
from LTM_paper import SimpleLatentPipeline
from dataset import load_tinyshakespeare

# ✅ Initialize WandB
wandb.init(project="LTM", entity="yuertang", name="train_eval_run")

# ✅ Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Load dataset
file_path = "data/input.txt"
text = load_tinyshakespeare(file_path)

# ✅ Tokenization
def build_vocab(text):
    vocab = sorted(set(text))
    char2idx = {ch: idx for idx, ch in enumerate(vocab)}
    return vocab, char2idx

def encode_text(text, char2idx):
    return torch.tensor([char2idx[c] for c in text], dtype=torch.long)

vocab, char2idx = build_vocab(text)
encoded_text = encode_text(text, char2idx)
vocab_size = len(vocab)

# ✅ Create data for training, validation, and testing
seq_len = 50
batch_size = 32

def create_dataloader(data, seq_len, batch_size):
    sequences = [data[i:i + seq_len + 1] for i in range(len(data) - seq_len)]
    dataset = TensorDataset(torch.stack(sequences))  # Convert to PyTorch dataset
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_dataloader = create_dataloader(encoded_text[:8000], seq_len, batch_size)
val_dataloader = create_dataloader(encoded_text[8000:9000], seq_len, batch_size)
test_dataloader = create_dataloader(encoded_text[9000:10000], seq_len, batch_size)

# ✅ Initialize model
z_dim = 32
model = SimpleLatentPipeline(vocab_size, z_dim).to(device)

# ✅ Training configuration
config = {
    "learning_rate": 1e-4,
    "epochs": 5,
    "batch_size": batch_size
}

# ✅ Train model (save best weights)
best_model = train(
    model=model,
    device=device,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    config=config
)

# ✅ Load best model and evaluate on test set
model.load_state_dict(torch.load("best_model.pth"))
eval(model=model, device=device, val_dataloader=test_dataloader)
