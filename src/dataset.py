import os
import torch
from torch.utils.data import Dataset

def download_tinyshakespeare(data_dir="data"):
    """Downloads TinyShakespeare dataset if not already downloaded."""
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "input.txt")

    if not os.path.exists(file_path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Downloaded dataset to {file_path}")
    else:
        print(f"Dataset already exists at {file_path}")

    return file_path

def load_tinyshakespeare(file_path):
    """Loads text data from TinyShakespeare dataset."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

class TinyShakespeareDataset(Dataset):
    """Character-level dataset for Tiny Shakespeare text."""
    
    def __init__(self, file_path, seq_length=100):
        self.text = load_tinyshakespeare(file_path)
        self.vocab = sorted(set(self.text))  # Unique characters
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}
        self.seq_length = seq_length

        # Convert text to integer indices
        self.encoded_text = [self.char_to_idx[c] for c in self.text]

    def __len__(self):
        return len(self.encoded_text) - self.seq_length

    def __getitem__(self, idx):
        """Returns (input_sequence, target_sequence)"""
        input_seq = self.encoded_text[idx: idx + self.seq_length]
        target_seq = self.encoded_text[idx + 1: idx + self.seq_length + 1]

        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)
