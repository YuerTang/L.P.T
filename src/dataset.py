import os
import torch
from torch.utils.data import Dataset

def load_tinyshakespeare(file_path):
    """Loads text data from TinyShakespeare dataset."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

class TinyShakespeareDataset(Dataset):
    """Character-level dataset for Tiny Shakespeare text."""
    
    def __init__(self, file_path, seq_length=100):
        self.text = load_tinyshakespeare(file_path)
        self.vocab = sorted(set(self.text))  
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
