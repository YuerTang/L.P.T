import torch
import torch.nn as nn
import torch.nn.functional as F

class PureTransformerBaseline(nn.Module):
    """
    A simple 'decoder-only' Transformer baseline for next-token prediction.
    Uses 4 layers, 8 attention heads, and hidden_dim=256 by default.
    """
    def __init__(self, vocab_size, hidden_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

        # We can use nn.TransformerEncoder as a "decoder-only" approach by passing
        # a causal mask. Each layer has self-attention + feedforward.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x_tokens):
        """
        x_tokens: LongTensor of shape (batch_size, seq_len)
        Returns logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x_tokens.size()

        # Embed the tokens
        x_emb = self.token_embedding(x_tokens)          # (batch_size, seq_len, hidden_dim)
        x_emb = x_emb.permute(1, 0, 2)                  # (seq_len, batch_size, hidden_dim)

        # Build a causal mask (upper triangle = -inf) so each token only sees the past
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x_tokens.device), 
            diagonal=1
        ).bool()
        causal_mask = causal_mask.masked_fill(causal_mask, float('-inf'))
        
        # Pass through the Transformer encoder (really a "decoder-only" approach).
        encoded = self.transformer_encoder(x_emb, mask=causal_mask)  # (seq_len, batch_size, hidden_dim)

        # Project back to vocab logits
        encoded = encoded.permute(1, 0, 2)              # (batch_size, seq_len, hidden_dim)
        logits = self.output_layer(encoded)             # (batch_size, seq_len, vocab_size)

        return logits
