import torch
import torch.nn as nn
import torch.nn.functional as F

class P_XGivenZ_Model(nn.Module):
    def __init__(self, z_embed_dim, vocab_size, hidden_dim=256, num_heads=8):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)  # ✅ Fix: Use an embedding layer
        self.z_projection = nn.Linear(z_embed_dim, hidden_dim)  # ✅ Project latent thought vector to match hidden dim

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, dropout=0.1, activation="relu"
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.output_layer = nn.Linear(hidden_dim, vocab_size)  # ✅ Maps hidden state back to vocabulary logits

    def forward(self, x_tokens, z_latent): 
        

        batch_size, seq_len = x_tokens.shape  


        x_emb = self.token_embedding(x_tokens)  # (batch, seq_len, hidden_dim)

        z_latent = self.z_projection(z_latent)  # (batch, hidden_dim)
        z_latent = z_latent.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_dim)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(x_tokens.device)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))

        decoder_output = self.transformer_decoder(
            tgt=x_emb.permute(1, 0, 2),  # (seq_len, batch, hidden_dim)
            memory=z_latent.permute(1, 0, 2),  # (seq_len, batch, hidden_dim)
            tgt_mask=causal_mask
        )
        logits = self.output_layer(decoder_output.permute(1, 0, 2))  # (batch, seq_len, vocab_size)
        return logits



class SimpleLatentPipeline(nn.Module):
    """
    Uses Langevin Dynamics to refine z and generate sequences.
    """
    def __init__(self, vocab_size, z_embed_dim, hidden_dim=256, num_heads=8):
        super().__init__()
        self.z_embed_dim = z_embed_dim
        self.px_model = P_XGivenZ_Model(z_embed_dim, vocab_size, hidden_dim, num_heads)

    def sample_z_langevin(self, x, num_steps=10, step_size=0.01, noise_scale=1.0):
        
        batch_size = x.size(0)
        device = x.device
        z = torch.randn(batch_size, self.z_embed_dim, device=device)
        z.requires_grad = True

        for _ in range(num_steps):
            if z.grad is not None:
                z.grad.zero_()

            x_recon = self.px_model(x, z)  # Forward pass
            recon_loss = F.cross_entropy(x_recon.reshape(-1, x_recon.shape[-1]), x.reshape(-1).long())
            recon_loss.backward()

            with torch.no_grad():
                z = z - 0.5 * step_size * z.grad
                z += noise_scale * torch.sqrt(torch.tensor(step_size)) * torch.randn_like(z)
                z.requires_grad = True

        return z.detach()

    def forward(self, x):
        z = self.sample_z_langevin(x)
        x = x.long()
        logits = self.px_model(x, z)
        return logits
