import torch
import torch.nn as nn
import torch.nn.functional as F

class P_XGivenZ_Model(nn.Module):
    def __init__(self, vocab_size, z_embed_dim, hidden_dim=256, num_heads=8, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads), num_layers=num_layers
        )
        self.out_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, z):
        x_emb = self.embedding(x)
        z = z.unsqueeze(1)  # Shape: [batch_size, 1, z_dim]
        z = z.expand(-1, x_emb.size(1), -1)  # [batch_size, seq_len, z_dim]
        tgt = x_emb + z
        out = self.transformer(tgt, tgt)
        logits = self.out_proj(out)
        return logits


class LangevinTransformerModel(nn.Module):
    """
    Model that uses Langevin Dynamics to refine z and generate sequences.
    """
    def __init__(self, vocab_size, z_embed_dim, hidden_dim=256, num_heads=8, num_layers=1):
        super().__init__()
        self.z_embed_dim = z_embed_dim
        self.px_model = P_XGivenZ_Model(vocab_size, z_embed_dim, hidden_dim, num_heads, num_layers)

    def sample_z_langevin(self, x, num_steps=10, step_size=0.01, noise_scale=1.0):
        batch_size = x.size(0)
        device = x.device
        z = torch.randn(batch_size, self.z_embed_dim, device=device, requires_grad=True)

        final_recon_loss = None

        for _ in range(num_steps):
            x_recon = self.px_model(x, z)

            recon_loss = F.cross_entropy(
                x_recon.reshape(-1, x_recon.shape[-1]),
                x.reshape(-1).long()
            )
            final_recon_loss = recon_loss  # Save last recon_loss to return

            # Compute gradient manually
            grad = torch.autograd.grad(recon_loss, z, retain_graph=True)[0]

            # Langevin update step
            noise = noise_scale * torch.sqrt(torch.tensor(step_size, device=device)) * torch.randn_like(z)
            z = z - 0.5 * step_size * grad + noise
            z.requires_grad_(True)

        return z, final_recon_loss

    def forward(self, x):
        z, recon_loss = self.sample_z_langevin(x)
        logits = self.px_model(x.long(), z)
        return logits, recon_loss
