import torch
import torch.nn as nn
import torch.nn.functional as F

class P_XGivenZ_Model(nn.Module):
    """
    p(x|z). A small MLP that outputs x from z.
    """
    def __init__(self, z_embed_dim, vocab_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_embed_dim, 256),  # Increased hidden dimension
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, vocab_size)  # Final output
        )


    def forward(self, z_latent):
        batch_size, seq_len_z, embed_dim = z_latent.shape  # Extract sequence length
        z_flat = z_latent.reshape(batch_size * seq_len_z, embed_dim)  # ✅ This works
        logits = self.fc(z_flat)  # (batch_size * seq_len_z, vocab_size)
        logits = logits.view(batch_size, seq_len_z, -1)  # Reshape back to (batch, seq_len, vocab_size)
        return logits


class SimpleLatentPipeline(nn.Module):
    """
    Uses Langevin Dynamics to refine z and generate sequences.
    """
    def __init__(self, vocab_size, z_embed_dim):
        super().__init__()
        self.z_embed_dim = z_embed_dim
        self.px_model = P_XGivenZ_Model(z_embed_dim, vocab_size)

    def sample_z_langevin(self, x, num_steps=10, step_size=0.01, noise_scale=1.0):
        batch_size = x.size(0)
        device = x.device
        z = torch.randn(batch_size, self.z_embed_dim, device=device)
        z.requires_grad = True

        for _ in range(num_steps):
            if z.grad is not None:
                z.grad.zero_()

            z_latent = z.unsqueeze(1)  # (batch, 1, z_dim)
            x_recon = self.px_model(z_latent)
            recon_loss = F.cross_entropy(x_recon.view(-1, x_recon.shape[-1]), x.view(-1)[: x_recon.shape[0]].long())
            recon_loss.backward()
            with torch.no_grad():
                z = z - 0.5 * step_size * z.grad
                z += noise_scale * torch.sqrt(torch.tensor(step_size)) * torch.randn_like(z)
                z.requires_grad = True

        return z.detach()

    def forward(self, x):
        z = self.sample_z_langevin(x)
        z_latent = z.unsqueeze(1).expand(-1, x.shape[1], -1)  # Expand to match seq_len
        logits = self.px_model(z_latent)
        return logits

