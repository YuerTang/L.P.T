
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datasets import load_dataset

!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
 
class P_XGivenZ_Model(nn.Module):
    """
    p(x|z). A small MLP that outputs x from z.
    """
    def __init__(self, z_embed_dim, x_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, x_dim)
        )
    
    def forward(self, z_latent):
        """
        z_latent: (batch, seq_len_z, z_embed_dim)
        """
        batch_size, seq_len_z, embed_dim = z_latent.shape
        z_flat = z_latent.view(batch_size, seq_len_z * embed_dim)
        x_recon = self.fc(z_flat)  # (batch, x_dim)
        return x_recon

class SimpleLatentPipeline(nn.Module):
    """
    Demonstrates:
      1) Random init of z
      2) p(x|z)   -> P_XGivenZ_Model
      3) Langevin-based refinement of z (SGLD/ULD style), minimizing reconstruction loss.
      
    Removed the 'y' variable and all cross-attention code.
    """
    def __init__(self, x_dim, z_embed_dim):
        super().__init__()
        
        # We'll skip any posterior network entirely.
        self.z_embed_dim = z_embed_dim

        # MLP to reconstruct x from z
        self.px_model = P_XGivenZ_Model(z_embed_dim=z_embed_dim, x_dim=x_dim)
        
        # Langevin hyperparameters
        self.num_langevin_steps = 10
        self.langevin_step_size = 0.01
        self.langevin_noise_scale = 1.0

    def sample_z_langevin(self, x, num_steps=None, step_size=None, noise_scale=None, debug=False):
        """
        A basic Langevin sampler that starts z from random noise, 
        then iteratively refines it by minimizing MSE reconstruction of x.
        """
        if num_steps is None:
            num_steps = self.num_langevin_steps
        if step_size is None:
            step_size = self.langevin_step_size
        if noise_scale is None:
            noise_scale = self.langevin_noise_scale

        # (1) Initialize z from pure random noise
        batch_size = x.size(0)
        z_dim = self.z_embed_dim
        device = x.device

        z = torch.randn(batch_size, z_dim, device=device)

        # (2) Iterative Langevin updates
        z = z.detach().clone()
        z.requires_grad = True

        for i in range(num_steps):
            if z.grad is not None:
                z.grad.zero_()

            """Is z updated by the x? """
            # Example "energy": MSE reconstruction of x from z
            z_latent = z.unsqueeze(1)  # (batch, 1, z_dim)
            x_recon = self.px_model(z_latent)
            recon_loss = F.mse_loss(x_recon, x, reduction='sum')

            # Backprop
            recon_loss.backward()

            # Langevin update: z <- z - (step_size/2)*grad + noise
            with torch.no_grad():
                z = z - 0.5 * step_size * z.grad
                # Add noise
                z += noise_scale * torch.sqrt(torch.tensor(step_size)) * torch.randn_like(z)
                
                # Enable grad for next iteration
                z.requires_grad = True

            if debug:
                print(f"Langevin step {i}, loss={recon_loss.item():.4f}")

        return z.detach()

    def forward(self, x):
        """
        x: (batch, x_dim)
        Returns: x_recon
        """
        # 1) Sample z from random + Langevin refinement
        z = self.sample_z_langevin(x, debug=False)

        # 2) Reconstruct x from z
        z_latent = z.unsqueeze(1)          # (batch, 1, z_dim)
        x_recon = self.px_model(z_latent)  # (batch, x_dim)
        
        return x_recon


def train_example():
    x_dim = 10
    z_dim = 32
    batch_size = 8
    model = SimpleLatentPipeline(x_dim=x_dim, z_embed_dim=z_dim)

    # An optimizer for model.parameters()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Suppose we have some dummy dataset of x
    x_data = torch.randn(100, x_dim)

    # We'll do a few epochs
    n_epochs = 5
    for epoch in range(n_epochs):
        # Shuffle or batchify however you like
        idx_perm = torch.randperm(x_data.size(0))
        x_data = x_data[idx_perm]

        # We'll do small mini-batches
        for start_idx in range(0, x_data.size(0), batch_size):
            end_idx = start_idx + batch_size
            x_batch = x_data[start_idx:end_idx]

            x_recon = model(x_batch)

            loss_recon = F.mse_loss(x_recon, x_batch)

            optimizer.zero_grad()
            loss_recon.backward()
            optimizer.step()

        print(f"Epoch {epoch} done. (Last batch recon loss={loss_recon.item():.4f})")

    print("Training complete!")


if __name__ == "__main__":
    train_example()
