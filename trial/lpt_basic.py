import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# -------------------------------
# CrossAttentionTransformer
# -------------------------------
class CrossAttentionTransformer(nn.Module):
    """
    A very simplified Transformer block that treats x as the main sequence
    and attends to z (latent plan) via cross-attention.
    """
    def __init__(self, x_embed_dim, z_embed_dim, hidden_dim=128, num_heads=4):
        super().__init__()
        self.x_proj = nn.Linear(x_embed_dim, hidden_dim)
        self.z_proj = nn.Linear(z_embed_dim, hidden_dim)
        
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        self.layernorm3 = nn.LayerNorm(hidden_dim)
        
        self.output_layer = nn.Linear(hidden_dim, 1)  
        
    def forward(self, x_tokens, z_latent):
        """
        x_tokens: (batch, seq_len_x, x_embed_dim)
        z_latent: (batch, seq_len_z, z_embed_dim)

        Returns: y_pred (batch, seq_len_x, 1)
        """
        # 1) Project into hidden_dim
        x_h = self.x_proj(x_tokens)  # (batch, seq_len_x, hidden_dim)
        z_h = self.z_proj(z_latent)  # (batch, seq_len_z, hidden_dim)
        
        # 2) Self-attention over x
        x_h_t = x_h.transpose(0, 1)  # (seq_len_x, batch, hidden_dim)
        x_sa, _ = self.self_attn(x_h_t, x_h_t, x_h_t)
        x_h_sa = self.layernorm1(x_h_t + x_sa)  # residual
        
        # 3) Cross-attention: queries = x, keys/values = z
        z_h_t = z_h.transpose(0, 1)
        x_ca, _ = self.cross_attn(x_h_sa, z_h_t, z_h_t)
        x_h_ca = self.layernorm2(x_h_sa + x_ca)  # residual
        
        # 4) Feed-forward
        ff = self.ffn(x_h_ca)
        x_out = self.layernorm3(x_h_ca + ff)  # residual
        
        # 5) Predict y
        x_out_t = x_out.transpose(0, 1)  # (batch, seq_len_x, hidden_dim)
        y_pred = self.output_layer(x_out_t)  # (batch, seq_len_x, 1)
        return y_pred

# -------------------------------
# P_XGivenZ_Model
# -------------------------------
class P_XGivenZ_Model(nn.Module):
    """
    P(x|z). A small MLP that outputs x from z.
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

# -------------------------------
# SimpleLatentPipeline (No PosteriorNetwork)
# -------------------------------
class SimpleLatentPipeline(nn.Module):
    """
    Demonstrates:
      1) Random init of z
      2) p(x|z)   -> P_XGivenZ_Model
      3) p(y|x,z) -> CrossAttentionTransformer
      4) Langevin-based refinement of z (SGLD/ULD style)
    """
    def __init__(self, x_dim, z_embed_dim, x_embed_dim=64):
        super().__init__()
        
        # We'll skip any posterior network entirely.
        self.z_embed_dim = z_embed_dim

        # MLP to reconstruct x from z
        self.px_model = P_XGivenZ_Model(z_embed_dim=z_embed_dim, x_dim=x_dim)
        
        # Cross-attention to predict y from (x,z)
        self.cross_attn_transformer = CrossAttentionTransformer(
            x_embed_dim=x_embed_dim,
            z_embed_dim=z_embed_dim,
            hidden_dim=128,
            num_heads=4
        )
        
        self.x_embedding = nn.Linear(x_dim, x_embed_dim)
        
        # Langevin hyperparameters
        self.num_langevin_steps = 10
        self.langevin_step_size = 0.01
        self.langevin_noise_scale = 1.0

    def sample_z_langevin(self, x, y=None, num_steps=None, step_size=None, noise_scale=None, debug=False):
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

    def forward(self, x, y=None):
        """
        x: (batch, x_dim)
        y: (batch, 1, 1) optional, if you want to compare to some target
        """
        # 1) Sample z from random + Langevin refinement
        z = self.sample_z_langevin(x, y, debug=False)

        # 2) Reconstruct x from z
        z_latent = z.unsqueeze(1)          # (batch, 1, z_dim)
        x_recon = self.px_model(z_latent)  # (batch, x_dim)
        
        # 3) Predict y with cross-attention
        x_tokens = self.x_embedding(x).unsqueeze(1)  # (batch, 1, x_embed_dim)
        y_pred = self.cross_attn_transformer(x_tokens, z_latent)  # (batch, 1, 1)
        
        return x_recon, y_pred


def train_example():
    x_dim = 10
    z_dim = 32
    batch_size = 8
    model = SimpleLatentPipeline(x_dim=x_dim, z_embed_dim=z_dim, x_embed_dim=64)

    # An optimizer for model.parameters()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Suppose we have some dummy dataset of (x, y) pairs
    # In a real scenario, you'd have a DataLoader etc.
    x_data = torch.randn(100, x_dim)
    y_data = torch.randn(100, 1, 1)  # shape (N, 1, 1)

    # We'll do a few epochs over this mini dataset
    n_epochs = 5
    for epoch in range(n_epochs):
        # Shuffle or batchify however you like; here's a small example
        idx_perm = torch.randperm(x_data.size(0))
        x_data = x_data[idx_perm]
        y_data = y_data[idx_perm]

        # We'll do small mini-batches of size 8
        for start_idx in range(0, x_data.size(0), batch_size):
            end_idx = start_idx + batch_size
            x_batch = x_data[start_idx:end_idx]
            y_batch = y_data[start_idx:end_idx]

            # ============ Forward Pass ============
            x_recon, y_pred = model(x_batch, y_batch)

            # ============ Define Loss ============
            # For example: reconstruction + MSE on y
            loss_recon = F.mse_loss(x_recon, x_batch)
            loss_y = F.mse_loss(y_pred, y_batch)
            loss = loss_recon + loss_y

            # ============ Backward + Optimize ============
            optimizer.zero_grad()   # clear previous grads
            loss.backward()         # accumulates grads in model parameters
            optimizer.step()        # update model weights

        print(f"Epoch {epoch} done. (Last batch loss={loss.item():.4f})")

    print("Training complete!")

if __name__ == "__main__":
    train_example()
