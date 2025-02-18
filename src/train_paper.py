import torch
import torch.optim as optim
import torch.nn.functional as F
from dataset import load_tinyshakespeare
from LTM_paper import SimpleLatentPipeline
import wandb

def train(model, device, train_dataloader, val_dataloader, config):
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    best_loss = float("inf")
    best_model = None

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for x_batch in train_dataloader:
            x_batch = x_batch.to(device)

            # Create input-target pairs for autoregressive decoding
            x_input = x_batch[:, :-1]
            x_target = x_batch[:, 1:]

            optimizer.zero_grad()
            logits = model(x_input)

            # Loss calculation
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x_target.reshape(-1), ignore_index=-100)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        wandb.log({"train_loss": avg_loss, "epoch": epoch + 1})

        # Evaluate model on validation data
        val_loss = eval(model, device, val_dataloader, log_wandb=True)
        
        # Save best model based on validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            torch.save(best_model, "best_model.pth")
            print(f"Best model saved at epoch {epoch+1}")

    return best_model  # Return the best model weights


def eval(model, device, val_dataloader, log_wandb=False):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for x_batch in val_dataloader:
            x_batch = x_batch.to(device)

            # Create input-target pairs
            x_input = x_batch[:, :-1]
            x_target = x_batch[:, 1:]

            logits = model(x_input)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x_target.reshape(-1), ignore_index=-100)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"Validation Loss: {avg_loss:.4f}")

    if log_wandb:
        wandb.log({"val_loss": avg_loss})

    return avg_loss  # Return validation loss
