import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

def train(model, device, train_dataloader, val_dataloader, config):
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    best_loss = float("inf")

    wandb.watch(model, log="all")

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0
        num_batches = 0

        print(f"🚀 Epoch {epoch+1}/{config['epochs']} - Training Started")

        for batch_idx, (x_batch, y_batch) in enumerate(train_dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            logits, recon_loss = model(x_batch)

            ce_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y_batch.reshape(-1),
                ignore_index=-100
            )

            loss = ce_loss + config.get("recon_loss_weight", 0.1) * recon_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                print(f"✅ Batch {batch_idx+1} - Loss: {loss.item():.4f}")
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_ce_loss": ce_loss.item(),
                    "batch_recon_loss": recon_loss.item(),
                    "batch_idx": batch_idx
                })

        avg_loss = epoch_loss / num_batches
        print(f"🟢 Epoch {epoch+1} Completed - Avg Loss: {avg_loss:.4f}")

        wandb.log({"train_loss": avg_loss, "epoch": epoch+1})

        val_loss = evaluate(model, device, val_dataloader, config)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"💾 Best model saved at epoch {epoch+1}")

    return best_loss


def evaluate(model, device, val_dataloader, config):
    model.eval()

    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for x_batch, y_batch in val_dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            logits, recon_loss = model(x_batch)

            ce_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y_batch.reshape(-1),
                ignore_index=-100
            )
            loss = ce_loss + config.get("recon_loss_weight", 0.1) * recon_loss

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"Validation Loss: {avg_loss:.4f}")
    wandb.log({"val_loss": avg_loss})

    return avg_loss
