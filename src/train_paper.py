import torch
import torch.optim as optim
import torch.nn.functional as F
import wandb
import sys

def train(model, device, train_dataloader, val_dataloader, config):
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    best_loss = float("inf")
    best_model = None

    
    wandb.watch(model, log="all")

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0
        num_batches = 0

        print(f"🚀 Epoch {epoch+1}/{config['epochs']} - Training Started")
        sys.stdout.flush()  

        for batch_idx, (x_batch, y_batch) in enumerate(train_dataloader):
            sys.stdout.flush()

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device) 

            optimizer.zero_grad()
            logits = model(x_batch)

            
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y_batch.reshape(-1), ignore_index=-100)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            
            if batch_idx % 10 == 0:
                print(f"✅ Batch {batch_idx+1} - Loss: {loss.item():.4f}")
                wandb.log({"batch_loss": loss.item(), "batch_idx": batch_idx})

        avg_loss = epoch_loss / num_batches
        print(f"🟢 Epoch {epoch+1} Completed - Avg Loss: {avg_loss:.4f}")
        sys.stdout.flush()

        wandb.log({"train_loss": avg_loss, "epoch": epoch + 1})

        
        val_loss = eval(model, device, val_dataloader, log_wandb=True)

        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            torch.save(best_model, "best_model.pth")
            print(f"💾 Best model saved at epoch {epoch+1}")
            sys.stdout.flush()

    return best_model  

def eval(model, device, val_dataloader, log_wandb=False):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(val_dataloader):
            print(f"🔍 Evaluating Batch {batch_idx+1}/{len(val_dataloader)}")
            sys.stdout.flush()

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y_batch.reshape(-1), ignore_index=-100)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"📊 Validation Loss: {avg_loss:.4f}")
    sys.stdout.flush()

    if log_wandb:
        wandb.log({"val_loss": avg_loss})

    return avg_loss  
