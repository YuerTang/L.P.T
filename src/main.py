import torch
from torch.utils.data import DataLoader, random_split
import wandb
from train_paper import train, eval
from LTM_paper import SimpleLatentPipeline  
from dataset import TinyShakespeareDataset 

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

wandb.init(project="LTM", entity="yuertang", name="train_eval_run")

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

file_path = "data/input.txt"
seq_len = 50
batch_size = 8

logging.info("Loading TinyShakespeare dataset...")
full_dataset = TinyShakespeareDataset(file_path, seq_length=seq_len)
tiny_subset_size = int(0.1 * len(full_dataset))
small_dataset, _ = random_split(full_dataset, [tiny_subset_size, len(full_dataset) - tiny_subset_size])

# ✅ Split into train, val, and test
train_size = int(0.8 * tiny_subset_size)
val_size = int(0.1 * tiny_subset_size)
test_size = tiny_subset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    small_dataset, [train_size, val_size, test_size]
)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

logging.info(f"Train DataLoader initialized with batch size {batch_size}")
logging.info(f"Validation DataLoader initialized with batch size {batch_size}")
logging.info(f"Test DataLoader initialized with batch size {batch_size}")


z_embed_dim_values = [32, 64, 128, 256]
for z_dim in z_embed_dim_values:
    logging.info(f"Initializing model with z_dim: {z_dim}...")
    model = SimpleLatentPipeline(vocab_size=len(full_dataset.vocab), z_embed_dim=z_dim).to(device)
    print(model)
    logging.info(f"Model initialized with z_dim: {z_dim} and vocab size: {len(full_dataset.vocab)}")


    config = {
        "learning_rate": 1e-3,
        "epochs": 2,
        "batch_size": batch_size
    }

    logging.info(f"Training Configuration: {config}")

    logging.info("Starting training process...")
    best_model_state_dict = train(
        model=model,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config
    )

    best_model_path = "best_model.pth"
    torch.save(best_model_state_dict, best_model_path)
    logging.info(f"Best model saved at {best_model_path}")
    from torchinfo import summary

    summary(model, input_size=(batch_size, seq_len))  # adjust as needed
    latent_model_params = sum(p.numel() for p in latent_model.parameters())
    print(f"Latent model has {latent_model_params} parameters.")
    logging.info("Loading best model for evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)

    logging.info("Starting evaluation on test set...")
    test_loss = eval(model=model, device=device, val_dataloader=test_dataloader)

    logging.info(f"✅ Final Test Loss: {test_loss:.4f}")
logging.info("🎉 All processes complete. Exiting.")
