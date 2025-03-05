import logging
import torch
from torch.utils.data import DataLoader, random_split
import wandb

# Import train/eval functions
from train_eval import train, eval
# Import dataset class
from dataset import TinyShakespeareDataset
# Import models
from LTM_paper import SimpleLatentPipeline
from baseline import PureTransformerBaseline

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_data(file_path, seq_len, batch_size, subset_ratio=0.1, train_ratio=0.8, val_ratio=0.1):
    logging.info("Loading TinyShakespeare dataset...")
    full_dataset = TinyShakespeareDataset(file_path, seq_length=seq_len)

    # Create smaller subset for quick experimentation
    tiny_subset_size = int(subset_ratio * len(full_dataset))
    small_dataset, _ = random_split(full_dataset, [tiny_subset_size, len(full_dataset) - tiny_subset_size])

    # Split into train, val, test
    train_size = int(train_ratio * tiny_subset_size)
    val_size = int(val_ratio * tiny_subset_size)
    test_size = tiny_subset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(small_dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    vocab_size = len(full_dataset.vocab)

    logging.info(f"Train/Val/Test sizes: {train_size}/{val_size}/{test_size}")
    logging.info(f"Batch size: {batch_size}, Vocab size: {vocab_size}")

    return train_dataloader, val_dataloader, test_dataloader, vocab_size

def run_experiment(
    model,
    model_name,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    device,
    config
):
    logging.info(f"\n=== Running Experiment: {model_name} ===")
    logging.info(str(model))

    # Log configuration
    logging.info(f"Training Config: {config}")
    logging.info(f"Using device: {device}")

    # Train the model
    best_model_state_dict = train(
        model=model,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config
    )

    # Save best model
    best_model_path = f"best_model_{model_name}.pth"
    torch.save(best_model_state_dict, best_model_path)
    logging.info(f"Saved best model at {best_model_path}")

    # Optional: print model summary
    try:
        from torchinfo import summary
        dummy_input = torch.randint(
            low=0, high=65,  # assuming vocab_size=65 for TinyShakespeare
            size=(config["batch_size"], config["seq_len"]),
            device=device,
            dtype=torch.long  # critical fix to match Embedding's expectation
        )
        summary(model, input_data=dummy_input)

    except ImportError:
        logging.warning("Install torchinfo for detailed summaries (pip install torchinfo)")

    params_count = sum(p.numel() for p in model.parameters())
    logging.info(f"{model_name} has {params_count} parameters")

    # Load best model before evaluation
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)

    # Evaluate on test set
    test_loss = eval(model=model, device=device, val_dataloader=test_dataloader)
    logging.info(f"{model_name} - Final Test Loss: {test_loss:.4f}")

    return test_loss

def main():
    wandb.init(project="LTM", entity="yuertang", name="train_eval_run_modular")

    # Config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    file_path = "data/input.txt"
    seq_len = 50
    batch_size = 8

    train_dl, val_dl, test_dl, vocab_size = get_data(file_path, seq_len, batch_size)

    config = {
        "learning_rate": 1e-3,
        "epochs": 2,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }

    z_embed_dim = 32  # Fixed z_dim for both latent models

    # ====== Experiment 1: Latent model with 1-layer decoder ======
    latent_model_1layer = SimpleLatentPipeline(
        vocab_size=vocab_size,
        z_embed_dim=z_embed_dim,
        hidden_dim=256,
        num_heads=8,
        num_layers=1   # <-- 1 Layer TransformerDecoder
    ).to(device)
    run_experiment(latent_model_1layer, "Latent_1Layer", train_dl, val_dl, test_dl, device, config)

    # ====== Experiment 2: Latent model with 3-layer decoder ======
    # latent_model_3layer = SimpleLatentPipeline(
    #     vocab_size=vocab_size,
    #     z_embed_dim=z_embed_dim,
    #     hidden_dim=256,
    #     num_heads=8,
    #     num_layers=3   # <-- 3 Layer TransformerDecoder
    # ).to(device)
    # run_experiment(latent_model_3layer, "Latent_3Layer", train_dl, val_dl, test_dl, device, config)

    # ====== Experiment 3: Pure Transformer baseline (3 layers) ======
    # baseline_model = PureTransformerBaseline(
    #     vocab_size=vocab_size,
    #     hidden_dim=256,
    #     num_heads=8,
    #     num_layers=3   # <-- Traditional 4-layer TransformerEncoder
    # ).to(device)
    # run_experiment(baseline_model, "PureTransformerBaseline", train_dl, val_dl, test_dl, device, config)

    logging.info("All experiments completed successfully!")

if __name__ == "__main__":
    main()
