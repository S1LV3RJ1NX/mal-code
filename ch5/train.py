"""
Main Training Script for DeepSeek Language Model.

This script trains a DeepSeek model on text data with configurable parameters.
"""

from sympy import false
import torch
from datasets import load_dataset
import tiktoken

from components.deepseek import DeepSeekModel
from constants import DEEPSEEK_CONFIG, DEEPSEEK_CONFIG_SMALL
from dataset import create_dataloaders
from trainer import Trainer
from utils import print_model_info


def main():
    """
    Main training function.
    """
    # =========================================================================
    # Configuration
    # =========================================================================
    
    # Choose configuration (SMALL for quick testing, default for full training)
    USE_SMALL_CONFIG = True  # Set to True for quick testing
    
    if USE_SMALL_CONFIG:
        MODEL_CONFIG = DEEPSEEK_CONFIG_SMALL.copy()
        print("Using SMALL configuration for quick testing")
    else:
        MODEL_CONFIG = DEEPSEEK_CONFIG.copy()
        print("Using DEFAULT configuration")
    
    # Training hyperparameters
    TRAINING_CONFIG = {
        "num_epochs": 5,
        "learning_rate": 3e-4,
        "min_lr": 3e-5,
        "batch_size": 16,
        "max_grad_norm": 1.0,
        "weight_decay": 0.1,
        "checkpoint_dir": "checkpoints",
        "save_every": 2,
        "use_wandb": false,  # Enable Weights & Biases logging
        "run_name": "deepseek-tinystories",  # Name for this training run
    }
    
    # Dataset configuration
    DATASET_NAME = "roneneldan/TinyStories"
    TRAIN_SPLIT = 0.95
    
    # Device configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # =========================================================================
    # Load Dataset
    # =========================================================================
    
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    print(f"Dataset: {DATASET_NAME}")
    
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"Total examples: {len(dataset)}")
    
    # =========================================================================
    # Initialize Tokenizer
    # =========================================================================
    
    print("\n" + "="*60)
    print("INITIALIZING TOKENIZER")
    print("="*60)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"Tokenizer: GPT-2 (vocab_size={tokenizer.n_vocab})")
    
    # =========================================================================
    # Create DataLoaders
    # =========================================================================
    
    print("\n" + "="*60)
    print("CREATING DATALOADERS")
    print("="*60)
    
    train_loader, val_loader = create_dataloaders(
        dataset,
        tokenizer,
        context_length=MODEL_CONFIG["context_length"],
        batch_size=TRAINING_CONFIG["batch_size"],
        train_split=TRAIN_SPLIT,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Tokens per batch: {TRAINING_CONFIG['batch_size'] * MODEL_CONFIG['context_length']}")
    
    # =========================================================================
    # Initialize Model
    # =========================================================================
    
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    
    model = DeepSeekModel(MODEL_CONFIG)
    print_model_info(model, MODEL_CONFIG)
    
    # =========================================================================
    # Initialize Trainer
    # =========================================================================
    
    print("\n" + "="*60)
    print("INITIALIZING TRAINER")
    print("="*60)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=TRAINING_CONFIG,
        device=device
    )
    
    # =========================================================================
    # Train Model
    # =========================================================================
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Epochs: {TRAINING_CONFIG['num_epochs']}")
    print(f"Learning rate: {TRAINING_CONFIG['learning_rate']:.2e}")
    print(f"Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"Total training steps: {TRAINING_CONFIG['num_epochs'] * len(train_loader)}")
    print("="*60)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving checkpoint...")
        trainer.save_checkpoint("interrupted_checkpoint.pt")
        print("Checkpoint saved. You can resume training later.")
        return
    except Exception as e:
        print(f"\n\nError during training: {e}")
        print("Saving emergency checkpoint...")
        trainer.save_checkpoint("error_checkpoint.pt")
        raise
    
    # =========================================================================
    # Training Complete
    # =========================================================================
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Best model saved to: {trainer.checkpoint_dir / 'best_model.pt'}")
    print("\nTo generate text, run:")
    print(f"  python inference.py --model_path {trainer.checkpoint_dir / 'best_model.pt'} --interactive")
    print("="*60)


if __name__ == "__main__":
    main()