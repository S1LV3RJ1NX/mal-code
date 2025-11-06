"""
Main Training Script for DeepSeek Language Model with FP8 Quantization.

This script trains a DeepSeek model on text data with configurable parameters.
Includes support for FP8 quantization for 2× speedup and 50% memory savings.
"""

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
    # Configuration Selection
    # =========================================================================
    CONFIG_MODE = "SMALL"  
    if CONFIG_MODE == "SMALL":
        MODEL_CONFIG = DEEPSEEK_CONFIG_SMALL.copy()
        print("\n🎯 Using SMALL configuration")
    else:
        MODEL_CONFIG = DEEPSEEK_CONFIG.copy()
        print("\n🎯 Using FULL configuration")
    
    # Training configuration
    TRAINING_CONFIG = {
        "num_epochs": 1,
        "learning_rate": 3e-4,
        "min_lr": 3e-5,
        "batch_size": 32,
        "max_grad_norm": 1.0,
        "weight_decay": 0.1,
        "checkpoint_dir": f"checkpoints_{CONFIG_MODE.lower()}",
        "save_every": 1,
        "use_wandb": True,  # Enable Weights & Biases logging if needed
        "run_name": f"deepseek-{CONFIG_MODE.lower()}",
        'enable_quantization': True,
    }
    
    # Dataset configuration
    # DATASET_NAME = "roneneldan/TinyStories"
    DATASET_NAME = "sgoel9/paul_graham_essays"
    
    TRAIN_SPLIT = 0.95
    MAX_EXAMPLES = None
    print(f"   Dataset: Using {MAX_EXAMPLES} examples from {DATASET_NAME} dataset")
    print(f"   Train split: {TRAIN_SPLIT}")
    
    # Device configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n✓ Using GPU: {gpu_name}")
        
        # Enable TF32 for additional speedup on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ TF32 enabled for additional speedup")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\n✓ Using Apple Silicon GPU")
    else:
        device = torch.device("cpu")
        print("\n✓ Using CPU")
    
    # =========================================================================
    # Load Dataset
    # =========================================================================
    
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    print(f"Dataset: {DATASET_NAME}")
    
    dataset = load_dataset(DATASET_NAME, split="train")
    
    # Limit dataset size for DEMO mode
    if MAX_EXAMPLES and len(dataset) > MAX_EXAMPLES:
        dataset = dataset.select(range(MAX_EXAMPLES))
        print(f"Using subset: {len(dataset):,} examples (for faster training)")
    else:
        print(f"Total examples: {len(dataset):,}")
    
    # =========================================================================
    # Initialize Tokenizer
    # =========================================================================
    
    print("\n" + "="*70)
    print("INITIALIZING TOKENIZER")
    print("="*70)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"Tokenizer: GPT-2 (vocab_size={tokenizer.n_vocab})")
    
    # =========================================================================
    # Create DataLoaders
    # =========================================================================
    
    print("\n" + "="*70)
    print("CREATING DATALOADERS")
    print("="*70)
    
    train_loader, val_loader = create_dataloaders(
        dataset,
        tokenizer,
        context_length=MODEL_CONFIG["context_length"],
        batch_size=TRAINING_CONFIG["batch_size"],
        train_split=TRAIN_SPLIT,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        max_examples=MAX_EXAMPLES  # Pass max_examples for DEMO mode
    )
    
    print(f"Train batches: {len(train_loader):,}")
    print(f"Validation batches: {len(val_loader):,}")
    print(f"Tokens per batch: {TRAINING_CONFIG['batch_size'] * MODEL_CONFIG['context_length']:,}")
    
    # Estimate training time
    if CONFIG_MODE == "DEMO":
        total_steps = TRAINING_CONFIG['num_epochs'] * len(train_loader)
        print(f"\n⏱️  Estimated training time:")
        print(f"   Total steps: {total_steps:,}")
        if TRAINING_CONFIG["enable_quantization"]:
            print(f"   With FP8 on H100: ~5-6 hours")
            print(f"   With FP8 on other GPU: ~8-10 hours")
        else:
            print(f"   Without quantization: ~12-15 hours")
    
    # =========================================================================
    # Initialize Model
    # =========================================================================
    
    print("\n" + "="*70)
    print("INITIALIZING MODEL")
    print("="*70)
    
    model = DeepSeekModel(MODEL_CONFIG)
    print_model_info(model, MODEL_CONFIG)
    
    # =========================================================================
    # Initialize Trainer
    # =========================================================================
    
    print("\n" + "="*70)
    print("INITIALIZING TRAINER")
    print("="*70)
    
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
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"Configuration: {CONFIG_MODE}")
    print(f"Epochs: {TRAINING_CONFIG['num_epochs']}")
    print(f"Learning rate: {TRAINING_CONFIG['learning_rate']:.2e}")
    print(f"Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"Total training steps: {TRAINING_CONFIG['num_epochs'] * len(train_loader):,}")
    print(f"FP8 Quantization: {'Enabled' if TRAINING_CONFIG['enable_quantization'] else 'Disabled'}")
    print("="*70)
    
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
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Best model saved to: {trainer.checkpoint_dir / 'best_model.pt'}")
    
    if CONFIG_MODE == "DEMO":
        print("\n🎉 DEMO training complete!")
        print("   This model demonstrates all DeepSeek features:")
        print("   ✓ Multi-Head Latent Attention (MLA)")
        print("   ✓ Mixture of Experts (MoE)")
        if MODEL_CONFIG.get("use_mtp"):
            print("   ✓ Multi-Token Prediction (MTP)")
        if TRAINING_CONFIG["enable_quantization"]:
            print("   ✓ FP8 Quantization")
        print("\n   Perfect for book demonstration and understanding the architecture!")
    
    print("\nTo generate text, run:")
    print(f"  python inference.py --checkpoint {trainer.checkpoint_dir / 'best_model.pt'}")
    print("="*70)


if __name__ == "__main__":
    main()