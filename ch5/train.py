"""
Main Training Script for DeepSeek Language Model with FP8 Quantization.

This script trains a DeepSeek model on text data with configurable parameters.
Includes support for FP8 quantization for 2× speedup and 50% memory savings.

Multi-Dataset Training:
- Uses diverse sources (TinyStories, WikiText, OpenWebText) to prevent overfitting
- Optimized for 24-hour training window with proper sample sizing
- Maintains model config while maximizing data diversity
"""
import datetime
import torch
from datasets import load_dataset, concatenate_datasets
import tiktoken

from components.deepseek import DeepSeekModel
from constants import DEEPSEEK_CONFIG, DEEPSEEK_CONFIG_SMALL
from dataset import create_dataloaders
from trainer import Trainer
from utils import print_model_info
# 28700

def create_diverse_dataset(target_epochs: int = 3, seed: int = 42):
    """
    Create a diverse mixed dataset optimized for 24-hour training window.
    
    Based on measured baseline: 11.47 s/step on H100 NVL
    - 24 hours = 86,400 seconds ≈ 7,536 total steps available
    - For 3 epochs: ~2,512 steps/epoch → ~3,000 examples needed
    - For 2 epochs: ~3,768 steps/epoch → ~5,000 examples needed
    
    Mixing strategy:
    - TinyStories: High-quality synthetic stories (good for coherence)
    - WikiText-2: Encyclopedia-style text (factual knowledge)
    - OpenWebText: Web content (diverse writing styles)
    
    Args:
        target_epochs (int): Number of epochs to train (2 or 3)
        seed (int): Random seed for reproducibility
    
    Returns:
        Dataset: Combined and shuffled dataset
    """
    print("\n" + "="*70)
    print("CREATING DIVERSE MULTI-SOURCE DATASET")
    print("="*70)
    
    # Calculate target sample size based on epochs
    # Using measured ratio: 50k examples → 41,113 steps → ~1.216 examples/step
    if target_epochs == 3:
        target_examples = 3000
        samples_per_large_source = 1000  # TinyStories, WikiText
        samples_per_small_source = 500   # OpenWebText (smaller for variety)
        print(f"Target: 3 epochs with ~3,000 examples (~2,512 steps/epoch)")
    elif target_epochs == 2:
        target_examples = 5000
        samples_per_large_source = 2000
        samples_per_small_source = 1000
        print(f"Target: 2 epochs with ~5,000 examples (~3,768 steps/epoch)")
    else:
        raise ValueError("target_epochs must be 2 or 3")
    
    print(f"Expected total training time: ≤ 24 hours on H100 NVL")
    print(f"Estimated steps: ~{7536 // target_epochs * target_epochs:,} total")
    print()
    
    datasets_to_mix = []
    
    # 1. TinyStories - High-quality synthetic stories
    print("Loading TinyStories...")
    try:
        ds_tiny = load_dataset("roneneldan/TinyStories", split="train", trust_remote_code=True)
        ds_tiny = ds_tiny.shuffle(seed=seed).select(range(min(len(ds_tiny), samples_per_large_source)))
        print(f"  ✓ TinyStories: {len(ds_tiny):,} examples")
        datasets_to_mix.append(ds_tiny)
    except Exception as e:
        print(f"  ✗ TinyStories failed: {e}")
    
    # 2. WikiText-2 - Encyclopedia-style factual text
    print("Loading WikiText-2...")
    try:
        ds_wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
        # Filter out empty or very short texts
        ds_wiki = ds_wiki.filter(lambda x: len(x["text"].strip()) > 50)
        ds_wiki = ds_wiki.shuffle(seed=seed).select(range(min(len(ds_wiki), samples_per_large_source)))
        print(f"  ✓ WikiText-2: {len(ds_wiki):,} examples")
        datasets_to_mix.append(ds_wiki)
    except Exception as e:
        print(f"  ✗ WikiText-2 failed: {e}")
    
    # 3. OpenWebText - Diverse web content
    print("Loading OpenWebText...")
    try:
        ds_owt = load_dataset("Skylion007/openwebtext", split="train", trust_remote_code=True)
        ds_owt = ds_owt.shuffle(seed=seed).select(range(min(len(ds_owt), samples_per_small_source)))
        print(f"  ✓ OpenWebText: {len(ds_owt):,} examples")
        datasets_to_mix.append(ds_owt)
    except Exception as e:
        print(f"  ✗ OpenWebText failed: {e}")
    
    # Fallback: if we couldn't load enough datasets, use more from what we have
    if len(datasets_to_mix) == 0:
        print("\n⚠ WARNING: No datasets loaded successfully!")
        print("Falling back to TinyStories only...")
        ds_tiny = load_dataset("roneneldan/TinyStories", split="train", trust_remote_code=True)
        ds_tiny = ds_tiny.shuffle(seed=seed).select(range(target_examples))
        return ds_tiny
    
    # Combine all datasets
    print("\nCombining datasets...")
    combined = concatenate_datasets(datasets_to_mix)
    
    # Shuffle the combined dataset for good mixing
    combined = combined.shuffle(seed=seed)
    
    total_examples = len(combined)
    print(f"\n✓ Combined dataset created:")
    print(f"  Total examples: {total_examples:,}")
    print(f"  Sources mixed: {len(datasets_to_mix)}")
    print(f"  Diversity: {'High' if len(datasets_to_mix) >= 3 else 'Medium' if len(datasets_to_mix) == 2 else 'Low'}")
    print("="*70)
    
    return combined


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
    
    # =========================================================================
    # Training Configuration with 24-Hour Optimization
    # =========================================================================
    # Based on measured baseline: 11.47 s/step → 7,536 steps fit in 24 hours
    TARGET_EPOCHS = 3  # Change to 2 for longer training per epoch
    MAX_TOTAL_STEPS = 7536  # Hard cap for 24-hour window
    
    # Add slight dropout increase to combat overfitting on smaller dataset
    MODEL_CONFIG["drop_rate"] = 0.12  # Up from 0.1
    
    TRAINING_CONFIG = {
        "num_epochs": TARGET_EPOCHS,
        "learning_rate": 3e-4,
        "min_lr": 3e-5,
        "batch_size": 256,
        "max_grad_norm": 1.0,
        "weight_decay": 0.1,  # Keep regularization for small dataset
        "checkpoint_dir": f"checkpoints_{CONFIG_MODE.lower()}_diverse_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "save_every": 1,  # Save every epoch (only 2-3 epochs total)
        "use_wandb": True,  # Enable Weights & Biases logging if needed
        "run_name": f"deepseek-{CONFIG_MODE.lower()}-diverse_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "enable_quantization": True,
        "max_steps": MAX_TOTAL_STEPS,  # Safety cap for 24-hour window
    }
    
    # Dataset configuration - Multi-source for diversity
    USE_DIVERSE_DATASET = True  # Set to False to use single dataset
    TRAIN_SPLIT = 0.95
    
    print(f"\n📊 Training Strategy:")
    print(f"   Epochs: {TARGET_EPOCHS}")
    print(f"   Max steps: {MAX_TOTAL_STEPS:,} (24-hour cap)")
    print(f"   Diverse dataset: {USE_DIVERSE_DATASET}")
    print(f"   Train split: {TRAIN_SPLIT}")
    
    # Device configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n✓ Using GPU: {gpu_name}")
        
        # Enable TF32 for additional speedup on Ampere+ GPUs (using new API)
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        torch.backends.cudnn.conv.fp32_precision = 'tf32'
        print("✓ TF32 enabled for additional speedup")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\n✓ Using Apple Silicon GPU")
    else:
        device = torch.device("cpu")
        print("\n✓ Using CPU")
    
    # =========================================================================
    # Load Dataset - Multi-Source or Single-Source
    # =========================================================================
    
    if USE_DIVERSE_DATASET:
        # Use diverse multi-source dataset (recommended for book)
        dataset = create_diverse_dataset(target_epochs=TARGET_EPOCHS, seed=42)
    else:
        # Fallback: single dataset (original behavior)
        print("\n" + "="*70)
        print("LOADING SINGLE-SOURCE DATASET")
        print("="*70)
        DATASET_NAME = "roneneldan/TinyStories"
        print(f"Dataset: {DATASET_NAME}")
        
        dataset = load_dataset(DATASET_NAME, split="train", trust_remote_code=True)
        
        # Calculate appropriate sample size based on epochs
        if TARGET_EPOCHS == 3:
            MAX_EXAMPLES = 3000
        elif TARGET_EPOCHS == 2:
            MAX_EXAMPLES = 5000
        else:
            MAX_EXAMPLES = 3000
        
        if len(dataset) > MAX_EXAMPLES:
            dataset = dataset.shuffle(seed=42).select(range(MAX_EXAMPLES))
            print(f"Using subset: {len(dataset):,} examples")
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
        max_examples=None  # Dataset already sized appropriately
    )
    
    print(f"Train batches: {len(train_loader):,}")
    print(f"Validation batches: {len(val_loader):,}")
    print(f"Tokens per batch: {TRAINING_CONFIG['batch_size'] * MODEL_CONFIG['context_length']:,}")
    
    # Calculate actual training metrics
    total_steps = TRAINING_CONFIG['num_epochs'] * len(train_loader)
    steps_per_epoch = len(train_loader)
    
    print(f"\n⏱️  Training Time Estimates (based on 11.47 s/step baseline):")
    print(f"   Total steps: {total_steps:,}")
    print(f"   Steps per epoch: {steps_per_epoch:,}")
    print(f"   Max allowed steps: {MAX_TOTAL_STEPS:,} (24-hour cap)")
    
    if total_steps > MAX_TOTAL_STEPS:
        print(f"\n⚠️  WARNING: Estimated steps ({total_steps:,}) exceed 24-hour cap!")
        print(f"   Training will stop at step {MAX_TOTAL_STEPS:,}")
        estimated_hours = (MAX_TOTAL_STEPS * 11.47) / 3600
    else:
        estimated_hours = (total_steps * 11.47) / 3600
        print(f"   ✓ Within budget: ~{estimated_hours:.1f} hours estimated")
    
    if TRAINING_CONFIG["enable_quantization"]:
        print(f"   With FP8 quantization: ~{estimated_hours:.1f} hours")
    else:
        print(f"   Without quantization: ~{estimated_hours * 1.5:.1f} hours")
    
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
    
    print("\n🎉 Training complete!")
    print("   This model demonstrates all DeepSeek features:")
    print("   ✓ Multi-Head Latent Attention (MLA)")
    print("   ✓ Mixture of Experts (MoE)")
    if MODEL_CONFIG.get("use_mtp"):
        print("   ✓ Multi-Token Prediction (MTP)")
    if TRAINING_CONFIG["enable_quantization"]:
        print("   ✓ FP8 Quantization")
    if USE_DIVERSE_DATASET:
        print("   ✓ Multi-Source Dataset (TinyStories + WikiText + OpenWebText)")
    print("\n   Perfect for book demonstration and understanding the architecture!")
    
    print("\n📈 Training Statistics:")
    print(f"   Total steps: {trainer.global_step:,}")
    print(f"   Epochs completed: {trainer.current_epoch + 1}")
    print(f"   Best validation loss: {trainer.best_val_loss:.4f}")
    
    print("\n💡 To generate text, run:")
    print(f"  python inference.py --model_path {trainer.checkpoint_dir / 'best_model.pt'} --interactive")
    print("\n   Or for a single prompt:")
    print(f"  python inference.py --model_path {trainer.checkpoint_dir / 'best_model.pt'} \\")
    print('    --prompt "Once upon a time" --max_new_tokens 100')
    print("="*70)


if __name__ == "__main__":
    main()