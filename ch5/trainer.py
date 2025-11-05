"""
DeepSeek Training Module

This module implements a training pipeline for the DeepSeek language model with MLA and MoE.
It includes a Trainer class that handles the training loop, validation, checkpointing, and logging.

The training process follows best practices like:
- Gradient clipping to prevent exploding gradients
- Learning rate scheduling with cosine annealing
- Model checkpointing to save best models and regular snapshots
- Wandb integration for experiment tracking
- Progress bars with live metrics
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with 'pip install wandb' for experiment tracking.")


class Trainer:
    """
    Trainer class that handles the training of a DeepSeek model.

    This class encapsulates all the functionality needed for training including:
    - Forward and backward passes
    - Loss computation
    - Metrics tracking (loss and perplexity)
    - Model checkpointing
    - Validation
    - Progress logging

    The trainer supports CPU, CUDA, and MPS (Apple Silicon) training with automatic device detection.
    
    Args:
        model (nn.Module): The DeepSeek language model instance to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        config (dict): Configuration dictionary containing training parameters like:
            - num_epochs (int): Number of training epochs
            - learning_rate (float): Learning rate for optimizer
            - max_grad_norm (float): Maximum gradient norm for clipping
            - checkpoint_dir (str): Directory to save checkpoints
            - save_every (int): Save checkpoint every N epochs
            - weight_decay (float): Weight decay for optimizer
            - min_lr (float): Minimum learning rate for scheduler
            - use_wandb (bool): Whether to use Weights & Biases logging
            - run_name (str): Name of the wandb run
        device (torch.device): Device to train on (cuda/mps/cpu)
    
    Example:
        >>> from components.deepseek_model import DeepSeekModel
        >>> from constants import DEEPSEEK_CONFIG_SMALL
        >>> 
        >>> model = DeepSeekModel(DEEPSEEK_CONFIG_SMALL)
        >>> # ... create train_loader and val_loader ...
        >>> 
        >>> config = {
        ...     "num_epochs": 10,
        ...     "learning_rate": 3e-4,
        ...     "max_grad_norm": 1.0,
        ...     "checkpoint_dir": "checkpoints",
        ...     "save_every": 2,
        ...     "weight_decay": 0.1
        ... }
        >>> 
        >>> trainer = Trainer(model, train_loader, val_loader, config, device)
        >>> trainer.train()
    """
    
    def __init__(self, model, train_loader, val_loader, config, device):
        """Initialize the trainer with model, data loaders, and configuration."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Mixed precision training setup
        self.use_mixed_precision = config.get("mixed_precision", False) and device.type == "cuda"
        self.scaler = GradScaler('cuda') if self.use_mixed_precision else None
        
        # Setup optimizer with AdamW (better weight decay than Adam)
        self.optimizer = AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            betas=(0.9, 0.95),  # Standard betas for language models
            weight_decay=config.get("weight_decay", 0.1)
        )
        
        # Setup learning rate scheduler with cosine annealing
        # This gradually reduces learning rate following a cosine curve
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config["num_epochs"] * len(train_loader),
            eta_min=config.get("min_lr", 0.0)
        )
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Initialize wandb for experiment tracking if enabled
        if config.get("use_wandb", False):
            if WANDB_AVAILABLE:
                wandb.init(
                    project="deepseek-training",
                    config=config,
                    name=config.get("run_name", "deepseek-run"),
                )
                print("  Wandb: Enabled")
            else:
                print("  Wandb: Requested but not available (install with 'pip install wandb')")
        
        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Optimizer: AdamW (lr={config['learning_rate']:.2e})")
        print(f"  Scheduler: CosineAnnealingLR")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
    
    def train_epoch(self) -> tuple:
        """
        Train the model for one epoch.

        This method:
        1. Sets model to training mode
        2. Iterates through batches of data
        3. Performs forward and backward passes
        4. Updates model parameters with gradient clipping
        5. Tracks and logs metrics

        Returns:
            tuple: (avg_loss, avg_perplexity) for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Training Epoch {self.current_epoch + 1}/{self.config['num_epochs']}"
        )
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass - model returns both logits and loss
            self.optimizer.zero_grad()
            
            if self.use_mixed_precision:
                # Mixed precision forward pass
                with autocast('cuda'):
                    logits, loss = self.model(inputs, targets)
                
                # Mixed precision backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping with mixed precision
                if self.config.get("max_grad_norm", None):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["max_grad_norm"]
                    )
                
                # Mixed precision optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision forward pass
                logits, loss = self.model(inputs, targets)
                
                # Standard backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.get("max_grad_norm", None):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["max_grad_norm"]
                    )
                
                # Standard optimizer step
                self.optimizer.step()
            
            # Scheduler step (same for both precision modes)
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Calculate perplexity (exp of loss)
            perplexity = torch.exp(loss).item()
            
            # Update progress bar with current metrics
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ppl": f"{perplexity:.2f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log batch metrics to wandb if enabled
            if self.config.get("use_wandb", False) and WANDB_AVAILABLE:
                wandb.log(
                    {
                        "train/batch_loss": loss.item(),
                        "train/batch_perplexity": perplexity,
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                        "train/step": self.global_step,
                    }
                )
        
        # Calculate epoch averages
        avg_loss = total_loss / num_batches
        avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return avg_loss, avg_perplexity
    
    @torch.no_grad()
    def validate(self) -> tuple:
        """
        Validate the model on the validation set.

        This method:
        1. Sets model to evaluation mode
        2. Disables gradient computation (saves memory)
        3. Computes loss and perplexity on validation data

        Returns:
            tuple: (avg_loss, avg_perplexity) for the validation set
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(
            self.val_loader,
            desc=f"Validation Epoch {self.current_epoch + 1}"
        )
        
        for inputs, targets in progress_bar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass - no gradient computation
            if self.use_mixed_precision:
                with autocast('cuda'):
                    logits, loss = self.model(inputs, targets)
            else:
                logits, loss = self.model(inputs, targets)
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Calculate perplexity
            perplexity = torch.exp(loss).item()
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ppl": f"{perplexity:.2f}"
            })
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return avg_loss, avg_perplexity
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """
        Save a checkpoint with model and optimizer state.
        
        Args:
            filename (str): Name of the checkpoint file
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config
        }
        
        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        
        if is_best:
            print(f"  ✓ New best model saved: {save_path}")
        else:
            print(f"  Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, filename: str) -> bool:
        """
        Load a checkpoint to resume training.
        
        Args:
            filename (str): Name of the checkpoint file to load
            
        Returns:
            bool: True if successful, False otherwise
        """
        load_path = self.checkpoint_dir / filename
        if not load_path.exists():
            print(f"Checkpoint {load_path} not found")
            return False
        
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        print(f"Checkpoint loaded from {load_path}")
        print(f"  Resuming from epoch {self.current_epoch + 1}")
        print(f"  Best validation loss: {self.best_val_loss:.4f}")
        
        return True
    
    def train(self):
        """
        Main training loop.

        This method orchestrates the entire training process:
        1. Runs training for specified number of epochs
        2. Performs validation after each epoch
        3. Logs metrics
        4. Saves model checkpoints

        The training loop saves:
        - Best model based on validation loss
        - Regular checkpoints at specified intervals
        """
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        # Print model info
        if hasattr(self.model, 'count_parameters'):
            params = self.model.count_parameters()
            print(f"\nModel Parameters:")
            print(f"  Total: {params['total']:,}")
            if 'routed_experts' in params:
                print(f"  Routed experts: {params['routed_experts']:,}")
                print(f"  Shared experts: {params['shared_experts']:,}")
        
        print(f"\nDataset:")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print()
        
        for epoch in range(self.config["num_epochs"]):
            self.current_epoch = epoch
            
            # Training phase
            train_loss, train_ppl = self.train_epoch()
            
            # Validation phase
            val_loss, val_ppl = self.validate()
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
            print(f"  Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
            
            # Log epoch metrics to wandb if enabled
            if self.config.get("use_wandb", False) and WANDB_AVAILABLE:
                wandb.log(
                    {
                        "train/epoch_loss": train_loss,
                        "train/epoch_perplexity": train_ppl,
                        "val/epoch_loss": val_loss,
                        "val/epoch_perplexity": val_ppl,
                        "epoch": epoch,
                    }
                )
            
            # Save best model based on validation loss
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pt", is_best=True)
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.get("save_every", 5) == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
            
            print()
        
        print("="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best model saved to: {self.checkpoint_dir / 'best_model.pt'}")
        print("="*60)