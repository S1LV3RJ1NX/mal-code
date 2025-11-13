"""
DeepSeek Training Module with FP8 Quantization Support

This module implements a training pipeline for the DeepSeek language model with optional FP8 quantization.
It includes a Trainer class that handles the training loop, validation, checkpointing, and logging.

The training process follows best practices like:
- Optional FP8 quantization for 2× speedup and 50% memory savings
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

# Import quantization utilities
try:
    from components.quantization import (
        FP8Config,
        convert_linear_to_quantized,
        check_fp8_support
    )
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False
    print("Warning: Quantization not available. Training will use standard precision.")


class Trainer:
    """
    Trainer class that handles the training of a DeepSeek model with optional FP8 quantization.

    This class encapsulates all the functionality needed for training including:
    - Forward and backward passes
    - Loss computation
    - Optional FP8 quantization (2× speedup, 50% memory savings)
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
            - enable_quantization (bool): Whether to use FP8 quantization (default from model config)
        device (torch.device): Device to train on (cuda/mps/cpu)
    
    Example:
        >>> from components.deepseek import DeepSeekModel
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
        ...     "weight_decay": 0.1,
        ...     "enable_quantization": True  # Enable FP8!
        ... }
        >>> 
        >>> trainer = Trainer(model, train_loader, val_loader, config, device)
        >>> trainer.train()
    """
    
    def __init__(self, model, train_loader, val_loader, config, device):
        """Initialize the trainer with model, data loaders, and configuration."""
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # FP8 Quantization support
        self.enable_quantization = config.get("enable_quantization", False) and QUANTIZATION_AVAILABLE
        
        if self.enable_quantization:
            supports_fp8, device_info = check_fp8_support()
            print(f"\n{'='*70}")
            print("FP8 QUANTIZATION ENABLED")
            print(f"{'='*70}")
            print(f"Device: {device_info}")
            print(f"Converting model to FP8 quantized layers...")
            
            # Convert model to use quantized layers
            model = convert_linear_to_quantized(model, use_te=supports_fp8)
            print("✓ Model conversion complete")
            
            if supports_fp8:
                print("✓ Using TRUE FP8 (E4M3) - expect 2× speedup")
                # Transformer Engine expects BF16 inputs
                model = model.to(torch.bfloat16)
                print("✓ Model converted to BF16 for Transformer Engine")
            else:
                print("✓ Using FP16/BF16 simulation - expect 1.5× speedup")
            print(f"{'='*70}\n")
        
        # Move model to device
        self.model = model.to(device)
        
        # Mixed precision training setup
        self.use_mixed_precision = config.get("mixed_precision", False) and device.type == "cuda"
        self.scaler = GradScaler('cuda') if self.use_mixed_precision else None
        
        # Setup optimizer with AdamW (better weight decay than Adam)
        self.optimizer = AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            betas=(0.9, 0.95),  # Standard betas for language models
            weight_decay=config.get("weight_decay", 0.1),
            fused=True if (device.type == "cuda" and torch.cuda.is_available()) else False
        )
        
        # Setup learning rate scheduler with cosine annealing
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
        self._optimizer_states_converted = False
        
        # Max steps cap (for time-constrained training)
        self.max_steps = config.get("max_steps", None)
        
        # Loss tracking for stability
        self.loss_history = []
        
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
        print(f"  Optimizer: AdamW (lr={config['learning_rate']:.2e}, fused={device.type=='cuda'})")
        print(f"  Scheduler: CosineAnnealingLR")
        print(f"  FP8 Quantization: {'Enabled' if self.enable_quantization else 'Disabled'}")
        if self.max_steps:
            print(f"  Max steps: {self.max_steps:,} (time-constrained training)")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
    
    def _convert_optimizer_states_to_bf16(self):
        """
        Convert optimizer states (Adam moments) to BF16 for memory savings.
        Per DeepSeek's mixed precision framework.
        """
        if self._optimizer_states_converted or not self.enable_quantization:
            return
        
        for group in self.optimizer.param_groups:
            for p in group['params']:
                state = self.optimizer.state[p]
                if len(state) == 0:
                    continue
                
                # Convert momentum buffers to BF16
                if 'exp_avg' in state:
                    state['exp_avg'] = state['exp_avg'].to(torch.bfloat16)
                if 'exp_avg_sq' in state:
                    state['exp_avg_sq'] = state['exp_avg_sq'].to(torch.bfloat16)
        
        self._optimizer_states_converted = True
        if self.enable_quantization:
            print("✓ Optimizer states converted to BF16")
    
    def _check_loss_stability(self, loss: float) -> bool:
        """Check if training is stable (no divergence)."""
        self.loss_history.append(loss)
        
        # Keep only last 100 steps
        if len(self.loss_history) > 100:
            self.loss_history.pop(0)
        
        # Check for NaN
        if torch.isnan(torch.tensor(loss)):
            print(f"\n⚠ WARNING: NaN loss detected at step {self.global_step}")
            return False
        
        # Check for divergence
        if len(self.loss_history) > 10:
            recent_avg = sum(self.loss_history[-10:]) / 10
            if loss > recent_avg * 10:
                print(f"\n⚠ WARNING: Loss divergence at step {self.global_step}")
                print(f"  Current: {loss:.4f}, Recent avg: {recent_avg:.4f}")
                return False
        
        return True
    
    def train_epoch(self) -> tuple:
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Training Epoch {self.current_epoch + 1}/{self.config['num_epochs']}"
        )
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            # Check if we've hit the max steps limit
            if self.max_steps and self.global_step >= self.max_steps:
                print(f"\n⏱️  Reached max_steps limit ({self.max_steps:,}). Stopping training.")
                break
            
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_mixed_precision:
                with autocast('cuda'):
                    logits, loss = self.model(inputs, targets)
                self.scaler.scale(loss).backward()
                if self.config.get("max_grad_norm", None):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["max_grad_norm"]
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, loss = self.model(inputs, targets)
                loss.backward()
                if self.config.get("max_grad_norm", None):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["max_grad_norm"]
                    )
                self.optimizer.step()
            
            # Convert optimizer states to BF16 after first step (if quantization enabled)
            if not self._optimizer_states_converted:
                self._convert_optimizer_states_to_bf16()
            
            # Scheduler step
            self.scheduler.step()
            
            # Check stability
            self._check_loss_stability(loss.item())
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Calculate perplexity
            perplexity = torch.exp(loss).item()
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ppl": f"{perplexity:.2f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log to wandb
            if self.config.get("use_wandb", False) and WANDB_AVAILABLE:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/batch_perplexity": perplexity,
                    "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                    "train/step": self.global_step,
                })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_perplexity = torch.exp(torch.tensor(avg_loss)).item() if num_batches > 0 else float('inf')
        
        return avg_loss, avg_perplexity
    
    @torch.no_grad()
    def validate(self) -> tuple:
        """Validate the model on the validation set."""
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
            
            if self.use_mixed_precision:
                with autocast('cuda'):
                    logits, loss = self.model(inputs, targets)
            else:
                logits, loss = self.model(inputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
            
            perplexity = torch.exp(loss).item()
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ppl": f"{perplexity:.2f}"
            })
        
        avg_loss = total_loss / num_batches
        avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return avg_loss, avg_perplexity
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save a checkpoint with model and optimizer state."""
        # Store model config for inference
        model_config = {
            "vocab_size": self.model.cfg["vocab_size"],
            "context_length": self.model.cfg["context_length"],
            "emb_dim": self.model.cfg["emb_dim"],
            "n_layers": self.model.cfg["n_layers"],
            "n_heads": self.model.cfg["n_heads"],
            "kv_latent_dim": self.model.cfg["kv_latent_dim"],
            "num_experts": self.model.cfg["num_experts"],
            "num_shared_experts": self.model.cfg["num_shared_experts"],
            "top_k": self.model.cfg["top_k"],
            "expert_hidden_dim": self.model.cfg["expert_hidden_dim"],
            "drop_rate": self.model.cfg["drop_rate"],
            "qkv_bias": self.model.cfg.get("qkv_bias", False),
            "use_mtp": self.model.cfg.get("use_mtp", False),
            "mtp_depth": self.model.cfg.get("mtp_depth", 0),
            "mtp_weight": self.model.cfg.get("mtp_weight", 0.0),
        }
        
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,  # Training config
            "model_config": model_config,  # Model architecture config
            "quantization_enabled": self.enable_quantization,
        }
        
        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        
        if is_best:
            print(f"  ✓ New best model saved: {save_path}")
        else:
            print(f"  Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, filename: str) -> bool:
        """Load a checkpoint to resume training."""
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
        """Main training loop."""
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        
        if self.enable_quantization:
            print("\n🚀 FP8 Quantization Active:")
            print("  • Expected speedup: 1.5-2×")
            print("  • Expected memory savings: ~50%")
            print("  • Expected loss increase: <0.5%\n")
        
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
        if self.max_steps:
            print(f"  Max steps: {self.max_steps:,} (24-hour cap)")
        print()
        
        for epoch in range(self.config["num_epochs"]):
            self.current_epoch = epoch
            
            # Check if we've hit max steps before starting epoch
            if self.max_steps and self.global_step >= self.max_steps:
                print(f"\n⏱️  Reached max_steps limit ({self.max_steps:,}) before epoch {epoch + 1}.")
                print("Stopping training early.")
                break
            
            # Training phase
            train_loss, train_ppl = self.train_epoch()
            
            # Validation phase
            val_loss, val_ppl = self.validate()
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
            print(f"  Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
            if self.max_steps:
                print(f"  Steps completed: {self.global_step:,} / {self.max_steps:,}")
            
            # Log to wandb
            if self.config.get("use_wandb", False) and WANDB_AVAILABLE:
                wandb.log({
                    "train/epoch_loss": train_loss,
                    "train/epoch_perplexity": train_ppl,
                    "val/epoch_loss": val_loss,
                    "val/epoch_perplexity": val_ppl,
                    "epoch": epoch,
                })
            
            # Save best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pt", is_best=True)
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.get("save_every", 5) == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
            
            print()
        
        print("="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Total steps completed: {self.global_step:,}")
        print(f"Best model saved to: {self.checkpoint_dir / 'best_model.pt'}")
        print("="*70)