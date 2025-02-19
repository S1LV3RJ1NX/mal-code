"""
GPT2 Training Module

This module implements a training pipeline for the GPT2 language model. It includes a Trainer class
that handles the training loop, validation, checkpointing, and logging functionality.

The training process follows best practices like:
- Gradient clipping to prevent exploding gradients
- Learning rate scheduling with cosine annealing
- Model checkpointing to save best models and regular snapshots
- Wandb integration for experiment tracking
- Progress bars with live metrics
"""

import os

import numpy as np
import tiktoken
import torch
import wandb
from components.gpt2 import GPT2
from constants import GPT_CONFIG_124M
from dataset import create_dataloaders
from datasets import load_dataset
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


class Trainer:
    """
    Trainer class that handles the training of a GPT2 model.

    This class encapsulates all the functionality needed for training including:
    - Forward and backward passes
    - Loss computation
    - Metrics tracking
    - Model checkpointing
    - Validation
    - Progress logging

    The trainer supports both CPU and GPU training, with automatic device detection.
    It also integrates with Weights & Biases for experiment tracking.
    """

    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, config):
        """Initialize the trainer.

        Args:
            model (GPT2): The GPT2 model instance to train
            train_loader (DataLoader): DataLoader for training data
            val_loader (DataLoader): DataLoader for validation data
            optimizer (torch.optim.Optimizer): The optimizer instance
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
            config (dict): Configuration dictionary containing training parameters like:
                - num_epochs: Number of training epochs
                - max_grad_norm: Maximum gradient norm for clipping
                - save_dir: Directory to save model checkpoints
                - save_every: Frequency of saving regular checkpoints
                - use_wandb: Whether to use Weights & Biases logging
                - run_name: Name of the training run
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        # Set up device - use GPU/MPS if available, else CPU
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device)

        # Initialize wandb for experiment tracking if enabled
        if self.config.get("use_wandb", False):
            wandb.init(
                project="gpt2-training",
                config=config,
                name=config.get("run_name", "gpt2-run"),
            )

    def compute_loss(self, logits, targets):
        """Compute both cross-entropy loss and perplexity.

        This method handles reshaping the logits and targets appropriately before
        computing the loss. It also computes perplexity which is exp(loss).

        Args:
            logits (torch.Tensor): Model output logits of shape (B, T, vocab_size)
                where B is batch size, T is sequence length
            targets (torch.Tensor): Target token ids of shape (B, T)

        Returns:
            tuple: (loss, perplexity)
                - loss: Cross entropy loss value
                - perplexity: Perplexity value (exp of loss)
        """
        # Reshape logits and targets for loss computation
        B, T, V = logits.shape  # batch_size, sequence_length, vocab_size
        logits = logits.view(B * T, V)
        targets = targets.view(B * T)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, targets)

        # Compute perplexity (exp of loss)
        perplexity = torch.exp(loss)

        return loss, perplexity

    def train_epoch(self, epoch):
        """Train the model for one epoch.

        This method:
        1. Sets model to training mode
        2. Iterates through batches of data
        3. Performs forward and backward passes
        4. Updates model parameters
        5. Tracks and logs metrics

        Args:
            epoch (int): Current epoch number

        Returns:
            tuple: (avg_loss, avg_perplexity) for the epoch
        """
        self.model.train()
        total_loss = 0
        total_perplexity = 0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch}")

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(inputs)

            # Compute loss and perplexity
            loss, perplexity = self.compute_loss(logits, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config["max_grad_norm"]
            )

            # Optimizer and scheduler steps
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # Update metrics
            total_loss += loss.item()
            total_perplexity += perplexity.item()

            # Update progress bar with current metrics
            progress_bar.set_postfix(
                {"loss": f"{loss.item():.4f}", "ppl": f"{perplexity.item():.4f}"}
            )

            # Log batch metrics to wandb if enabled
            if self.config.get("use_wandb", False):
                wandb.log(
                    {
                        "train/batch_loss": loss.item(),
                        "train/batch_perplexity": perplexity.item(),
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                )

        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        avg_perplexity = total_perplexity / num_batches

        return avg_loss, avg_perplexity

    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model on the validation set.

        This method:
        1. Sets model to evaluation mode
        2. Disables gradient computation
        3. Computes loss and perplexity on validation data

        Args:
            epoch (int): Current epoch number

        Returns:
            tuple: (avg_loss, avg_perplexity) for the validation set
        """
        self.model.eval()
        total_loss = 0
        total_perplexity = 0
        num_batches = len(self.val_loader)

        progress_bar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch}")

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            logits = self.model(inputs)

            # Compute loss and perplexity
            loss, perplexity = self.compute_loss(logits, targets)

            # Update metrics
            total_loss += loss.item()
            total_perplexity += perplexity.item()

            # Update progress bar
            progress_bar.set_postfix(
                {"loss": f"{loss.item():.4f}", "perplexity": f"{perplexity.item():.4f}"}
            )

        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        avg_perplexity = total_perplexity / num_batches

        return avg_loss, avg_perplexity

    def train(self):
        """Main training loop.

        This method orchestrates the entire training process:
        1. Runs training for specified number of epochs
        2. Performs validation after each epoch
        3. Logs metrics
        4. Saves model checkpoints

        The training loop saves:
        - Best model based on validation loss
        - Regular checkpoints at specified intervals
        """
        best_val_loss = float("inf")

        for epoch in range(self.config["num_epochs"]):
            # Training phase
            train_loss, train_ppl = self.train_epoch(epoch)

            # Validation phase
            val_loss, val_ppl = self.validate(epoch)

            # Log epoch metrics to wandb if enabled
            if self.config.get("use_wandb", False):
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
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                    },
                    f"{self.config['save_dir']}/best_model.pt",
                )

            # Save regular checkpoints at specified intervals
            if epoch % self.config["save_every"] == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                    },
                    f"{self.config['save_dir']}/checkpoint_epoch_{epoch}.pt",
                )

            # Print epoch metrics
            print(
                f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.4f}"
            )


# Training script entry point
if __name__ == "__main__":
    # Define training configuration
    config = {
        "num_epochs": 10,  # Total number of training epochs
        "learning_rate": 3e-4,  # Initial learning rate
        "max_grad_norm": 1.0,  # Maximum gradient norm for clipping
        "warmup_steps": 1000,  # Number of warmup steps for learning rate
        "save_dir": "checkpoints",  # Directory to save model checkpoints
        "save_every": 1,  # Save checkpoints every N epochs
        "run_name": "gpt2-pg-essays",  # Name of this training run
        "batch_size": 6,  # Batch size for training
        "use_wandb": True,  # Whether to use Weights & Biases logging
    }

    # Update config with model architecture parameters
    config.update(GPT_CONFIG_124M)

    # Create directory for saving checkpoints
    os.makedirs(config["save_dir"], exist_ok=True)

    # Load and prepare dataset
    # Using Paul Graham essays dataset with 3k tokens per essay
    # Selecting 10 random essays for experimental training
    dataset_raw = load_dataset("sgoel9/paul_graham_essays", split="train")
    dataset_raw = dataset_raw.select(np.random.randint(0, len(dataset_raw), 10))
    tokenizer = tiktoken.get_encoding("gpt2")

    # Initialize model and create data loaders
    model = GPT2(config)
    train_loader, val_loader = create_dataloaders(
        dataset_raw=dataset_raw,
        tokenizer=tokenizer,
        context_len=GPT_CONFIG_124M["context_len"],
        batch_size=config["batch_size"],
    )

    # Initialize optimizer with AdamW and cosine learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"])

    # Create and initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
    )

    # Begin training
    trainer.train()
