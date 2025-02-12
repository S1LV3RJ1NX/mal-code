import os
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from dataset import get_dataset
from torchmetrics.text import BLEUScore, CharErrorRate, WordErrorRate
from tqdm import tqdm
from utils import get_model, get_weights_file_path, greedy_decode


def get_config() -> dict:
    """
    Returns the default configuration dictionary for the transformer model training.

    The configuration includes parameters for:
    - Training: batch size, number of samples, epochs, learning rate
    - Model architecture: sequence length, model dimension
    - Data: source/target languages, vocabulary size
    - File paths: model weights, tokenizer files

    Returns:
        dict: Configuration with the following keys:
            batch_size (int): Number of samples per training batch
            num_samples (int): Total number of training samples to use from dataset
            num_epochs (int): Number of training epochs
            lr (float): Learning rate for optimizer
            seq_len (int): Maximum sequence length for inputs
            d_model (int): Dimension of model embeddings and hidden states
            datasource (str): Root folder for saving model artifacts
            tgt_language (str): Target language code (e.g. 'hi' for Hindi)
            model_folder (str): Subfolder for saving model weights
            model_basename (str): Prefix for model checkpoint filenames
            preload (str|None): Path to pretrained weights, None to train from scratch
            tokenizer_folder (str): Folder containing tokenizer files
            vocab_size (int): Size of BPE vocabulary for tokenizer
    """
    return {
        "batch_size": 85,  # Training batch size
        "num_samples": 1000000,  # Use 1M samples from dataset
        "num_epochs": 10,  # Number of training epochs
        "lr": 10**-4,  # Learning rate
        "seq_len": 128,  # Max sequence length
        "d_model": 512,  # Model dimension
        "datasource": "runs",  # Root folder for artifacts
        "tgt_language": "hi",  # Target language (Hindi)
        "model_folder": "weights",  # Folder for model checkpoints
        "model_basename": "tmodel_",  # Prefix for checkpoint files
        "preload": None,  # No preloaded weights
        "tokenizer_folder": "tokenizer",  # Tokenizer file location
        "vocab_size": 52000,  # BPE vocabulary size
    }


## Validation Loop
def evaluate(
    model,
    val_dataloader,
    tokenizer_tgt,
    max_len,
    device,
    print_msg,
    global_step,
    num_examples=2,
):
    """
    Evaluate the model on validation data and compute various metrics.

    This function:
    1. Runs the model in evaluation mode on validation data
    2. Performs greedy decoding to generate translations
    3. Computes and logs multiple evaluation metrics
    4. Prints example translations for manual inspection

    Args:
        model: The transformer model to evaluate
        val_dataloader: DataLoader containing validation data
        tokenizer_tgt: Tokenizer for the target language
        max_len: Maximum sequence length for generated translations
        device: Device to run evaluation on (cuda/cpu)
        print_msg: Function to print messages (allows custom logging)
        global_step: Current training step (for logging metrics)
        num_examples: Number of example translations to print

    The function computes and logs:
    - Character Error Rate (CER)
    - Word Error Rate (WER)
    - BLEU Score
    """
    # Set model to evaluation mode
    model.eval()
    count = 0

    # Lists to store texts for metric computation
    source_texts = []
    expected = []
    predicted = []

    # Try to get console width for pretty printing
    try:
        # Use system call to get terminal dimensions
        with os.popen("stty size", "r") as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # Fallback width if terminal dimensions unavailable
        console_width = 80

    # Disable gradient computation for validation
    with torch.no_grad():
        for batch in val_dataloader:
            count += 1
            # Move input tensors to target device
            encoder_input = batch["encoder_input"].to(
                device
            )  # Shape: (batch_size, seq_len)
            encoder_mask = batch["encoder_mask"].to(
                device
            )  # Shape: (batch_size, 1, 1, seq_len)

            # Validation requires batch size of 1 for consistent printing
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            # Generate translation using greedy decoding
            model_out = greedy_decode(
                model=model,
                encoder_input=encoder_input,
                encoder_mask=encoder_mask,
                tokenizer_tgt=tokenizer_tgt,
                max_len=max_len,
                device=device,
            )

            # Extract texts for this example
            source_text = batch["src_text"][0]  # Original source text
            target_text = batch["tgt_text"][0]  # Expected translation
            model_out_text = tokenizer_tgt.decode(
                model_out.detach().cpu().numpy()
            )  # Model's translation

            # Store for metric computation
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print example translations
            print_msg("-" * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            # Stop after printing requested number of examples
            if count == num_examples:
                print_msg("-" * console_width)
                break

    # Compute and log evaluation metrics

    # Character Error Rate
    metric = CharErrorRate()
    cer = metric(predicted, expected)
    wandb.log({"validation/cer": cer, "global_step": global_step})

    # Word Error Rate
    metric = WordErrorRate()
    wer = metric(predicted, expected)
    wandb.log({"validation/wer": wer, "global_step": global_step})

    # BLEU Score
    metric = BLEUScore()
    bleu = metric(predicted, expected)
    wandb.log({"validation/BLEU": bleu, "global_step": global_step})


def train_model(config: dict):
    """
    Train a Transformer model for sequence-to-sequence tasks.

    This function handles the complete training loop including:
    - Setting up the device (GPU/CPU)
    - Loading/initializing the model and optimizer
    - Training loop with batches
    - Validation after each epoch
    - Model checkpointing
    - Logging metrics to Weights & Biases

    Args:
        config (dict): Configuration dictionary containing model and training parameters
                      including learning rate, number of epochs, batch size etc.
    """
    # Set up the device - prefer CUDA GPU, fallback to MPS (Apple Silicon), then CPU
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Create model weights directory if it doesn't exist
    Path(f"{config['datasource']}/{config['model_folder']}").mkdir(
        parents=True, exist_ok=True
    )

    # Get data loaders and tokenizers
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)

    # Initialize model and move to device
    model = get_model(
        config=config,
        vocab_src_len=tokenizer_src.get_vocab_size(),
        vocab_tgt_len=tokenizer_tgt.get_vocab_size(),
    ).to(device)

    # Initialize Adam optimizer with specified learning rate and numerical stability factor
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    # Training state variables
    initial_epoch = 0
    global_step = 0

    # Load pre-trained model if specified
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)

        # Resume training from saved state
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
        del state
    else:
        print("No pre-trained model found. Starting from scratch.")

    # Initialize loss function with label smoothing
    # Ignore padding tokens in loss calculation
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("<pad>"), label_smoothing=0.1
    ).to(device)

    # Set up Weights & Biases logging
    wandb.define_metric("global_step")
    wandb.define_metric("validation/*", step_metric="global_step")
    wandb.define_metric("train/*", step_metric="global_step")

    # Main training loop
    for epoch in range(initial_epoch, config["num_epochs"]):
        torch.cuda.empty_cache()  # Clear GPU memory
        model.train()

        # Initialize progress bar for current epoch
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")

        # Process each batch
        for batch in batch_iterator:
            # Move batch data to device
            encoder_input = batch["encoder_input"].to(device)  # Source sequence
            decoder_input = batch["decoder_input"].to(device)  # Target sequence
            encoder_mask = batch["encoder_mask"].to(device)  # Source attention mask
            decoder_mask = batch["decoder_mask"].to(device)  # Target attention mask

            # Forward pass through the model
            encoder_output = model.encode(src=encoder_input, src_mask=encoder_mask)
            decoder_output = model.decode(
                tgt=decoder_input,
                encoder_output=encoder_output,
                src_mask=encoder_mask,
                tgt_mask=decoder_mask,
            )
            proj = model.project(decoder_output)  # Project to vocabulary size

            # Move labels to device and compute loss
            label = batch["label"].to(device)
            loss = loss_fn(
                proj.view(-1, tokenizer_tgt.get_vocab_size()).float(),
                label.view(-1).long(),
            )

            # Update progress bar with current loss
            batch_iterator.set_postfix({"loss": f"{loss.item():.3f}"})

            # Log training metrics
            wandb.log({"train/loss": loss.item(), "global_step": global_step})

            # Backpropagation and optimization step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # Clear gradients

            global_step += 1

        # Run validation after each epoch
        evaluate(
            model,
            val_dataloader,
            tokenizer_tgt,
            config["seq_len"],
            device,
            lambda msg: batch_iterator.write(msg),
            global_step,
        )

        # Save model checkpoint
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        config = get_config()
        wandb.init(project="ch1", config=config)
        train_model(config)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Cleaning up...")
    except Exception as e:
        print(f"\nError occurred: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if wandb.run is not None:
            wandb.finish()
