"""
Dataset Module for Training the DeepSeek Language Model.

This module handles data loading and preprocessing for text datasets,
using a sliding window approach for sequence generation.

Default dataset: TinyStories - High-quality synthetic stories perfect for
demonstrating MTP benefits on A10 GPU.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class TextDataset(Dataset):
    """
    Dataset class for text data using a sliding window approach.
    
    This dataset processes text into overlapping sequences of tokens for training.
    Each sequence is of length context_length, allowing the model to learn from
    all possible contexts in the text.
    
    The dataset uses a sliding window with stride 1:
    - For a text of length N and context length L, it creates N-L sequences
    - Each sequence overlaps with the next, shifted by 1 token
    - For each input sequence, the target is the same sequence shifted by 1 position

    For example, with context_length=4 and text "The cat sat down":
    Input sequences:    Targets:
    [The, cat, sat]    [cat, sat, down]
    [cat, sat, down]   [sat, down, <eos>]

    Args:
        raw_dataset: Iterable containing text data (e.g., Hugging Face dataset)
        tokenizer: Tokenizer for encoding text (e.g., tiktoken)
        context_length (int): Maximum sequence length for model input
    
    Example:
        >>> from datasets import load_dataset
        >>> import tiktoken
        >>> dataset = load_dataset("roneneldan/TinyStories", split="train")
        >>> tokenizer = tiktoken.get_encoding("gpt2")
        >>> text_dataset = TextDataset(dataset, tokenizer, context_length=256)
        >>> input_ids, target_ids = text_dataset[0]
        >>> print(input_ids.shape, target_ids.shape)
    """

    def __init__(self, raw_dataset, tokenizer, context_length: int):
        self.tokenizer = tokenizer
        self.context_length = context_length

        # Tokenize all texts and concatenate them
        all_tokens = []
        for item in raw_dataset:
            # Get text from dataset item (handle both dict and plain text)
            text = item["text"] if isinstance(item, dict) else str(item)
            
            # Encode text to token IDs
            tokens = self.tokenizer.encode(text)
            all_tokens.extend(tokens)
            
            # Add end-of-text token between documents to mark boundaries
            all_tokens.append(self.tokenizer.eot_token)

        # Convert to numpy array for efficient slicing during training
        self.tokens = np.array(all_tokens)

        # Calculate number of possible sequences with overlap
        # If text length is N and context length is L, we can create N-L sequences
        self.n_sequences = len(self.tokens) - context_length
        
        if self.n_sequences <= 0:
            raise ValueError(
                f"Dataset too small: {len(self.tokens)} tokens, "
                f"need at least {context_length + 1}"
            )

    def __len__(self) -> int:
        """Return the total number of sequences in the dataset."""
        return self.n_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence of tokens and its corresponding target sequence.

        For each position i, we create:
        - Input sequence: tokens[i:i+context_length]
        - Target sequence: tokens[i+1:i+context_length+1]

        The target sequence is offset by one position, as we want to predict
        the next token at each position in the input sequence.

        Args:
            idx (int): Index of the sequence to retrieve

        Returns:
            tuple: (input_sequence, target_sequence) as torch tensors of dtype long
        """
        # Extract input sequence starting at idx
        input_sequence = self.tokens[idx : idx + self.context_length]
        
        # Extract target sequence shifted by 1 position
        target_sequence = self.tokens[idx + 1 : idx + self.context_length + 1]

        # Convert to PyTorch tensors with long dtype for embeddings
        return (
            torch.from_numpy(input_sequence).long(),
            torch.from_numpy(target_sequence).long(),
        )


def create_dataloaders(
    dataset_raw,
    tokenizer,
    context_length: int,
    batch_size: int,
    train_split: float = 0.9,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    max_examples: int = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    This function handles:
    1. Splitting the raw dataset into train and validation sets
    2. Creating TextDataset objects for each split
    3. Wrapping datasets in DataLoader objects with appropriate settings

    Args:
        dataset_raw: Raw dataset containing text (e.g., Hugging Face dataset)
        tokenizer: Tokenizer for encoding text (e.g., tiktoken)
        context_length (int): Maximum sequence length for model input
        batch_size (int): Number of sequences per batch
        train_split (float): Fraction of data to use for training (default: 0.9)
        shuffle (bool): Whether to shuffle the training data (default: True)
        num_workers (int): Number of worker processes for data loading (default: 0)
        drop_last (bool): Drop incomplete batches (default: True)
        max_examples (int): Maximum number of examples to use (for fast demo training)

    Returns:
        tuple: (train_dataloader, val_dataloader)
            - train_dataloader (DataLoader): DataLoader for training data
            - val_dataloader (DataLoader): DataLoader for validation data
    
    Example:
        >>> from datasets import load_dataset
        >>> import tiktoken
        >>> dataset = load_dataset("roneneldan/TinyStories", split="train")
        >>> tokenizer = tiktoken.get_encoding("gpt2")
        >>> # Fast demo training with 10K examples
        >>> train_loader, val_loader = create_dataloaders(
        ...     dataset, tokenizer, context_length=64, batch_size=256,
        ...     max_examples=10000
        ... )
        >>> print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    """
    # Limit dataset size if requested (for fast demo training)
    if max_examples and len(dataset_raw) > max_examples:
        # Use select to create a smaller subset
        dataset_raw = dataset_raw.select(range(max_examples))
        print(f"  Using subset of {max_examples:,} examples for faster training")
    
    # Split dataset into train and validation sets
    train_size = int(train_split * len(dataset_raw))
    val_size = len(dataset_raw) - train_size
    
    if val_size == 0:
        raise ValueError(
            f"Validation set is empty with train_split={train_split}. "
            "Use a smaller train_split or larger dataset."
        )
    
    train_raw, val_raw = random_split(dataset_raw, [train_size, val_size])

    # Create dataset objects for both splits
    train_dataset = TextDataset(train_raw, tokenizer, context_length)
    val_dataset = TextDataset(val_raw, tokenizer, context_length)

    # Create and configure dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,          # Shuffle training data for better generalization
        num_workers=num_workers,  # Parallel data loading (0 for compatibility)
        pin_memory=True,          # Speed up data transfer to GPU
        drop_last=drop_last,      # Drop incomplete batches for consistent batch size
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,            # No need to shuffle validation data
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )

    return train_loader, val_loader