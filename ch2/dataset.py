import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class PaulGrahamDataset(Dataset):
    """Dataset class for Paul Graham essays.

    This dataset processes text into overlapping sequences of tokens for training
    the GPT-2 model using a sliding window approach. Each sequence is of length
    context_len, allowing the model to learn from all possible contexts in the text.

    The dataset uses a sliding window approach to create training examples:
    - For a text of length N and context length L, it creates N-L sequences
    - Each sequence overlaps with the next, shifted by 1 token
    - For each input sequence, the target is the same sequence shifted by 1 position

    For example, with context_len=4 and text "The cat sat down":
    Input sequences:    Targets:
    [The, cat, sat]    [cat, sat, down]
    [cat, sat, down]   [sat, down, <eos>]

    This approach allows the model to:
    1. Learn from all possible contexts in the text
    2. Predict the next token at every position
    3. Maintain contextual understanding across token sequences
    """

    def __init__(self, raw_dataset, tokenizer, context_len):
        """Initialize the dataset.

        Args:
            raw_dataset: Hugging Face dataset containing Paul Graham essays
            tokenizer: tiktoken tokenizer for GPT-2 tokenization
            context_len: Maximum sequence length for the model input

        The initialization process:
        1. Tokenizes each essay in the dataset
        2. Concatenates all tokenized essays with EOS tokens between them
        3. Converts tokens to numpy array for efficient slicing
        4. Calculates total number of possible sequences
        """
        self.tokenizer = tokenizer
        self.context_len = context_len

        # Tokenize all essays and concatenate them
        all_tokens = []
        for item in raw_dataset:
            # Convert each essay to token IDs
            tokens = self.tokenizer.encode(item["text"])
            all_tokens.extend(tokens)
            # Add EOS token between essays to mark boundaries
            all_tokens.append(self.tokenizer.eot_token)

        # Convert to numpy array for efficient slicing during training
        self.tokens = np.array(all_tokens)

        # Calculate number of possible sequences with overlap
        # If text length is N and context length is L, we can create N-L sequences
        self.n_sequences = len(self.tokens) - context_len

    def __len__(self):
        """Return the total number of sequences in the dataset."""
        return self.n_sequences

    def __getitem__(self, idx):
        """Get a sequence of tokens and its corresponding target sequence.

        For each position i, we create:
        - Input sequence: tokens[i:i+context_len]
        - Target sequence: tokens[i+1:i+context_len+1]

        The target sequence is offset by one position, as we want to predict
        the next token at each position in the input sequence.

        Args:
            idx: Index of the sequence to retrieve

        Returns:
            tuple: (input_sequence, target_sequence) as torch tensors
        """
        # Extract input sequence starting at idx
        input_sequence = self.tokens[idx : idx + self.context_len]
        # Extract target sequence shifted by 1 position
        target_sequence = self.tokens[idx + 1 : idx + self.context_len + 1]

        # Convert to PyTorch tensors and ensure long dtype for embeddings
        return (
            torch.from_numpy(input_sequence).long(),
            torch.from_numpy(target_sequence).long(),
        )


def create_dataloaders(
    dataset_raw,
    tokenizer,
    context_len,
    batch_size,
    train_split=0.9,
    shuffle=True,
    num_workers=4,
):
    """Create train and validation dataloaders.

    This function handles:
    1. Splitting the raw dataset into train and validation sets
    2. Creating PaulGrahamDataset objects for each split
    3. Wrapping datasets in DataLoader objects with appropriate settings

    Args:
        dataset_raw: Raw Hugging Face dataset containing essays
        tokenizer: tiktoken tokenizer for GPT-2
        context_len: Maximum sequence length for model input
        batch_size: Number of sequences per batch
        train_split: Fraction of data to use for training (default: 0.9)
        shuffle: Whether to shuffle the training data (default: True)
        num_workers: Number of worker processes for data loading (default: 4)

    Returns:
        tuple: (train_dataloader, val_dataloader)
            - train_dataloader: DataLoader for training data
            - val_dataloader: DataLoader for validation data
    """
    # Split dataset into train and validation sets
    train_size = int(train_split * len(dataset_raw))
    val_size = len(dataset_raw) - train_size
    train_raw, val_raw = random_split(dataset_raw, [train_size, val_size])

    # Create dataset objects for both splits
    train_dataset = PaulGrahamDataset(train_raw, tokenizer, context_len)
    val_dataset = PaulGrahamDataset(val_raw, tokenizer, context_len)

    # Create and configure dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,  # Shuffle training data for better generalization
        num_workers=num_workers,  # Parallel data loading
        pin_memory=True,  # Speed up data transfer to GPU
        drop_last=True,  # Drop incomplete batches
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, val_loader
