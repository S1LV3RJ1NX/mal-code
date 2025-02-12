import datasets
import numpy as np
import torch
from datasets import load_dataset
from tokenizers.implementations import ByteLevelBPETokenizer
from torch.utils.data import DataLoader, Dataset, random_split
from utils import causal_mask, get_or_build_tokenizer


class BilingualDataset(Dataset):
    """
    PyTorch Dataset for parallel text data in English and Hindi.
    """

    def __init__(
        self,
        dataset: datasets.Dataset,
        tokenizer_source: ByteLevelBPETokenizer,
        tokenizer_target: ByteLevelBPETokenizer,
        seq_len: int,
    ):
        self.dataset = dataset
        self.tokenizer_source = tokenizer_source
        self.tokenizer_target = tokenizer_target
        self.seq_len = seq_len
        self.skipped_sequences = 0
        self.total_requests = 0

        # Keep track of valid/invalid indices
        self.index_cache = {}  # Cache validation results

        # Special token IDs
        self.sos_token = tokenizer_source.token_to_id("<s>")
        self.eos_token = tokenizer_source.token_to_id("</s>")
        self.pad_token = tokenizer_source.token_to_id("<pad>")

    def find_valid_index(self, idx: int) -> int:
        """Find the next valid index starting from idx, using random jumps to avoid duplicates"""
        attempts = 0
        tried_indices = set()

        while attempts < len(self.dataset):  # Limit attempts to dataset size
            # Check if we've already validated this index
            if idx in self.index_cache:
                if self.index_cache[idx]:  # If it's valid
                    return idx
                # If invalid, do a random jump
                idx = (idx + np.random.randint(1, len(self.dataset))) % len(
                    self.dataset
                )
                continue

            try:
                src_text = self.dataset[idx]["src"]
                tgt_text = self.dataset[idx]["tgt"]

                # Check sequence lengths
                encoder_input_tokens = self.tokenizer_source.encode(src_text).ids
                decoder_input_tokens = self.tokenizer_target.encode(tgt_text).ids

                is_valid = (
                    len(encoder_input_tokens) <= self.seq_len
                    and len(decoder_input_tokens) <= self.seq_len
                    and len(encoder_input_tokens) + 2 <= self.seq_len
                    and len(decoder_input_tokens) + 1 <= self.seq_len
                )

                # Cache the result
                self.index_cache[idx] = is_valid

                if is_valid:
                    return idx

            except Exception as e:
                self.index_cache[idx] = False

            # If we get here, the current index was invalid
            tried_indices.add(idx)

            # Do a random jump to avoid sequential duplicates
            remaining_indices = set(range(len(self.dataset))) - tried_indices
            if not remaining_indices:
                raise RuntimeError(
                    f"No valid sequences found after trying {attempts} indices"
                )

            idx = np.random.choice(list(remaining_indices))
            attempts += 1

        raise RuntimeError(f"Could not find valid sequence after {attempts} attempts")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        self.total_requests += 1

        # Find a valid index
        valid_idx = self.find_valid_index(idx)

        # Get the valid sequence
        src_target_pair = self.dataset[valid_idx]
        src_text = src_target_pair["src"]
        tgt_text = src_target_pair["tgt"]

        # Process the valid sequence
        encoder_input_tokens = self.tokenizer_source.encode(src_text).ids
        decoder_input_tokens = self.tokenizer_target.encode(tgt_text).ids

        # Calculate padding
        total_encoder_padding = self.seq_len - len(encoder_input_tokens) - 2
        total_decoder_padding = self.seq_len - len(decoder_input_tokens) - 1

        # Build tensors
        encoder_input = torch.cat(
            [
                torch.tensor([self.sos_token]),
                torch.tensor(encoder_input_tokens, dtype=torch.int64),
                torch.tensor([self.eos_token]),
                torch.tensor([self.pad_token] * total_encoder_padding),
            ],
            dim=0,
        )

        decoder_input = torch.cat(
            [
                torch.tensor([self.sos_token]),
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * total_decoder_padding),
            ],
            dim=0,
        )

        label = torch.cat(
            [
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                torch.tensor([self.eos_token]),
                torch.tensor([self.pad_token] * total_decoder_padding),
            ],
            dim=0,
        )

        if self.total_requests % 10000 == 0:
            print(f"\nDataset Stats:")
            print(f"Total requests: {self.total_requests}")
            print(f"Cache size: {len(self.index_cache)}")
            print(f"Valid sequences in cache: {sum(self.index_cache.values())}")

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),
            "decoder_mask": (decoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int()
            & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def get_dataset(config: dict):
    """Load and prepare the Samanantar dataset"""
    dataset_raw = load_dataset(
        "ai4bharat/samanantar", config["tgt_language"], split="train"
    )

    # Get only random 1M samples
    dataset_raw = dataset_raw.select(range(config["num_samples"]))

    # Build tokenizers
    tokenizer_source = get_or_build_tokenizer(config, dataset_raw, "en")
    tokenizer_target = get_or_build_tokenizer(
        config, dataset_raw, config["tgt_language"]
    )

    # Split dataset into train and validation sets (90-10)
    train_ds_size = int(0.9 * len(dataset_raw))
    val_ds_size = len(dataset_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(dataset_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(
        train_ds_raw, tokenizer_source, tokenizer_target, config["seq_len"]
    )
    val_ds = BilingualDataset(
        val_ds_raw, tokenizer_source, tokenizer_target, config["seq_len"]
    )

    train_dataloader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_source, tokenizer_target
