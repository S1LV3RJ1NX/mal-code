from pathlib import Path

import datasets
import torch
from components.transformer import build_transformer
from tokenizers.implementations import ByteLevelBPETokenizer
from tqdm import tqdm


def get_or_build_tokenizer(
    config: dict, dataset: datasets.Dataset, language: str
) -> ByteLevelBPETokenizer:
    """
    Gets an existing tokenizer or builds and trains a new one.

    Args:
        config (dict): Configuration dictionary containing tokenizer path
        dataset: Dataset containing sentences to train tokenizer on
        language (str): Language code. Source language is always 'en'

    Returns:
        ByteLevelBPETokenizer: A trained tokenizer object
    """
    # Check if tokenizer folder exists, if not create it
    Path(config["tokenizer_folder"]).mkdir(parents=True, exist_ok=True)

    tokenizer_path = Path(config["tokenizer_folder"]) / language
    vocab_path = tokenizer_path / "vocab.json"
    merges_path = tokenizer_path / "merges.txt"

    if not (vocab_path.exists() and merges_path.exists()):
        # Create a temporary file to store the sentences
        temp_file = Path(config["tokenizer_folder"]) / f"temp_{language}.txt"

        # Write sentences to temporary file
        with open(temp_file, "w", encoding="utf-8") as f:
            for item in tqdm(dataset):
                f.write(item["src" if language == "en" else "tgt"] + "\n")

        # Initialize a new tokenizer
        tokenizer = ByteLevelBPETokenizer()

        # Train the tokenizer
        tokenizer.train(
            files=[str(temp_file)],
            vocab_size=config["vocab_size"],
            min_frequency=2,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
        )

        # Save the tokenizer
        tokenizer_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_model(str(tokenizer_path))

        # Clean up temporary file
        temp_file.unlink()

    # Load the tokenizer
    return ByteLevelBPETokenizer(str(vocab_path), str(merges_path))


def causal_mask(size: int) -> torch.Tensor:
    """
    Generates a causal mask for a given sequence length.

    Args:
        size (int): Length of the sequence to generate mask for

    Returns:
        torch.Tensor: A boolean tensor of shape (1, size, size) where True values allow attention
                     and False values prevent attention. The upper triangle is False to ensure
                     causal/autoregressive attention.
    """
    # Create a square matrix where each position (i,j) contains 1 if j<=i and 0 otherwise
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).int()
    # Convert to boolean mask where 1->False and 0->True
    return mask == 0


def get_model(config: dict, vocab_src_len: int, vocab_tgt_len: int):
    model = build_transformer(
        src_vocab_size=vocab_src_len,
        tgt_vocab_size=vocab_tgt_len,
        src_seq_len=config["seq_len"],
        tgt_seq_len=config["seq_len"],
        d_model=config["d_model"],
    )
    return model


# Function to construct the path for saving and retrieving model weights
def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}/{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)


# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}/{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])


# Define function to obtain the most probable next token
def greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device):
    """
    Performs greedy decoding to generate target sequence from encoder input.

    This implements an auto-regressive decoding process where at each step:
    1. The encoder output is computed once for the input sequence
    2. The decoder generates one token at a time, using previous tokens as context
    3. The most probable next token is selected greedily at each step
    4. Generation stops at max_len or when end token is produced

    Args:
        model: The transformer model for translation
        encoder_input (Tensor): Input sequence tensor [batch_size, src_seq_len]
        encoder_mask (Tensor): Mask for input sequence [batch_size, 1, src_seq_len]
        tokenizer_tgt: Target language tokenizer with token_to_id mapping
        max_len (int): Maximum length of generated sequence
        device: Device to run generation on (cuda/cpu)

    Returns:
        Tensor: Generated sequence of token ids [seq_len]
    """
    # Get special token ids from target tokenizer
    sos_idx = tokenizer_tgt.token_to_id("<s>")  # Start of sequence token
    eos_idx = tokenizer_tgt.token_to_id("</s>")  # End of sequence token

    # Run input through encoder once since output will be reused
    encoder_output = model.encode(
        encoder_input, encoder_mask
    )  # [batch_size, src_seq_len, d_model]

    # Initialize decoder input with start token
    decoder_input = (
        torch.empty(1, 1).fill_(sos_idx).type_as(encoder_input).to(device)
    )  # [1, 1]

    while True:
        # Stop if max length reached
        if decoder_input.size(1) == max_len:
            break

        # Create causal mask to prevent attending to future tokens
        decoder_mask = (
            causal_mask(decoder_input.size(1))
            .type_as(encoder_mask)
            .unsqueeze(0)
            .to(device)  # [1, 1, tgt_seq_len, tgt_seq_len]
        )

        # Generate next token probabilities
        decoder_output = model.decode(
            tgt=decoder_input,
            encoder_output=encoder_output,
            src_mask=encoder_mask,
            tgt_mask=decoder_mask,
        )  # [batch_size, tgt_seq_len, d_model]

        # Project to vocabulary size and get most likely token
        prob = model.project(decoder_output[:, -1])  # [batch_size, vocab_size]
        _, next_word = torch.max(prob, dim=1)  # [batch_size]

        # Append predicted token to decoder input
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1)
                .type_as(encoder_input)
                .fill_(next_word.item())
                .to(device),
            ],
            dim=1,
        )  # [1, tgt_seq_len + 1]

        # Stop if end token generated
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)  # Return sequence without batch dimension
