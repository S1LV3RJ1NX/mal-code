# Before running inference, download a checkpoint from the Hugging Face and place it in the `ch1/runs/weights` directory
# Repo: https://huggingface.co/s1lv3rj1nx/ch1


import argparse
from pathlib import Path

import torch
import torch.nn as nn
from components.transformer import build_transformer
from tokenizers.implementations import ByteLevelBPETokenizer
from utils import causal_mask


def get_config():
    """
    Returns default configuration for the translation model.

    Returns:
        dict: Configuration parameters including:
            seq_len (int): Maximum sequence length
            d_model (int): Model dimension size
            datasource (str): Directory for model artifacts
            tgt_language (str): Target language code (e.g. 'hi' for Hindi)
            model_folder (str): Subdirectory for model weights
            model_basename (str): Prefix for model checkpoint files
            tokenizer_folder (str): Directory containing tokenizer files
            vocab_size (int): Size of tokenizer vocabulary
    """
    return {
        "seq_len": 128,  # Maximum sequence length for inputs
        "d_model": 512,  # Model dimension size
        "datasource": "runs",  # Root directory for model artifacts
        "tgt_language": "hi",  # Target language (Hindi)
        "model_folder": "weights",  # Directory for model checkpoints
        "model_basename": "tmodel_",  # Prefix for checkpoint filenames
        "tokenizer_folder": "tokenizer",  # Directory for tokenizer files
        "vocab_size": 52000,  # Vocabulary size for tokenizers
    }


def load_tokenizers(config: dict):
    """
    Loads pre-trained BPE tokenizers for source and target languages.

    Args:
        config (dict): Configuration dictionary containing tokenizer paths

    Returns:
        tuple: (source_tokenizer, target_tokenizer) - Trained BPE tokenizers for both languages

    Raises:
        FileNotFoundError: If tokenizer files are not found in specified paths
    """
    tokenizer_path = Path(config["tokenizer_folder"])

    # Load English (source) tokenizer from vocab and merges files
    en_tokenizer = ByteLevelBPETokenizer(
        str(tokenizer_path / "en/vocab.json"), str(tokenizer_path / "en/merges.txt")
    )

    # Load target language tokenizer (e.g. Hindi)
    hi_tokenizer = ByteLevelBPETokenizer(
        str(tokenizer_path / f"{config['tgt_language']}/vocab.json"),
        str(tokenizer_path / f"{config['tgt_language']}/merges.txt"),
    )

    return en_tokenizer, hi_tokenizer


def translate(
    sentence: str, model: nn.Module, tokenizer_src, tokenizer_tgt, device, max_len: int
):
    """
    Translates a sentence from source to target language using the transformer model.

    Args:
        sentence (str): Input sentence in source language
        model (nn.Module): Trained transformer model
        tokenizer_src: Source language tokenizer
        tokenizer_tgt: Target language tokenizer
        device: Device to run model on (cuda/cpu)
        max_len (int): Maximum output sequence length

    Returns:
        str: Translated sentence in target language

    The translation process:
    1. Tokenize and encode input sentence
    2. Generate translation using greedy decoding
    3. Decode output tokens to target language text
    """
    # Encode source sentence and add special tokens
    encoder_input = tokenizer_src.encode(sentence)
    encoder_input_tokens = torch.tensor(
        [tokenizer_src.token_to_id("<s>")]
        + encoder_input.ids  # Add start token
        + [tokenizer_src.token_to_id("</s>")]  # Add sentence tokens  # Add end token
    ).unsqueeze(
        0
    )  # Add batch dimension

    # Create attention mask for padding
    encoder_mask = (
        (encoder_input_tokens != tokenizer_src.token_to_id("<pad>"))
        .unsqueeze(0)
        .unsqueeze(0)
        .int()
    )

    # Move tensors to target device
    encoder_input_tokens = encoder_input_tokens.to(device)
    encoder_mask = encoder_mask.to(device)

    # Generate translation using greedy decoding
    model.eval()
    with torch.no_grad():
        # Get encoder output for input sequence
        encoder_output = model.encode(encoder_input_tokens, encoder_mask)

        # Initialize decoder with start token
        decoder_input = (
            torch.empty(1, 1)
            .fill_(tokenizer_tgt.token_to_id("<s>"))
            .type_as(encoder_input_tokens)
            .to(device)
        )

        # Generate tokens one at a time
        while True:
            # Stop if max length reached
            if decoder_input.size(1) == max_len:
                break

            # Create causal mask for decoder
            decoder_mask = (
                causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)
            )

            # Get decoder output
            decoder_output = model.decode(
                decoder_input, encoder_output, encoder_mask, decoder_mask
            )

            # Project to vocabulary and get most likely token
            proj_output = model.project(decoder_output[:, -1])
            next_token = proj_output.argmax(dim=-1)

            # Add predicted token to decoder input
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)

            # Stop if end token generated
            if next_token.item() == tokenizer_tgt.token_to_id("</s>"):
                break

        # Convert output tokens to text
        output_tokens = decoder_input.squeeze(0).tolist()
        output_text = tokenizer_tgt.decode(output_tokens)

        # Remove special tokens from output
        output_text = output_text.replace("<s>", "").replace("</s>", "").strip()

        return output_text


def main():
    """
    Main function to run the translation model.

    Handles:
    1. Command line argument parsing
    2. Model and tokenizer loading
    3. Interactive translation loop
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str, help="Path to model weights", default=None
    )
    args = parser.parse_args()

    # Set up configuration and device
    config = get_config()
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load tokenizers
    print("Loading tokenizers...")
    try:
        tokenizer_src, tokenizer_tgt = load_tokenizers(config)
    except Exception as e:
        print(f"Error loading tokenizers: {str(e)}")
        print("Make sure the tokenizers are present in the tokenizer folder")
        return

    # Initialize transformer model
    print("Building model...")
    model = build_transformer(
        src_vocab_size=tokenizer_src.get_vocab_size(),
        tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
        src_seq_len=config["seq_len"],
        tgt_seq_len=config["seq_len"],
        d_model=config["d_model"],
    ).to(device)

    # Load model weights
    weights_path = (
        args.weights
        if args.weights
        else Path(f"{config['datasource']}/{config['model_folder']}/").glob(
            f"{config['model_basename']}*.pt"
        )
    )
    weights_path = (
        args.weights if args.weights else sorted(weights_path)[-1]
    )  # Get latest weights if not specified
    print(f"Loading weights from: {weights_path}")

    # Load model state with security flag
    state = torch.load(weights_path, weights_only=True, map_location=device)

    # Load state dict based on format
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)  # Direct weights when weights_only=True

    # Interactive translation loop
    print("\nEnglish to Hindi Translation")
    print("Enter 'q' or 'quit' to exit")
    print("-" * 50)

    while True:
        input_text = input("\nEnter English text: ").strip()

        # Handle exit commands
        if input_text.lower() in ["q", "quit"]:
            break

        # Skip empty input
        if not input_text:
            continue

        # Perform translation
        try:
            translation = translate(
                input_text,
                model,
                tokenizer_src,
                tokenizer_tgt,
                device,
                config["seq_len"],
            )
            print(f"Hindi translation: {translation}")
        except Exception as e:
            print(f"Translation error: {str(e)}")
        except KeyboardInterrupt:
            print("\nTranslation interrupted. Exiting...")
            break


if __name__ == "__main__":
    main()
