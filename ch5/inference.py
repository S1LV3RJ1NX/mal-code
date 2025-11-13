"""
Inference Module for the DeepSeek Language Model.

This module provides utilities for loading trained models and generating text.
Supports both standard and FP8 quantized models.
"""

import torch
import tiktoken
from pathlib import Path

from components.deepseek import DeepSeekModel
from utils import encode_text, decode_text, generate, print_model_info

# Import quantization utilities
try:
    from components.quantization import convert_linear_to_quantized, check_fp8_support
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False
    print("Note: Quantization module not available. Standard precision will be used.")


class TextGenerator:
    """
    Text generator for the DeepSeek Language Model.
    
    This class handles loading a trained model and generating text with
    various sampling strategies. Automatically supports FP8 quantized models.
    
    Args:
        model_path (str): Path to the saved model checkpoint
        device (str): Device to run inference on ('cuda', 'mps', or 'cpu')
        tokenizer_name (str): Name of the tokenizer to use (default: 'gpt2')
    
    Example:
        >>> generator = TextGenerator('checkpoints/best_model.pt', device='cuda')
        >>> text = generator.generate(
        ...     prompt="Once upon a time",
        ...     max_new_tokens=100,
        ...     temperature=0.8
        ... )
        >>> print(text)
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = None,
        tokenizer_name: str = "gpt2"
    ):
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        print(f"Loaded tokenizer: {tokenizer_name}")
        
        # Load model
        self.model, self.config = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from: {model_path}")
        print_model_info(self.model, self.config)
    
    def _load_model(self, model_path: str) -> tuple:
        """
        Load model from checkpoint.
        
        Automatically detects and applies quantization if the model was trained with it.
        
        Args:
            model_path (str): Path to checkpoint file
        
        Returns:
            tuple: (model, config)
        """
        checkpoint_path = Path(model_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        
        print("\n" + "="*60)
        print("LOADING MODEL")
        print("="*60)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract model config
        if "model_config" in checkpoint:
            # New format: separate model and training configs
            config = checkpoint["model_config"]
            print("  Using model config from checkpoint")
        elif "config" in checkpoint:
            # Check if this is a model config or training config
            checkpoint_config = checkpoint["config"]
            if "vocab_size" in checkpoint_config and "emb_dim" in checkpoint_config:
                # This is a model config
                config = checkpoint_config
                print("  Using model config from checkpoint")
            elif "num_epochs" in checkpoint_config:
                # This is a training config, try to infer model config
                print("  Detected old checkpoint format with training config only")
                print("  Attempting to infer model config...")
                
                # Try to infer from constants - check if this looks like SMALL config
                from constants import DEEPSEEK_CONFIG_SMALL, DEEPSEEK_CONFIG
                
                # Use SMALL config as default for existing checkpoints
                config = DEEPSEEK_CONFIG_SMALL.copy()
                print("  Using DEEPSEEK_CONFIG_SMALL as fallback model config")
                print("  Note: If this fails, please retrain with updated trainer")
            else:
                raise ValueError("Unknown config format in checkpoint")
        else:
            raise ValueError("No config found in checkpoint")
        
        # Print model info
        print(f"Model configuration:")
        print(f"  Layers: {config['n_layers']}")
        print(f"  Embedding dim: {config['emb_dim']}")
        print(f"  Context length: {config['context_length']}")
        print(f"  Vocab size: {config['vocab_size']}")
        print(f"  Experts: {config['num_experts']} routed + {config['num_shared_experts']} shared")
        
        # Create model
        model = DeepSeekModel(config)
        
        # Check if model was trained with quantization
        quantization_enabled = checkpoint.get("quantization_enabled", False)
        
        # Check if checkpoint actually has quantized weights by inspecting keys
        sample_keys = list(checkpoint["model_state_dict"].keys())
        has_quantized_weights = any('.te_linear.' in key for key in sample_keys)
        
        if quantization_enabled and has_quantized_weights:
            # Case 1: Checkpoint has quantized weights (trained with quantization properly applied)
            if QUANTIZATION_AVAILABLE:
                print("\n✓ Quantization: Enabled (checkpoint has FP8 quantized weights)")
                supports_fp8, device_info = check_fp8_support()
                print(f"  {device_info}")
                
                # Convert model structure to match quantized checkpoint
                model = convert_linear_to_quantized(model, use_te=supports_fp8)
                print("  ✓ Quantized model structure created")
                
                # Load quantized weights
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                print("  ✓ Quantized weights loaded successfully")
                
                # Ensure model is in BF16 for Transformer Engine
                model = model.to(torch.bfloat16)
                print("  ✓ Model prepared for inference")
            else:
                print("\n⚠ Warning: Model was trained with quantization but module not available")
                print("  Cannot load quantized checkpoint without quantization support")
                print("  Please install transformer-engine or retrain without quantization")
                raise RuntimeError("Quantization module required to load this checkpoint")
        
        elif quantization_enabled and not has_quantized_weights:
            # Case 2: Flag says quantized but weights are standard (bug in old trainer)
            print("\n⚠ Note: Checkpoint marked as quantized but contains standard weights")
            print("  Loading as standard precision model")
            
            # Load weights directly (they're standard weights)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            print("  ✓ Weights loaded successfully (standard precision)")
        
        else:
            # Case 3: Standard precision model
            print("\n✓ Quantization: Not used (standard precision)")
            
            # Load weights directly
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            print("  ✓ Weights loaded successfully")
        
        print("="*60 + "\n")
        
        return model, config
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = None,
        verbose: bool = True
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt (str): The starting text prompt
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature (higher = more random)
                - 0.0: Greedy decoding (always pick most likely)
                - 0.7-0.9: Creative but coherent
                - 1.0: Sample from model distribution
                - >1.0: More random and diverse
            top_k (int, optional): If set, only sample from top k tokens
            verbose (bool): Print generation info
        
        Returns:
            str: Generated text
        
        Example:
            >>> generator = TextGenerator('checkpoints/best_model.pt')
            >>> text = generator.generate(
            ...     prompt="The best way to learn programming is",
            ...     max_new_tokens=50,
            ...     temperature=0.8,
            ...     top_k=40
            ... )
        """
        if verbose:
            print("\n" + "="*60)
            print("TEXT GENERATION")
            print("="*60)
            print(f"Prompt: {prompt}")
            print(f"Max new tokens: {max_new_tokens}")
            print(f"Temperature: {temperature}")
            print(f"Top-k: {top_k}")
            print("-"*60)
        
        # Encode prompt
        input_ids = encode_text(prompt, self.tokenizer).to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = generate(
                model=self.model,
                idx=input_ids,
                max_new_tokens=max_new_tokens,
                context_size=self.config["context_length"],
                temperature=temperature,
                top_k=top_k,
                eos_id=self.tokenizer.eot_token
            )
        
        # Decode
        generated_text = decode_text(output_ids, self.tokenizer)
        
        if verbose:
            print("\nGenerated text:")
            print("-"*60)
            print(generated_text)
            print("="*60)
        
        return generated_text
    
    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = None
    ) -> list[str]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts (list[str]): List of prompts
            max_new_tokens (int): Maximum tokens to generate per prompt
            temperature (float): Sampling temperature
            top_k (int, optional): Top-k sampling parameter
        
        Returns:
            list[str]: List of generated texts
        """
        results = []
        for i, prompt in enumerate(prompts):
            print(f"\nGenerating {i+1}/{len(prompts)}...")
            text = self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                verbose=False
            )
            results.append(text)
        
        return results
    
    def interactive_generation(self):
        """
        Interactive text generation loop.
        
        Allows the user to continuously enter prompts and generate text
        until they choose to exit.
        """
        print("\n" + "="*60)
        print("INTERACTIVE TEXT GENERATION")
        print("="*60)
        print("Enter your prompts below. Type 'quit' or 'exit' to stop.")
        print("Type 'settings' to adjust generation parameters.")
        print("="*60)
        
        # Default settings
        max_new_tokens = 100
        temperature = 0.8
        top_k = 40
        
        while True:
            print("\n" + "-"*60)
            prompt = input("\nPrompt: ").strip()
            
            # Check for exit commands
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Exiting interactive mode. Goodbye!")
                break
            
            # Check for settings command
            if prompt.lower() == 'settings':
                print("\nCurrent settings:")
                print(f"  max_new_tokens: {max_new_tokens}")
                print(f"  temperature: {temperature}")
                print(f"  top_k: {top_k}")
                
                # Allow user to update settings
                try:
                    new_tokens = input(f"Max new tokens [{max_new_tokens}]: ").strip()
                    if new_tokens:
                        max_new_tokens = int(new_tokens)
                    
                    new_temp = input(f"Temperature [{temperature}]: ").strip()
                    if new_temp:
                        temperature = float(new_temp)
                    
                    new_top_k = input(f"Top-k [{top_k}]: ").strip()
                    if new_top_k:
                        top_k = int(new_top_k) if new_top_k.lower() != 'none' else None
                    
                    print("\nSettings updated!")
                except ValueError as e:
                    print(f"Invalid input: {e}")
                
                continue
            
            # Skip empty prompts
            if not prompt:
                print("Please enter a prompt.")
                continue
            
            # Generate text
            try:
                self.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    verbose=True
                )
            except Exception as e:
                print(f"Error during generation: {e}")
                continue


def main():
    """
    Main function for command-line inference.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate text using a trained DeepSeek model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (0.0 = greedy, higher = more random)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k sampling parameter (None to disable)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/mps/cpu)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = TextGenerator(
        model_path=args.model_path,
        device=args.device
    )
    
    # Interactive mode
    if args.interactive:
        generator.interactive_generation()
    
    # Single generation mode
    elif args.prompt:
        generator.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            verbose=True
        )
    
    # No prompt provided
    else:
        print("Please provide either --prompt or --interactive flag")
        print("Run with --help for usage information")


if __name__ == "__main__":
    main()