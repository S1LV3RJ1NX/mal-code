"""
Configuration constants for the DeepSeek Language Model.

This file defines the architecture parameters for our DeepSeek-inspired model
combining Multi-Head Latent Attention (MLA) and Mixture of Experts (MoE).
"""

# Base configuration for DeepSeek Language Model
DEEPSEEK_CONFIG = {
    # Vocabulary and Context
    "vocab_size": 50257,       # GPT-2 tokenizer vocabulary size
    "context_length": 256,     # Maximum sequence length
    
    # Model Architecture
    "emb_dim": 512,           # Embedding dimension (d_model)
    "n_heads": 8,             # Number of attention heads
    "n_layers": 6,            # Number of transformer blocks
    "drop_rate": 0.1,         # Dropout rate for regularization
    
    # MLA (Multi-Head Latent Attention) Parameters
    "kv_latent_dim": 128,     # Latent dimension for KV compression
                               # Rule of thumb: emb_dim / 4 or emb_dim / 8
                               # 512 / 4 = 128 (4× compression)
    
    # MoE (Mixture of Experts) Parameters
    "num_experts": 8,         # Total number of routed experts
    "num_shared_experts": 2,  # Number of shared experts (always activated)
    "top_k": 2,               # Number of experts to activate per token
    
    # Expert Architecture
    "expert_hidden_dim": 2048,  # Hidden dimension for each expert FFN
                                 # Typically 4× the embedding dimension
    
    # MTP (Multi-Token Prediction) Parameters
    "use_mtp": True,
    "mtp_depth": 3,
    "mtp_weight": 0.3,
    
    # Memory Optimization Parameters
    "use_gradient_checkpointing": False,  # Disable for speed (we have enough memory)
    "mixed_precision": True,             # Use FP16/BF16 for memory savings
}

# Smaller configuration for quick testing and experimentation
DEEPSEEK_CONFIG_SMALL = {
    "vocab_size": 50257,
    "context_length": 128,     # Shorter context for faster training
    "emb_dim": 256,            # Smaller embedding dimension
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.1,
    "kv_latent_dim": 64,       # 256 / 4 = 64 (4× compression)
    "num_experts": 4,
    "num_shared_experts": 1,
    "top_k": 2,
    "expert_hidden_dim": 1024,
    # MTP (Multi-Token Prediction) Parameters
    "use_mtp": True,
    "mtp_depth": 3,
    "mtp_weight": 0.3,
    # Memory Optimization Parameters
    "use_gradient_checkpointing": False,  # Disable for speed (we have enough memory)
    "mixed_precision": True,             # Use FP16/BF16 for memory savings
}