from components.common import torch, nn
from components.transformer import TransformerBlock

class GPT2(nn.Module):
    """GPT-2 language model implementation.
    
    This implements the GPT-2 architecture from "Language Models are Unsupervised Multitask Learners" 
    (Radford et al., 2019). The model uses a decoder-only transformer architecture with learned 
    positional embeddings and causal self-attention.

    The model processes input tokens through:
    1. Token embeddings + Learned positional embeddings
    2. Multiple transformer decoder blocks
    3. Layer normalization
    4. Final projection to vocabulary size
    """

    def __init__(self, config: dict):
        """Initialize the GPT-2 model.

        Args:
            config (dict): Model configuration containing:
                - vocab_size: Size of token vocabulary
                - context_len: Maximum context length
                - emb_dim: Embedding dimension
                - n_heads: Number of attention heads
                - n_layers: Number of transformer blocks
                - drop_rate: Dropout probability
                - qkv_bias: Whether to use bias in attention projections
        """
        super().__init__()
        self.config = config

        # Token embedding layer
        self.embedding = nn.Embedding(config['vocab_size'], config['emb_dim'])
        
        # Learned positional embeddings, unlike fixed sinusoidal encodings in original transformer
        self.positional_encoding = nn.Embedding(config['context_len'], config['emb_dim'])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config['drop_rate'])
        
        # Stack of transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config['n_layers'])]
        )
        
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(config['emb_dim'])
        
        # Project to vocabulary size for token prediction
        self.linear_projection = nn.Linear(config['emb_dim'], config['vocab_size'], bias=False)
        
    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        """Process input tokens through the GPT-2 model.

        Args:
            x_input (torch.Tensor): Input token ids of shape (batch_size, num_tokens)

        Returns:
            torch.Tensor: Logits over vocabulary of shape (batch_size, num_tokens, vocab_size)
        """
        batch_size, num_tokens = x_input.shape
        
        # Get embeddings for input tokens
        token_embeddings = self.embedding(x_input)
        
        # Get position indices and corresponding embeddings
        positional_embeddings = self.positional_encoding(
            torch.arange(num_tokens, device=x_input.device)
        )
        
        # Combine token and positional embeddings
        x = token_embeddings + positional_embeddings
        x = self.dropout(x)
        
        # Pass through transformer blocks
        x = self.transformer_blocks(x)
        
        # Final layer norm and projection
        x = self.layer_norm(x)
        logits = self.linear_projection(x)
        
        return logits