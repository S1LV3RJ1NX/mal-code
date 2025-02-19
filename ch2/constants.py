GPT_CONFIG_124M = dict(
    vocab_size=50257,  # Vocabulary size
    context_len=1024,  # Sequence length or the maximum context length the model can handle
    emb_dim=768,  # Embedding dimension
    n_heads=12,  # Number of attention heads
    n_layers=12,  # Number of layers
    drop_rate=0.1,  # Dropout rate
    qkv_bias=False,  # Query-Key-Value bias
)
