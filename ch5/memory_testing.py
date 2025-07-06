import torch
from simple_mla import SimpleMLA

def test_simple_mla(batch_size, context_length, d_model, num_heads, kv_latent_dim):
    print("====Memory Usage Testing=====")
    torch.manual_seed(42)
    model = SimpleMLA(d_model=d_model, num_heads=num_heads, kv_latent_dim=kv_latent_dim)
    x = torch.randn(batch_size, context_length, d_model)
    output, kv_cache = model(x)
    print(f"Output shape: {output.shape}")
    print(f"KV Cache shape: {kv_cache.shape}")

    # Assume number of transformer layers (l) is 1
    num_transformer_layers = 1
    head_dim = d_model // num_heads

    # Memory calculation
    # standard kv = l * b * h * d * s * 2 * 2 / 1024 # KB 
    # where l = number of transformer layers, 
    # b = batch size
    # h = number of heads
    # d = d_model
    # s = context length
    # 2 = 2 (K,V)
    # 2 = 2 (bytes per float)
    standard_kv = num_transformer_layers * batch_size * num_heads * head_dim * context_length * 2 * 2 / 1024 # KB 
    # latent kv = l * b * k * s * 2 / 1024 # KB
    # where l = number of transformer layers, 
    # b = batch size
    # dl = kv_latent_dim
    # s = context length
    # 2 = 2 (K,V)
    latent_size = num_transformer_layers * batch_size * kv_latent_dim * context_length * 2 / 1024 # KB
    
    print("====Parameters=====")
    print(f"Number of transformer layers: {num_transformer_layers}")
    print(f"Batch size: {batch_size}")
    print(f"Number of heads: {num_heads}")
    print(f"Head dim: {head_dim}")
    print(f"d_model: {d_model}")
    print(f"Context length: {context_length}")
    print(f"KV latent dim: {kv_latent_dim}")
    print("====Memory=====")
    print(f"Memory standard KV: {standard_kv} KB")
    print(f"Memory latent KV: {latent_size} KB")
    print(f"Memory reduction: {standard_kv / latent_size}")

def test_cache_size_increase(batch_size, context_length, d_model, num_heads, kv_latent_dim):
    print("====Cache Size Increase Testing=====")
    torch.manual_seed(42)

    x = torch.randn(batch_size, context_length, d_model)
    model = SimpleMLA(d_model=d_model, num_heads=num_heads, kv_latent_dim=kv_latent_dim)
    _, kv_cache = model(x)
    print(f"Step 0: Total tokens: {context_length}, cache size: {kv_cache.shape}")

    # Incrementally add tokens to the context length
    for i in range(3):
        x = torch.randn(batch_size, 1, d_model)
        _, kv_cache = model(x, kv_cache=kv_cache, past_length=kv_cache.shape[1])
        print(f"Step {i+1}: Total tokens: {context_length + i + 1}, cache size: {kv_cache.shape}")



    
if __name__ == "__main__":
    batch_size = 1
    context_length = 10
    d_model = 512
    num_heads = 8
    kv_latent_dim = 256
    test_simple_mla(batch_size=batch_size, context_length=context_length, d_model=d_model, num_heads=num_heads, kv_latent_dim=kv_latent_dim)
    test_cache_size_increase(batch_size=batch_size, context_length=context_length, d_model=d_model, num_heads=num_heads, kv_latent_dim=kv_latent_dim)