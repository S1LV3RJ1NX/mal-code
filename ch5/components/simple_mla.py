# Implementation of simple Multi-Head Latent Attention

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLA(nn.Module):
    def __init__(self, d_model, num_heads, kv_latent_dim):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model # input embedding dimension
        self.num_heads = num_heads # number of attention heads
        self.kv_latent_dim = kv_latent_dim # latent dimension of key and value
        self.head_dim = d_model // num_heads # dimension of each attention head


        # Projection layers
        # We keep both the input and output embedding dimensions the same

        # Trainable weight matrix corresponding to query (d_model, d_model)
        self.W_q = nn.Linear(d_model, d_model, bias=False)          

        # Compress into latent space for key and value (d_model, kv_latent_dim)
        self.W_dkv = nn.Linear(d_model, kv_latent_dim, bias=False)  

        # Decompress K (key) (kv_latent_dim, d_model)
        self.W_uk = nn.Linear(kv_latent_dim, d_model, bias=False)  

        # Decompress V (value) (kv_latent_dim, d_model)
        self.W_uv = nn.Linear(kv_latent_dim, d_model, bias=False)  

        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)          

        # Layer normalization
        self.ln = nn.LayerNorm(kv_latent_dim)                        

        # Store W_q @ W_uk
        self.register_buffer('absorbed_query_key_projection', None) 


    def forward(self, x: torch.Tensor, kv_cache: torch.Tensor = None, past_length: int = 0) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            kv_cache: Key-value cache tensor of shape (batch_size, num_heads, past_length, head_dim)
            past_length: Length of the past sequence
        """
        batch_size, context_length, d_model = x.shape

        # Compute KV representation for new token (current input token)
        # X * W_dkv = (batch_size, context_length, d_model) * (d_model, kv_latent_dim)
        kv_latent_rep = self.ln(self.W_dkv(x)) # (batch_size, context_length, kv_latent_dim)
        if kv_cache is None:
            kv_cache = kv_latent_rep
        else:
            # Append to the existing kv cache
            # (batch_size, past_length_of_kv_cache + context_length, kv_latent_dim)
            kv_cache = torch.cat([kv_cache, kv_latent_rep], dim=1) 

        #### Compute attention weights ####
        updated_kv_cache_len = kv_cache.size(1)
        # kv cache is of shape (batch_size, updated_kv_cache_len, kv_latent_dim)

        ## 1. Compute the absorbed query matrix only once
        if self.absorbed_query_key_projection is None:
            # multiply W_q by W_uk transpose
            absorbed_query_key_projection = self.W_q.weight @ self.W_uk.weight # (d_model, kv_latent_dim)
            # Split absorbed query matrix among the different attention heads
            # NOTE: The difference between MHA and MLA is that in MHA we split d_model along columns, not the rows unlike here.
            # As a consequence, we would going ahead, we would have to split the input as well, as the last dimension of the absorbed query matrix is the kv_latent_dim, instead of d_model.
            self.absorbed_query_key_projection = absorbed_query_key_projection.view(self.num_heads, self.head_dim, -1)  # (num_heads, head_dim, kv_latent_dim)


        ## 2. Split the input into num_heads by head_dim
        ### (batch_size, context_length, d_model) -> (batch_size, context_length, num_heads, head_dim)
        split_input = x.view(batch_size, context_length, self.num_heads, self.head_dim)
        


        ## 3. Compute the attention scores
        ### We first compute the absorbed query vector for each head by multiplying the split_input by the absorbed query key projection.
        ### We then multiply the absorbed query vector by the updated cache to get the attention scores.

        attention_scores = torch.zeros(batch_size, self.num_heads, context_length, updated_kv_cache_len, device=x.device)
        for head_idx in range(self.num_heads):
            # (batch_size, context_length, head_dim) @ (head_dim, kv_latent_dim) = (batch_size, context_length, kv_latent_dim)
            absorbed_query_vector_for_head = split_input[:, :, head_idx] @ self.absorbed_query_key_projection[head_idx]
            # torch.bmm performs batch matrix multiplication, efficiently computing:
            # This computes attention scores between each query position and all key positions in the cache
            # (batch_size, context_length, kv_latent_dim) @ (batch_size, kv_latent_dim, updated_kv_cache_len) 
            # = (batch_size, context_length, updated_kv_cache_len)
            attention_scores[:, head_idx] = torch.bmm(absorbed_query_vector_for_head, kv_cache.transpose(1, 2))

        ## 4. Scale and apply mask
        attention_scores = attention_scores / (self.head_dim ** 0.5)
        mask = torch.tril(torch.ones(context_length, updated_kv_cache_len, device=x.device), diagonal=past_length)
        # (batch_size, num_heads, context_length, updated_kv_cache_len)
        attention_scores.masked_fill_(mask.view(1, 1, context_length, updated_kv_cache_len) == 0, -torch.inf)

        ## 5. Apply softmax to get attention weights
        # (batch_size, num_heads, context_length, updated_kv_cache_len)
        attention_weights = F.softmax(attention_scores, dim=-1)

        #### Find the context vector ####

        ## 1. We again split the V matrix into heads
        # (batch_size, updated_kv_cache_len, kv_latent_dim) -> (batch_size, updated_kv_cache_len, d_model)
        v = self.W_uv(kv_cache)
        # (batch_size, updated_kv_cache_len, d_model) -> (batch_size, updated_kv_cache_len, num_heads, head_dim) -> (batch_size, num_heads, updated_kv_cache_len, head_dim)
        v_split = v.view(batch_size, updated_kv_cache_len, self.num_heads, self.head_dim).transpose(1, 2)

        ## 2. Find context vector for each head
        context_vectors = []
        for head_idx in range(self.num_heads):
            # (batch_size, context_length, updated_kv_cache_len) @ (batch_size, updated_kv_cache_len, head_dim)
            # = (batch_size, context_length, head_dim)
            context_vector = attention_weights[:, head_idx] @ v_split[:, head_idx]
            context_vectors.append(context_vector)
        
        # Concatenate context vectors along the head dimension
        # (batch_size, context_length, num_heads, head_dim) -> (batch_size, context_length, d_model)
        context_vectors = torch.cat(context_vectors, dim=-1)
        
        ## 3. Project the context vectors back to the original dimension (d_model, d_model)
        output = self.W_o(context_vectors)

        ## 4. Return the output and the updated kv cache
        return output, kv_cache
        
        
        
        