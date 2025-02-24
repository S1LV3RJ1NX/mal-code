import torch
from components.common import nn
from safetensors.torch import load_file


def assign(left: nn.Parameter, right: torch.Tensor, tensor_name: str = "unknown"):
    """
    Assigns the values of a right tensor to a left parameter tensor after checking for shape compatibility.

    Args:
        left (nn.Parameter): The parameter tensor to which values will be assigned.
        right (torch.Tensor): The tensor containing the values to assign.
        tensor_name (str): The name of the tensor for error reporting (default is "unknown").

    Returns:
        nn.Parameter: A new parameter tensor with the assigned values.

    Raises:
        ValueError: If the shapes of the left and right tensors do not match.
    """
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}"
        )

    if isinstance(right, torch.Tensor):
        # Clone and detach the tensor to avoid modifying the original tensor
        return torch.nn.Parameter(right.clone().detach())
    else:
        # Convert the right input to a tensor and return as a parameter
        return torch.nn.Parameter(torch.tensor(right))


def load_pretrained_weights(model: nn.Module, param_config: dict):
    """
    Loads pretrained weights into the model from specified files.

    Args:
        model (nn.Module): The model instance into which weights will be loaded.
        param_config (dict): Configuration dictionary containing model parameters such as number of layers.

    This function loads weights for the embedding layer, transformer blocks, and output layer.
    It also checks for weight tying in the output layer.
    """
    # Initialize a dictionary to hold the loaded parameters
    params = {}
    # Load weights from the specified pretrained files
    for i in range(1, 3):
        current_weights = load_file(
            f"./pretrained-weights/model-0000{i}-of-00002.safetensors"
        )
        params.update(current_weights)

    # Assign the embedding weights
    model.embedding.weight = assign(
        model.embedding.weight,
        params["model.embed_tokens.weight"],
        "model.embed_tokens.weight",
    )

    # Load weights for each transformer block
    for l in range(param_config["n_layers"]):
        # Load attention weights for the query, key, value, and output projection
        model.transformer_blocks[l].att.W_query.weight = assign(
            model.transformer_blocks[l].att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight",
        )
        model.transformer_blocks[l].att.W_key.weight = assign(
            model.transformer_blocks[l].att.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight",
        )
        model.transformer_blocks[l].att.W_value.weight = assign(
            model.transformer_blocks[l].att.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight",
        )
        model.transformer_blocks[l].att.out_proj.weight = assign(
            model.transformer_blocks[l].att.out_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight",
        )
        model.transformer_blocks[l].norm1.weight = assign(
            model.transformer_blocks[l].norm1.weight,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight",
        )

        # Load FeedForward weights for the feedforward network
        model.transformer_blocks[l].ff.fc1.weight = assign(
            model.transformer_blocks[l].ff.fc1.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight",
        )
        model.transformer_blocks[l].ff.fc2.weight = assign(
            model.transformer_blocks[l].ff.fc2.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight",
        )
        model.transformer_blocks[l].ff.fc3.weight = assign(
            model.transformer_blocks[l].ff.fc3.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight",
        )
        model.transformer_blocks[l].norm2.weight = assign(
            model.transformer_blocks[l].norm2.weight,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight",
        )

    # Load the final normalization layer weights
    model.final_norm.weight = assign(
        model.final_norm.weight, params["model.norm.weight"], "model.norm.weight"
    )

    # Load the output layer weights, checking for weight tying
    if "lm_head.weight" in params.keys():
        model.output_head.weight = assign(
            model.output_head.weight, params["lm_head.weight"], "lm_head.weight"
        )
    else:
        model.output_head.weight = assign(
            model.output_head.weight,
            params["model.embed_tokens.weight"],
            "model.embed_tokens.weight",
        )
        print("Model uses weight tying.")
