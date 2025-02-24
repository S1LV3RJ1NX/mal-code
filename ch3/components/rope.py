# Ref: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/standalone-llama32.ipynb
from components.common import torch


class RoPE:
    """
    RoPE (Rotary Positional Embedding) class for implementing rotary positional embeddings
    in transformer models. This class provides methods to precompute the necessary sine and
    cosine values for the embeddings and to apply these embeddings to input tensors.

    The rotary positional embedding technique allows for a more efficient representation of
    positional information in the context of attention mechanisms, enhancing the model's
    ability to capture relationships between tokens in a sequence.
    """

    @staticmethod
    def precompute_rope(
        head_dim: int, theta_base: float, context_length: int, freq_config: dict = None
    ):
        """
        Precomputes the rotary positional embeddings.

        This method calculates the inverse frequencies and the corresponding sine and cosine
        values for rotary positional embeddings based on the specified head dimension,
        theta base, and context length. It also applies frequency adjustments if a frequency
        configuration is provided.

        Args:
            head_dim (int): The dimensionality of the head (must be even).
            theta_base (float): The base value for frequency scaling.
            context_length (int): The maximum length of the context for which embeddings are computed.
            freq_config (dict, optional): Configuration for frequency adjustments, including:
                - original_context_length: The original context length used for scaling.
                - low_freq_factor: Factor for low frequency adjustments.
                - high_freq_factor: Factor for high frequency adjustments.
                - factor: Scaling factor for frequency adjustments.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the precomputed cosine and sine
            values of shape (context_length, head_dim).
        """
        assert head_dim % 2 == 0, "Embedding dimension must be even"

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            theta_base
            ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim)
        )

        # Frequency adjustments based on the provided configuration
        if freq_config is not None:
            low_freq_wavelen = (
                freq_config["original_context_length"] / freq_config["low_freq_factor"]
            )
            high_freq_wavelen = (
                freq_config["original_context_length"] / freq_config["high_freq_factor"]
            )

            # Calculate wavelengths from inverse frequencies
            wavelen = 2 * torch.pi / inv_freq

            # Adjust inverse frequencies based on the wavelength conditions
            inv_freq_llama = torch.where(
                wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
            )

            # Calculate the smoothing factor for frequency adjustments
            smooth_factor = (
                freq_config["original_context_length"] / wavelen
                - freq_config["low_freq_factor"]
            ) / (freq_config["high_freq_factor"] - freq_config["low_freq_factor"])

            # Compute smoothed inverse frequencies
            smoothed_inv_freq = (1 - smooth_factor) * (
                inv_freq / freq_config["factor"]
            ) + smooth_factor * inv_freq

            # Determine medium frequency conditions
            is_medium_freq = (wavelen <= low_freq_wavelen) & (
                wavelen >= high_freq_wavelen
            )
            inv_freq_llama = torch.where(
                is_medium_freq, smoothed_inv_freq, inv_freq_llama
            )
            inv_freq = inv_freq_llama

        # Generate position indices for the context length
        positions = torch.arange(context_length)

        # Compute the angles for the rotary embeddings
        angles = (
            positions[:, None] * inv_freq[None, :]
        )  # Shape: (context_length, head_dim // 2)

        # Expand angles to match the head_dim
        angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

        # Precompute sine and cosine values for the angles
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        return cos, sin

    @staticmethod
    def apply_rope(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies rotary positional embeddings to the input tensor.

        This method takes an input tensor and applies the precomputed cosine and sine values
        to perform the rotary transformation, enhancing the input representation with positional
        information.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_heads, seq_len, head_dim).
            cos (torch.Tensor): Precomputed cosine values of shape (seq_len, head_dim).
            sin (torch.Tensor): Precomputed sine values of shape (seq_len, head_dim).

        Returns:
            torch.Tensor: Rotated input tensor of the same shape as x, with
            rotary positional embeddings applied.
        """
        # Extract the shape of the input tensor
        batch_size, num_heads, seq_len, head_dim = x.shape

        # Ensure that the head dimension is even
        # RoPE splits the embedding vector into two halves, applies rotations to one half,
        # and combines the results. An odd dimensionality would break this pairing.
        assert head_dim % 2 == 0, "Head dimension must be even"

        # Split the input tensor into two halves for the rotary transformation
        x1 = x[..., : head_dim // 2]  # First half of the embedding
        x2 = x[..., head_dim // 2 :]  # Second half of the embedding

        # Adjust the shapes of the cosine and sine tensors for broadcasting
        # cos and sin have shape (seq_len, head_dim)
        cos = (
            cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
        )  # Shape: (1, 1, seq_len, head_dim)
        sin = (
            sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
        )  # Shape: (1, 1, seq_len, head_dim)

        # Apply the rotary transformation by combining the two halves with sine and cosine
        # Swaps and negates the second half of the embedding vector (x2) to create a rotated counterpart.
        # For example, if the original vector is [x1, x2], the rotated vector becomes [-x2, x1].
        rotated = torch.cat((-x2, x1), dim=-1)
        # Combine the original and rotated tensors
        x_rotated = (x * cos) + (rotated * sin)

        return x_rotated.to(dtype=x.dtype)


if __name__ == "__main__":

    def test_precompute_rope():
        """
        Test the precompute_rope function from the RoPE class.

        This function verifies the correctness of the precomputed cosine and sine values
        used for rotary positional encoding. It checks that the shapes of the returned
        tensors match the expected dimensions and that the values at the first position
        are correct, as defined by the properties of cosine and sine functions.

        Steps:
        1. Define the head dimension, theta base, and context length.
        2. Call the precompute_rope method to obtain cosine and sine values.
        3. Assert that the shapes of the returned tensors are as expected.
        4. Verify that the values at the first position are correct.
        """
        head_dim = 4  # Dimensionality of the head for the rotary encoding
        theta_base = 10000.0  # Base for the angular frequency used in calculations
        context_length = 8  # Length of the context for which RoPE is computed

        # Precompute cosine and sine values for the specified parameters
        cos, sin = RoPE.precompute_rope(head_dim, theta_base, context_length)

        # Check that the shapes of the cosine and sine tensors are correct
        assert cos.shape == (context_length, head_dim), "Cosine shape mismatch"
        assert sin.shape == (context_length, head_dim), "Sine shape mismatch"

        # Check the first position values
        # Cosine at position 0 should be [1, 1, 1, 1] and sine should be [0, 0, 0, 0]
        assert torch.allclose(
            cos[0], torch.tensor([1.0, 1.0, 1.0, 1.0])
        ), "Cosine at position 0 incorrect"
        assert torch.allclose(
            sin[0], torch.tensor([0.0, 0.0, 0.0, 0.0])
        ), "Sine at position 0 incorrect"

    def test_apply_rope():
        """
        Test the apply_rope function from the RoPE class.

        This function generates a random input tensor and applies the rotary positional
        encoding (RoPE) transformation to it. It verifies that the output shape matches
        the input shape and checks the correctness of the rotation behavior by comparing
        the output against an expected result calculated using the defined rotation formula.

        The test ensures that the RoPE implementation behaves as expected, which is
        crucial for maintaining the integrity of the model's positional encoding.

        Steps:
        1. Generate a random input tensor `x` with shape (1, 1, 8, 4).
        2. Precompute the cosine and sine values for the given head dimension and context length.
        3. Apply the RoPE transformation to the input tensor.
        4. Assert that the output shape matches the input shape.
        5. Verify the specific rotation behavior by reconstructing the expected output
           using the rotation formula and comparing it to the actual output.
        """
        # Generate a random input tensor with shape (batch_size, num_heads, seq_len, head_dim)
        x = torch.randn(1, 1, 8, 4)  # Example input tensor

        # Define parameters for the RoPE transformation
        head_dim = 4  # Dimensionality of the head
        theta_base = 10000.0  # Base for the angular frequency
        context_length = 8  # Length of the context for which RoPE is computed

        # Precompute the cosine and sine values for the rotary positional encoding
        cos, sin = RoPE.precompute_rope(head_dim, theta_base, context_length)

        # Apply the RoPE transformation to the input tensor
        x_rotated = RoPE.apply_rope(x, cos, sin)

        # Check that the output shape matches the input shape
        assert x_rotated.shape == x.shape, "Output shape mismatch"

        # Check specific rotation behavior
        # Split the input tensor into two halves for rotation
        x1 = x[..., :2]  # First half of the embedding
        x2 = x[..., 2:]  # Second half of the embedding

        # Create the rotated version of the second half
        rotated = torch.cat((-x2, x1), dim=-1)

        # Apply the rotation formula to compute the expected output
        expected = (x * cos[:8, :].unsqueeze(0).unsqueeze(0)) + (
            rotated * sin[:8, :].unsqueeze(0).unsqueeze(0)
        )

        # Assert that the actual output is close to the expected output
        assert torch.allclose(x_rotated, expected), "Rotation formula incorrect"

    # Run tests to validate the functionality of RoPE methods
    test_precompute_rope()  # Test the precomputation of RoPE values
    test_apply_rope()  # Test the application of RoPE to an input tensor
    print("All tests passed!")  # Confirm that all tests have passed successfully
