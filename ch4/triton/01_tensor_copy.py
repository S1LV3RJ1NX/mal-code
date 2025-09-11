import os
os.environ['TRITON_INTERPRET'] = '1' # needs to be set *before* triton is imported to enable simulation mode
import torch
import triton
import triton.language as tl
from utils import *

"""
This file demonstrates a basic Triton kernel that copies data from one tensor to another.
Triton is a language and compiler for writing highly efficient GPU kernels.
"""

# This is the triton kernel:
# The triton.jit decorator takes a python function and turns it into a triton kernel, which is run on the GPU.
# Inside this function only a subset of all python ops are allowed.
# E.g., when NOT simulating, we can't print or use breakpoints, as these don't exist on the GPU. 
@triton.jit
def copy_k(x_ptr, z_ptr, n, bs: tl.constexpr):
    """
    A Triton kernel that copies values from tensor x to tensor z.
    
    Args:
        x_ptr: Pointer to the source tensor data
        z_ptr: Pointer to the destination tensor data
        n: Total number of elements in the tensor
        bs: Block size - number of elements to process per block (marked as compile-time constant)
    """
    # Get the current program ID (equivalent to block index in CUDA)
    pid = tl.program_id(0)
    
    # Create a range of indices for elements in this block (0 to bs-1)
    elements_in_block = tl.arange(0, bs)
    
    # Calculate global offsets for this block's elements
    # Each block processes 'bs' elements starting from (pid * bs)
    offsets = pid * bs + elements_in_block
    
    # Create a mask to handle boundary conditions (when n is not divisible by bs)
    # Only process elements that are within the tensor bounds
    mask = offsets < n
    
    # Load values from source tensor using the calculated offsets
    # The mask ensures we only load valid elements
    x = tl.load(x_ptr + offsets, mask)
    
    # Store the loaded values to the destination tensor at the same offsets
    tl.store(z_ptr + offsets, x, mask)

    # Debug print statement - only works in simulation mode
    # Shows the block ID, offsets, mask, and loaded values
    print_if(f'pid = {pid} | offsets = {offsets}, mask = {mask}, x = {x}', '')


def copy(x: torch.Tensor, bs: int, kernel_fn: triton.jit) -> torch.Tensor:
    """
    Launches a Triton kernel to copy data from tensor x to a new tensor.
    
    Args:
        x: Source tensor to copy from
        bs: Block size - number of elements to process per block
        kernel_fn: The Triton kernel function to use for copying
        
    Returns:
        z: A new tensor containing the copied data
    """
    # Create an output tensor with the same shape and dtype as x
    z = torch.zeros_like(x)
    
    # Ensure tensors are on GPU and have the correct memory layout
    check_tensors_gpu_ready(x, z)
    
    # Get total number of elements in the tensor
    n = x.numel()
    
    # Calculate number of blocks needed (ceiling division)
    n_blocks = cdiv(n, bs)
    
    # Define the grid - how many blocks to launch
    # This is a 1D grid, but could be 2D or 3D for other applications
    grid = (n_blocks,)

    # Launch the kernel with the specified grid and arguments
    # Each block will process 'bs' elements of the tensor
    kernel_fn[grid](x, z, n, bs)

    return z   

if __name__ == '__main__':
    # Use CUDA if available, otherwise fall back to CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Create a sample tensor on the selected device
    x = torch.tensor([1, 2, 3, 4, 5, 6], device=device)
    
    # Copy the tensor using our Triton kernel with block size 2
    z = copy(x, bs=2, kernel_fn=copy_k)
    
    # Print the result to verify the copy worked correctly
    print(z)
