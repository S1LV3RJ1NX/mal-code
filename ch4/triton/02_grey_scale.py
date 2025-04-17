import matplotlib.pyplot as plt

import torch
import torchvision.transforms.functional as tvf
from torchvision import io

import triton
import triton.language as tl

from utils import cdiv

"""
This file demonstrates how to convert an image to greyscale using Triton.
Triton is a language and compiler for writing highly efficient GPU kernels.
This example shows how to:
1. Load and preprocess an image
2. Create a Triton kernel for RGB to greyscale conversion
3. Apply the kernel to the image
4. Visualize the results
"""


# Load the image using torchvision's io module
img = io.read_image('puppy.jpg')
print(f"img.shape: {img.shape}")

def show_img(x, figsize=(4,3), **kwargs):
    """
    Display an image using matplotlib.
    
    Args:
        x: Image tensor to display
        figsize: Figure size as (width, height) tuple
        **kwargs: Additional arguments to pass to plt.imshow
    """
    plt.figure(figsize=figsize)
    plt.axis('off')
    if len(x.shape)==3: x = x.permute(1,2,0)  # CHW -> HWC (Channel, Height, Width to Height, Width, Channel)
    
    # Extract image_name from kwargs if present
    image_name = kwargs.pop('image_name', None)
    
    plt.imshow(x.cpu(), **kwargs)
    plt.show()
    
    # Save the figure if image_name was provided
    if image_name:
        plt.savefig(image_name)

# Resize the image to a smaller size for faster processing
img = tvf.resize(img, 150, antialias=True)
ch,h,w = img.shape  # Extract channels, height, and width
print(f"ch,h,w: {ch}, {h}, {w}")
print(f"ch*h*w: {ch*h*w}")  # Total number of elements in the image


@triton.jit
def rgb2grey_k(x_ptr, out_ptr, h, w, bs0: tl.constexpr, bs1: tl.constexpr):
    """
    Triton kernel for converting RGB image to greyscale.
    
    This kernel processes the image in 2D blocks, where each block is of size bs0 x bs1.
    
    Args:
        x_ptr: Pointer to the input RGB image data
        out_ptr: Pointer to the output greyscale image data
        h: Height of the image
        w: Width of the image
        bs0: Block size along the height dimension (compile-time constant)
        bs1: Block size along the width dimension (compile-time constant)
    """
    # Get the program IDs for the current block in both dimensions
    pid_0 = tl.program_id(0)  # Block ID in the height dimension
    pid_1 = tl.program_id(1)  # Block ID in the width dimension
    
    # Calculate offsets for this block's elements in both dimensions
    offs_0 = pid_0 * bs0 + tl.arange(0,bs0)  # 1d vector of height offsets
    offs_1 = pid_1 * bs1 + tl.arange(0,bs1)  # 1d vector of width offsets

    # Create a 2D matrix of offsets by combining the height and width offsets
    # Each element (i,j) in this matrix corresponds to a pixel position in the image
    offs = w * offs_0[:,None] + offs_1[None, :]  # 2d matrix! - we multiply first offset by width to get the correct linear index

    # Create masks to handle boundary conditions (when block extends beyond image dimensions)
    mask_0 = offs_0 < h  # 1d vector mask for height dimension
    mask_1 = offs_1 < w  # 1d vector mask for width dimension

    # Combine the masks to create a 2D mask
    mask = mask_0[:,None] & mask_1[None,:]  # 2d matrix! - data musn't go out of bounds along either axis, therefore `logical and` of the individual masks
    
    # Load the R, G, B channels for each pixel in the block
    # The channels are stored in a planar format (all R, then all G, then all B)
    r = tl.load(x_ptr + 0*h*w+offs, mask=mask)  # Load red channel values
    g = tl.load(x_ptr + 1*h*w+offs, mask=mask)  # Load green channel values
    b = tl.load(x_ptr + 2*h*w+offs, mask=mask)  # Load blue channel values

    # Convert RGB to greyscale using the standard luminance formula
    # These coefficients represent the human eye's sensitivity to each color
    out = 0.2989*r + 0.5870*g + 0.1140*b  # Standard RGB to greyscale conversion weights

    # Store the greyscale values to the output tensor
    tl.store(out_ptr + offs, out, mask=mask)


def rgb2grey(x, bs):
    """
    Convert an RGB image to greyscale using the Triton kernel.
    
    Args:
        x: Input RGB image tensor of shape (channels, height, width)
        bs: Tuple of block sizes (bs0, bs1) for the Triton kernel
        
    Returns:
        Greyscale image tensor of shape (height, width)
    """
    c,h,w = x.shape
    # Create an empty tensor for the output greyscale image
    out = torch.empty((h,w), dtype=x.dtype, device=x.device)

    # Define the grid of blocks to cover the entire image
    # The grid function returns a tuple (num_blocks_height, num_blocks_width)
    # cdiv is ceiling division to ensure we have enough blocks to cover the image
    grid = lambda meta: (cdiv(h, meta['bs0']), cdiv(w, meta['bs1']))
    
    # Launch the Triton kernel with the specified grid and block sizes
    rgb2grey_k[grid](x, out, h, w, bs0=bs[0], bs1=bs[1])  # all kwargs are passed into grid function
    return out.view(h,w)  # Reshape to ensure correct dimensions



# Convert the image to greyscale using our Triton kernel
# First move the image to GPU, then convert, then move back to CPU for display
grey_img = rgb2grey(img.to('cuda'), bs=(32, 32)).to('cpu')
# Display the resulting greyscale image
show_img(grey_img, cmap='gray', image_name='grey_scale_image.png')
