import torch
import triton
import triton.language as tl
from utils import cdiv # Assuming cdiv from previous examples

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides: how many elements to jump to get to the next element
    #          in the same column (stride_row) or same row (stride_col)
    stride_am, stride_ak, # For A (row-major: stride_ak=1)
    stride_bk, stride_bn, # For B (row-major: stride_bn=1)
    stride_cm, stride_cn, # For C (row-major: stride_cn=1)
    # --- Block sizes as compile-time constants ---
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # (Optional: Grouping for L2 cache, Activation function)
    # GROUP_SIZE_M: tl.constexpr,
    # ACTIVATION: tl.constexpr,
):
    """
    Triton Kernel for Blocked Matrix Multiplication C = A @ B.
    Assumes row-major storage layout.
    """
    # ------------ Program ID Calculation -------------
    pid = tl.program_id(axis=0)
    # Simple row-major block assignment (can be optimized)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # ------------ Pointer Setup (Initial) -------------
    # Offsets for the *elements* within the first blocks (k=0)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) # Rows for A & C
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) # Cols for B & C
    offs_k = tl.arange(0, BLOCK_SIZE_K)                      # K-dim for A & B

    # Pointers for the first block of A and B
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # ------------ Accumulation Loop -------------
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks with masking for K dimension
        mask_a = (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        mask_b = (offs_k[:, None] < K - k * BLOCK_SIZE_K)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # Perform matrix multiplication and accumulate
        accumulator += tl.dot(a, b)

        # Advance pointers for next iteration
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # ------------ Store Result -------------
    # Pointers for the output C block
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)

    # Masking for M and N dimensions
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Store the result (cast if necessary, e.g., to float16)
    c = accumulator.to(c_ptr.dtype.element_ty) # Match output tensor type
    tl.store(c_ptrs, c, mask=mask_c)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Python wrapper to launch the Triton MatMul kernel.
    Assumes input tensors are contiguous and on the same GPU device.
    """
    # --- Input Validation ---
    assert a.shape[1] == b.shape[0], "Incompatible matrix dimensions"
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    M, K = a.shape
    K, N = b.shape
    device = a.device
    assert device == b.device, "Tensors must be on the same device"

    # --- Output Allocation ---
    c = torch.empty((M, N), device=device, dtype=torch.float16) # Example: FP16 output

    # --- Kernel Launch Grid ---
    # Define block sizes (these are crucial for performance and depend on hardware)
    # Typical values are powers of 2 (16, 32, 64, 128, 256)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Calculate the grid size
    grid_m = cdiv(M, BLOCK_SIZE_M)
    grid_n = cdiv(N, BLOCK_SIZE_N)
    grid = (grid_m * grid_n,) # Launch as a 1D grid

    # --- Launch Kernel ---
    matmul_kernel[grid](
        a, b, c,                            # Data pointers
        M, N, K,                            # Dimensions
        a.stride(0), a.stride(1),           # Strides for A
        b.stride(0), b.stride(1),           # Strides for B
        c.stride(0), c.stride(1),           # Strides for C
        # Block sizes are passed as keyword args matching constexpr names
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return c

# --- Example Usage ---
# Ensure running on a CUDA device
if torch.cuda.is_available():
    device = torch.device('cuda')
    # Create sample tensors (e.g., FP16)
    a = torch.randn((512, 256), device=device, dtype=torch.float16)
    b = torch.randn((256, 1024), device=device, dtype=torch.float16)

    # Run Triton MatMul
    triton_c = matmul(a, b)

    # Compare with PyTorch's cuBLAS implementation
    torch_c = torch.matmul(a, b)

    # Check results (allow for small floating-point differences)
    print(f"Triton output shape: {triton_c.shape}")
    print(f"PyTorch output shape: {torch_c.shape}")
    if torch.allclose(triton_c, torch_c, atol=1e-2, rtol=0):
         print("✅ Triton and PyTorch results match!")
    else:
         print("❌ Triton and PyTorch results differ!")
         print(f"Max difference: {torch.max(torch.abs(triton_c - torch_c))}")
else:
    print("CUDA device not available. Skipping example.")