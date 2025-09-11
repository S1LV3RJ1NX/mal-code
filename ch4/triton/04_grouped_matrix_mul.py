import torch
import triton
import triton.language as tl
from utils import cdiv # Assuming cdiv from previous examples

@triton.jit
def grouped_matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # --- Block sizes ---
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # --- Grouping Parameter ---
    GROUP_SZ: tl.constexpr, # New parameter for grouping size
    # Optional: ACTIVATION: tl.constexpr
):
    """
    Triton Kernel for Grouped MatMul (L2 cache optimization).
    """
    # ------------ Program ID Calculation -------------
    pid = tl.program_id(axis=0)
    # Calculate total number of blocks along M and N
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # --- Calculate naive block coordinates ---
    naive_pid_m = pid // num_pid_n
    naive_pid_n = pid % num_pid_n

    # --- Apply Swizzling for Grouped Ordering ---
    # Weirdness: tl.swizzle2d doesn't seem to take group_sz directly in older docs?
    # Let's follow the structure from the reference image provided by user.
    # It seems GROUP_SZ here defines groups along the M dimension.
    # Swizzle the pid_m and pid_n based on the group size.
    # Note: The exact implementation/API might vary slightly between Triton versions.
    # This reorders the blocks processed by each pid.
    pid_m, pid_n = tl.swizzle2d(naive_pid_m, naive_pid_n, num_pid_m, num_pid_n, GROUP_SZ)

    # --- The rest of the kernel is IDENTICAL to matmul_kernel ---

    # ------------ Pointer Setup (Initial) -------------
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # ------------ Accumulation Loop -------------
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask_a = (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        mask_b = (offs_k[:, None] < K - k * BLOCK_SIZE_K)
        # Weirdness: allow_tf32 must be False? Check docs for current best practice.
        # For now, assuming standard load. TF32 is usually enabled via tl.dot.
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # Perform matrix multiplication and accumulate
        # Allow TF32 precision for potential speedup on compatible hardware
        accumulator += tl.dot(a, b, allow_tf32=True)

        # Advance pointers for next iteration
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # ------------ Store Result -------------
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c = accumulator.to(c_ptr.dtype.element_ty)
    tl.store(c_ptrs, c, mask=mask_c)

def grouped_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Python wrapper for the Grouped MatMul kernel.
    """
    assert a.shape[1] == b.shape[0], "Incompatible matrix dimensions"
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    M, K = a.shape
    K, N = b.shape
    device = a.device
    assert device == b.device, "Tensors must be on the same device"

    c = torch.empty((M, N), device=device, dtype=a.dtype) # Match input type usually

    # --- Kernel Launch Grid & Block Sizes ---
    # These should ideally be tuned
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SZ = 8 # Group size for swizzling (powers of 2 are typical)

    grid_m = cdiv(M, BLOCK_SIZE_M)
    grid_n = cdiv(N, BLOCK_SIZE_N)
    grid = (grid_m * grid_n,) # 1D grid

    # --- Launch Kernel ---
    grouped_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SZ=GROUP_SZ, # Pass the group size
    )
    return c

# --- Example Usage ---
if torch.cuda.is_available():
    device = torch.device('cuda')
    # Use larger matrices where L2 cache effects are more prominent
    a = torch.randn((2048, 1024), device=device, dtype=torch.float16)
    b = torch.randn((1024, 4096), device=device, dtype=torch.float16)

    # Run Triton Grouped MatMul
    triton_grouped_c = grouped_matmul(a, b)

    # Compare with PyTorch
    torch_c = torch.matmul(a, b)

    print(f"Triton Grouped output shape: {triton_grouped_c.shape}")
    print(f"PyTorch output shape: {torch_c.shape}")
    if torch.allclose(triton_grouped_c, torch_c, atol=1e-1, rtol=0.01): # Higher tolerance for large FP16 calcs
         print("✅ Triton Grouped and PyTorch results match!")
    else:
         print("❌ Triton Grouped and PyTorch results differ significantly!")
         print(f"Max difference: {torch.max(torch.abs(triton_grouped_c - torch_c))}")

    # Optional: Compare performance (requires more careful benchmarking)
    # import timeit
    # print("PyTorch:", timeit.timeit(lambda: torch.matmul(a, b), number=10))
    # print("Triton Grouped:", timeit.timeit(lambda: grouped_matmul(a, b), number=10))

else:
    print("CUDA device not available. Skipping example.")