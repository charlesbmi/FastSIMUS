"""Backend-specific fused kernels for FastSIMUS.

Custom kernels provide maximum performance by fusing the entire computation
into a single GPU dispatch. Each kernel is a different algorithm from the
Array API path (e.g., on-the-fly geometry instead of precomputed arrays).

Available kernels:
- metal_pfield: Apple Silicon Metal kernel for pfield (requires MLX)
"""
