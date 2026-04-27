"""NVRTC-based CUDA kernel compilation and launch for NVIDIA GPUs.

Loads libnvrtc and libcuda to compile .cu source at runtime and launch
kernels on JAX-managed GPU arrays without additional dependencies.
"""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

NVRTC_LIB_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..",
    ".venv/lib/python3.14/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12",
)

_cuda: ctypes.CDLL | None = None
_nvrtc: ctypes.CDLL | None = None


def _check(status: int, name: str = "") -> None:
    if status != 0:
        raise RuntimeError(f"CUDA error {status} in {name}")


def _get_cuda() -> ctypes.CDLL:
    global _cuda
    if _cuda is None:
        _cuda = ctypes.CDLL("libcuda.so.1")
        _check(_cuda.cuInit(0), "cuInit")
    return _cuda


def _get_nvrtc() -> ctypes.CDLL:
    global _nvrtc
    if _nvrtc is None:
        paths = [
            NVRTC_LIB_PATH,
            "libnvrtc.so.12",
            "libnvrtc.so",
        ]
        for p in paths:
            try:
                _nvrtc = ctypes.CDLL(os.path.abspath(p) if "/" in p else p)
                return _nvrtc
            except OSError:
                continue

        # Search in nvidia pip packages
        import importlib.util
        spec = importlib.util.find_spec("nvidia.cuda_nvrtc")
        if spec and spec.submodule_search_locations:
            for loc in spec.submodule_search_locations:
                candidate = os.path.join(loc, "lib", "libnvrtc.so.12")
                if os.path.exists(candidate):
                    _nvrtc = ctypes.CDLL(candidate)
                    return _nvrtc

        raise RuntimeError("Cannot find libnvrtc.so.12")
    return _nvrtc


def _get_context() -> ctypes.c_void_p:
    """Get or create a CUDA context on device 0."""
    cuda = _get_cuda()
    ctx = ctypes.c_void_p()
    status = cuda.cuCtxGetCurrent(ctypes.byref(ctx))
    if status == 0 and ctx.value:
        return ctx
    device = ctypes.c_int()
    _check(cuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
    _check(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, device), "cuCtxCreate")
    return ctx


@lru_cache(maxsize=32)
def compile_module(
    source: str,
    defines: tuple[tuple[str, Any], ...] | None = None,
    arch: str = "sm_89",
) -> ctypes.c_void_p:
    """Compile CUDA source via NVRTC and return a module handle.

    Results are cached by (source, defines, arch).
    Defines should be a tuple of (key, value) pairs for hashability.
    """
    nvrtc = _get_nvrtc()
    cuda = _get_cuda()
    _get_context()

    header = ""
    if defines:
        for k, v in defines:
            header += f"#define {k} {v}\n"

    full_source = (header + source).encode()

    prog = ctypes.c_void_p()
    _check(
        nvrtc.nvrtcCreateProgram(
            ctypes.byref(prog),
            full_source,
            b"kernel.cu",
            0, None, None,
        ),
        "nvrtcCreateProgram",
    )

    opts = [
        f"--gpu-architecture={arch}".encode(),
        b"--use_fast_math",
        b"--extra-device-vectorization",
    ]
    c_opts = (ctypes.c_char_p * len(opts))(*opts)
    status = nvrtc.nvrtcCompileProgram(prog, len(opts), c_opts)

    log_size = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
    if log_size.value > 1:
        log_buf = ctypes.create_string_buffer(log_size.value)
        nvrtc.nvrtcGetProgramLog(prog, log_buf)
        log_text = log_buf.value.decode()
        if status != 0:
            raise RuntimeError(f"NVRTC compilation failed:\n{log_text}")

    cubin_size = ctypes.c_size_t()
    nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(cubin_size))
    cubin = ctypes.create_string_buffer(cubin_size.value)
    nvrtc.nvrtcGetCUBIN(prog, cubin)

    nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

    module = ctypes.c_void_p()
    _check(
        cuda.cuModuleLoadData(ctypes.byref(module), cubin),
        "cuModuleLoadData",
    )
    return module


def get_function(module: ctypes.c_void_p, kernel_name: str) -> ctypes.c_void_p:
    """Get a kernel function handle from a compiled CUDA module."""
    cuda = _get_cuda()
    func = ctypes.c_void_p()
    _check(
        cuda.cuModuleGetFunction(ctypes.byref(func), module, kernel_name.encode()),
        "cuModuleGetFunction",
    )
    return func


def compile_kernel(
    source: str,
    kernel_name: str,
    defines: tuple[tuple[str, Any], ...] | None = None,
    arch: str = "sm_89",
) -> tuple[ctypes.c_void_p, ctypes.c_void_p]:
    """Compile CUDA source and return (module, function) for a named kernel.

    The module compilation is cached; only the function lookup varies.
    """
    module = compile_module(source, defines=defines, arch=arch)
    func = get_function(module, kernel_name)
    return module, func


def set_max_dynamic_shared_mem(func: ctypes.c_void_p, nbytes: int) -> None:
    """Set max dynamic shared memory for a kernel (Ampere+, up to 96-100KB)."""
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
    _check(
        _get_cuda().cuFuncSetAttribute(
            func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, nbytes,
        ),
        "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)",
    )


def launch_kernel(
    func: ctypes.c_void_p,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    args: list[Any],
    shared_mem: int = 0,
    stream: ctypes.c_void_p | None = None,
) -> None:
    """Launch a compiled CUDA kernel with the given grid/block dims and args.

    Args should be ctypes pointers (c_void_p for device pointers, c_int, c_float).
    """
    cuda = _get_cuda()
    _get_context()

    if shared_mem > 49152:
        set_max_dynamic_shared_mem(func, shared_mem)

    arg_ptrs = (ctypes.c_void_p * len(args))()
    for i, arg in enumerate(args):
        arg_ptrs[i] = ctypes.cast(ctypes.pointer(arg), ctypes.c_void_p)

    _check(
        cuda.cuLaunchKernel(
            func,
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
            shared_mem,
            stream,
            arg_ptrs,
            None,
        ),
        "cuLaunchKernel",
    )


def synchronize() -> None:
    """Synchronize the current CUDA context."""
    _check(_get_cuda().cuCtxSynchronize(), "cuCtxSynchronize")


def device_alloc(nbytes: int) -> ctypes.c_void_p:
    """Allocate device memory."""
    cuda = _get_cuda()
    _get_context()
    ptr = ctypes.c_void_p()
    _check(cuda.cuMemAlloc_v2(ctypes.byref(ptr), nbytes), "cuMemAlloc")
    return ptr


def device_free(ptr: ctypes.c_void_p) -> None:
    """Free device memory."""
    _get_cuda().cuMemFree_v2(ptr)


def memcpy_htod(dst: ctypes.c_void_p, src: np.ndarray) -> None:
    """Copy numpy array to device."""
    _check(
        _get_cuda().cuMemcpyHtoD_v2(dst, src.ctypes.data, src.nbytes),
        "cuMemcpyHtoD",
    )


def memcpy_dtoh(dst: np.ndarray, src: ctypes.c_void_p) -> None:
    """Copy device memory to numpy array."""
    _check(
        _get_cuda().cuMemcpyDtoH_v2(dst.ctypes.data, src, dst.nbytes),
        "cuMemcpyDtoH",
    )


def jax_array_ptr(arr: Any) -> ctypes.c_void_p:
    """Get the raw device pointer from a JAX array."""
    buf = arr.addressable_data(0)
    ptr = buf.unsafe_buffer_pointer()
    return ctypes.c_void_p(ptr)
