#!/usr/bin/env python3
"""A tiny CuTeDSL matrix multiplication example: C = A @ B.

This is intentionally a teaching example, not a fast GEMM.  Each CUDA thread
computes one C[m, n] element with a serial loop over K.  It demonstrates the
basic CuTeDSL flow used by the larger kernels in this repo:

1. write a @cute.kernel device kernel;
2. wrap its launch in a @cute.jit function;
3. compile once with dummy tensors;
4. call the compiled function with real PyTorch tensors.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from optimus_cutedsl._cutlass_monkeypatch import apply_patches

apply_patches()

import cutlass.cute as cute
from cutlass import Float32, Int32

from optimus_cutedsl._compile_cache import get_or_compile_isolated
from optimus_cutedsl.utils import tvm_ffi_tensor_spec as _to_cute_tensor

THREADS_X = 16
THREADS_Y = 16
_COMPILE_CACHE: Dict[Tuple[int, int, int, int], cute.JitFunction] = {}


@cute.kernel
def _naive_matmul_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    M: Int32,
    N: Int32,
    K: Int32,
):
    """Compute C[M, N] = A[M, K] @ B[K, N], one output per thread."""
    block_x, block_y, _ = cute.arch.block_idx()
    thread_x, thread_y, _ = cute.arch.thread_idx()

    row = block_y * Int32(THREADS_Y) + thread_y
    col = block_x * Int32(THREADS_X) + thread_x

    if row < M and col < N:
        acc = Float32(0.0)
        k = Int32(0)
        while k < K:
            a = Float32(mA[row, k])
            b = Float32(mB[k, col])
            acc += a * b
            k += Int32(1)
        mC[row, col] = acc


@cute.jit
def _launch_naive_matmul_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    M: int,
    N: int,
    K: int,
    stream,
):
    _naive_matmul_kernel(mA, mB, mC, Int32(M), Int32(N), Int32(K)).launch(
        grid=[cute.ceil_div(N, THREADS_X), cute.ceil_div(M, THREADS_Y), 1],
        block=[THREADS_X, THREADS_Y, 1],
        stream=stream,
    )


def _check_inputs(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int, int]:
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("a and b must be CUDA tensors")
    if a.device != b.device:
        raise ValueError(f"a and b must be on the same device: {a.device} vs {b.device}")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("this teaching example expects float32 tensors")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} @ {tuple(b.shape)}")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("a and b must be contiguous")
    M, K = (int(dim) for dim in a.shape)
    _, N = (int(dim) for dim in b.shape)
    if min(M, N, K) <= 0:
        raise ValueError("matrix dimensions must be positive")
    return M, N, K


def _get_compiled_kernel(
    device: torch.device,
    M: int,
    N: int,
    K: int,
) -> cute.JitFunction:
    device_index = int(device.index if device.index is not None else torch.cuda.current_device())
    # Keep the example simple and compile for the exact tensor shapes.  This
    # avoids introducing dynamic-shape layout machinery before the basic
    # kernel is understood.
    key = (device_index, M, N, K)

    def _compile_cache_miss() -> cute.JitFunction:
        dummy_a = torch.empty((M, K), device=device, dtype=torch.float32)
        dummy_b = torch.empty((K, N), device=device, dtype=torch.float32)
        dummy_c = torch.empty((M, N), device=device, dtype=torch.float32)
        return cute.compile(
            _launch_naive_matmul_kernel,
            _to_cute_tensor(dummy_a, leading_dim=1),
            _to_cute_tensor(dummy_b, leading_dim=1),
            _to_cute_tensor(dummy_c, leading_dim=1),
            M,
            N,
            K,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    return get_or_compile_isolated(_COMPILE_CACHE, key, _compile_cache_miss)


@torch.no_grad()
def naive_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return C = A @ B using the teaching CuTeDSL kernel."""
    M, N, K = _check_inputs(a, b)
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    compiled = _get_compiled_kernel(a.device, M, N, K)
    # The fake stream created with `use_tvm_ffi_env_stream=True` is an implicit
    # runtime argument; the compiled function launches on PyTorch's current
    # CUDA stream.
    compiled(a, b, c, M, N, K)
    return c


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(0)
    M, K, N = 32, 64, 48
    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32)
    c = naive_matmul(a, b)
    torch.cuda.synchronize()
    ref = a @ b
    max_abs_err = (c - ref).abs().max().item()
    print(f"shape: {tuple(a.shape)} @ {tuple(b.shape)} -> {tuple(c.shape)}")
    print(f"max_abs_err={max_abs_err:.6g}")
    torch.testing.assert_close(c, ref, atol=1e-4, rtol=1e-4)
    print("PASS")


if __name__ == "__main__":
    main()
