#!/usr/bin/env python3
"""A small CuTeDSL tiled GEMM using shared memory.

This is the next step after ``cutedsl_naive_matmul.py``.  It intentionally
keeps the math simple (scalar FMA, no Tensor Cores), but uses CuTe concepts
that are easy to see in a real CUTLASS kernel:

* ``cute.local_tile`` selects the CTA tile from each global tensor;
* ``cute.make_layout`` describes the shared-memory tile layout;
* ``cute.local_partition`` maps one tile element to each thread;
* ``cute.autovec_copy`` stages A/B tiles through shared memory;
* ``cute.arch.sync_threads`` protects the producer/consumer phases.

The operation is ``C[M, N] = A[M, K] @ B[K, N]`` for contiguous float32
PyTorch tensors.  This is a teaching kernel, not a production GEMM: it uses
one 16x16 CTA tile, one output per thread, and a synchronous shared-memory
pipeline.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from optimus_cutedsl._cutlass_monkeypatch import apply_patches

apply_patches()

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

from optimus_cutedsl._compile_cache import get_or_compile_isolated
from optimus_cutedsl.utils import tvm_ffi_tensor_spec as _to_cute_tensor

TILE_M = 16
TILE_N = 16
TILE_K = 16
THREADS_PER_CTA = TILE_M * TILE_N

_COMPILE_CACHE: Dict[Tuple[int, int, int, int], cute.JitFunction] = {}


@cute.kernel
def _tiled_matmul_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    M: Int32,
    N: Int32,
    K: Int32,
):
    """Compute one 16x16 output tile per CTA using shared-memory staging."""
    tidx = cute.arch.thread_idx()[0]
    # Keep the usual CUDA convention: x indexes N and y indexes M.
    block_n, block_m, _ = cute.arch.block_idx()

    # Global tiles.  The trailing ``None`` keeps the K-tile index as a loop
    # mode, so gA/gB represent all K tiles handled by this CTA.
    # A is row-major [M, K] and B is row-major [K, N].
    gA = cute.local_tile(mA, (TILE_M, TILE_K), (block_m, None))
    gB = cute.local_tile(mB, (TILE_K, TILE_N), (None, block_n))
    gC = cute.local_tile(mC, (TILE_M, TILE_N), (block_m, block_n))

    # CuTe's default compact layout is column-major (mode 0 is the
    # fastest-changing mode), while the PyTorch inputs in this example are
    # row-major.  Spell out the strides so the tile layouts, thread mapping,
    # and shared-memory indexing all agree:
    #
    #   A/C: (row, col) -> row * leading_dim + col
    #   B:   (k, col)   -> k * leading_dim + col
    #
    # This is a small but important CuTe point: the shape alone does not
    # completely describe a tensor.
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(
        mA.element_type,
        cute.make_layout((TILE_M, TILE_K), stride=(TILE_K, 1)),
        byte_alignment=16,
    )
    sB = smem.allocate_tensor(
        mB.element_type,
        cute.make_layout((TILE_K, TILE_N), stride=(TILE_N, 1)),
        byte_alignment=16,
    )

    # A compact thread layout gives each of the 256 threads one logical
    # coordinate in a 16x16 tile.  local_partition changes the layout; the
    # actual global->shared movement is performed by autovec_copy below.
    tile_thread_layout = cute.make_layout(
        (TILE_M, TILE_K),
        stride=(TILE_K, 1),
    )
    tAg = cute.local_partition(gA, tile_thread_layout, tidx)
    tAs = cute.local_partition(sA, tile_thread_layout, tidx)

    tile_thread_layout_b = cute.make_layout(
        (TILE_K, TILE_N),
        stride=(TILE_N, 1),
    )
    tBg = cute.local_partition(gB, tile_thread_layout_b, tidx)
    tBs = cute.local_partition(sB, tile_thread_layout_b, tidx)

    # Thread id -> output coordinate.  This is deliberately explicit so the
    # mapping can be compared directly with the naive CUDA-style example.
    out_m = tidx // Int32(TILE_N)
    out_n = tidx % Int32(TILE_N)
    global_m = block_m * Int32(TILE_M) + out_m
    global_n = block_n * Int32(TILE_N) + out_n

    acc = Float32.zero
    k_tile = Int32(0)
    num_k_tiles = cute.ceil_div(K, TILE_K)

    while k_tile < num_k_tiles:
        k_base = k_tile * Int32(TILE_K)

        # Every thread contributes one A and one B element.  For a partial
        # edge tile, write zero instead of reading outside the input tensor.
        a_k = k_base + (tidx % Int32(TILE_K))
        a_m = block_m * Int32(TILE_M) + (tidx // Int32(TILE_K))
        b_k = k_base + (tidx // Int32(TILE_N))
        b_n = block_n * Int32(TILE_N) + (tidx % Int32(TILE_N))

        if a_m < M and a_k < K:
            cute.autovec_copy(tAg[None, None, k_tile], tAs)
        else:
            sA[tidx // Int32(TILE_K), tidx % Int32(TILE_K)] = Float32(0.0)

        if b_k < K and b_n < N:
            # B is laid out as [K, N], so the K-tile is the trailing
            # "rest" mode produced by local_tile, just like for A.
            cute.autovec_copy(tBg[None, None, k_tile], tBs)
        else:
            sB[tidx // Int32(TILE_N), tidx % Int32(TILE_N)] = Float32(0.0)

        cute.arch.sync_threads()

        # K loop is compile-time unrolled.  All threads reuse the same staged
        # A/B tile from shared memory before moving to the next K tile.
        for k_local in cutlass.range_constexpr(TILE_K):
            acc += Float32(sA[out_m, k_local]) * Float32(sB[k_local, out_n])

        cute.arch.sync_threads()
        k_tile += Int32(1)

    if global_m < M and global_n < N:
        gC[out_m, out_n] = acc


@cute.jit
def _launch_tiled_matmul_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    M: int,
    N: int,
    K: int,
    stream,
):
    _tiled_matmul_kernel(mA, mB, mC, Int32(M), Int32(N), Int32(K)).launch(
        grid=[cute.ceil_div(N, TILE_N), cute.ceil_div(M, TILE_M), 1],
        block=[THREADS_PER_CTA, 1, 1],
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
    # Tile sizes are compile-time constants, so they are part of the kernel
    # configuration even though they are fixed in this small example.
    key = (device_index, M, N, K)

    def _compile_cache_miss() -> cute.JitFunction:
        dummy_a = torch.empty((M, K), device=device, dtype=torch.float32)
        dummy_b = torch.empty((K, N), device=device, dtype=torch.float32)
        dummy_c = torch.empty((M, N), device=device, dtype=torch.float32)
        return cute.compile(
            _launch_tiled_matmul_kernel,
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
def tiled_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return ``a @ b`` using the shared-memory tiled CuTeDSL kernel."""
    M, N, K = _check_inputs(a, b)
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    compiled = _get_compiled_kernel(a.device, M, N, K)
    compiled(a, b, c, M, N, K)
    return c


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(0)
    # Deliberately use non-multiple dimensions to exercise zero padding and
    # predicated output stores.
    M, K, N = 37, 29, 23
    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32)

    actual = tiled_matmul(a, b)
    torch.cuda.synchronize()
    expected = a @ b
    max_abs_err = (actual - expected).abs().max().item()

    print(f"shape: {tuple(a.shape)} @ {tuple(b.shape)} -> {tuple(actual.shape)}")
    print(f"tile: ({TILE_M}, {TILE_N}, {TILE_K}), max_abs_err={max_abs_err:.6g}")
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    print("PASS")


if __name__ == "__main__":
    main()
