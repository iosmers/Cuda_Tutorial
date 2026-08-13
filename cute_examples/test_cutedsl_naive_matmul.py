"""One small correctness case for :mod:`cutedsl_naive_matmul`."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_naive_matmul_small_case() -> None:
    example_path = Path(__file__).with_name("cutedsl_naive_matmul.py")
    spec = importlib.util.spec_from_file_location("cutedsl_naive_matmul", example_path)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    torch.manual_seed(0)
    a = torch.randn((3, 5), device="cuda", dtype=torch.float32)
    b = torch.randn((5, 4), device="cuda", dtype=torch.float32)

    actual = module.naive_matmul(a, b)
    torch.cuda.synchronize()
    expected = a @ b

    assert actual.shape == (3, 4)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
