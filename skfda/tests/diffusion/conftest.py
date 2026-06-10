"""Shared pytest fixtures for diffusion process tests.

Constants below duplicate _constants.py: conftest.py is loaded before the
package import system initialises, so relative imports do not work here.
"""
from __future__ import annotations

import pytest
import torch

# Canonical source: _constants.py — keep in sync.
_BATCH_SIZE = 8
_DATA_DIM   = 16
_SEED       = 13


@pytest.fixture(scope="module")
def generator() -> torch.Generator:
    """PyTorch generator seeded with _SEED, module-scoped."""
    g = torch.Generator()
    g.manual_seed(_SEED)
    return g


@pytest.fixture(scope="module")
def x_batch(generator) -> torch.Tensor:
    """Batch of reference functional data, shape (N, M), values in N(0, 1)."""
    return torch.randn(_BATCH_SIZE, _DATA_DIM, generator=generator)


@pytest.fixture(scope="module")
def x_batch_scaled(x_batch) -> torch.Tensor:
    """x_batch * 2.0.

    Used in linearity tests: mean_cond(2*x, t) == 2*mean_cond(x, t).
    """
    return x_batch * 2.0


@pytest.fixture(scope="module")
def t_batch() -> torch.Tensor:
    """Interior times in (0, 1), shape (N,).

    Endpoints excluded; use t_zero/t_one for boundary tests.
    """
    return torch.linspace(0.1, 0.8, _BATCH_SIZE)


@pytest.fixture(scope="module")
def t_zero() -> torch.Tensor:
    """Times at t=0, shape (N,). Initial conditions: mean_cond(x, 0) ≈ x, _cov(0) ≈ 0."""
    return torch.zeros(_BATCH_SIZE)


@pytest.fixture(scope="module")
def t_one() -> torch.Tensor:
    """Times at t=1, shape (N,). Final conditions: mean_cond(x, 1) ≈ 0, _cov(1) ≈ 1 for VP."""
    return torch.ones(_BATCH_SIZE)


@pytest.fixture(scope="module")
def t_sweep() -> torch.Tensor:
    """Dense sweep of (0, 1), shape (50,).

    Used for whole-domain property checks (monotonicity, mu^2+sigma^2=1).
    """
    return torch.linspace(0.01, 0.99, 50)
