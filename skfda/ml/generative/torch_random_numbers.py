"""Utilities for creating seeded PyTorch random number generators."""

import torch


def make_torch_rng(
        seed: int | None, device: torch.device | str,
    ) -> torch.Generator:
    """Create a ``torch.Generator`` on ``device``, optionally seeded.

    Args:
        seed: Integer seed for reproducibility. If ``None``, the generator
            is initialized with a random (non-deterministic) state.
        device: The device on which the generator is created.

    Returns:
        A seeded (or randomly initialized) ``torch.Generator``.
    """
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    return generator

