"""Tests for ScoreModel / UNetScoreModel.

The diffusion module previously had no direct coverage of the score network;
these tests exercise the forward pass (shape handling, time-argument forms,
the inverse-sigma hook and the NaN guard) and the checkpoint API.
"""
from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from skfda.ml.generative._diffusion_process import (
    VariancePreservingDiffusionProcess,
)
from skfda.ml.generative._score_model import ScoreModel, UNetScoreModel

from ._constants import DATA_DIM, SEED

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


class _NoKwargsModel(ScoreModel):
    """Minimal ScoreModel that never sets self._init_kwargs.

    Used to exercise the to_checkpoint() guard. Has one trainable parameter so
    it is a valid nn.Module, but omits the _init_kwargs contract.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(DATA_DIM, DATA_DIM)

    def forward(
        self, x: Tensor, t: Tensor, y: Tensor | None = None,
    ) -> Tensor:
        """Trivial pass-through forward."""
        return self.linear(x)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def unet() -> UNetScoreModel:
    """Reproducible UNetScoreModel on DATA_DIM points.

    Weights are sampled inside a forked RNG so the fixture does not perturb
    the global torch RNG and is identical across tests.
    """
    with torch.random.fork_rng():
        torch.manual_seed(SEED)
        return UNetScoreModel(n_points=DATA_DIM)


# ─────────────────────────────────────────────────────────────────────────────
# TestForward
# ─────────────────────────────────────────────────────────────────────────────


class TestUNetScoreModelForward:
    """Forward-pass shape and time-argument handling."""

    def test_forward_preserves_2d_shape(self, unet, x_batch, t_batch):
        """A 2-D input (N, M) must yield a 2-D output of the same shape."""
        out = unet(x_batch, t_batch)

        assert out.shape == x_batch.shape

    def test_forward_preserves_3d_shape(self, unet, x_batch, t_batch):
        """A 3-D input (N, 1, M) must yield a 3-D output of the same shape.

        Exercises the channel unsqueeze/squeeze round-trip in forward().
        """
        x_3d = x_batch.unsqueeze(1)

        out = unet(x_3d, t_batch)

        assert out.shape == x_3d.shape

    def test_forward_accepts_scalar_time_tensor(self, unet, x_batch):
        """A 0-d time tensor must be broadcast to the batch size."""
        out = unet(x_batch, torch.tensor(0.5))

        assert out.shape == x_batch.shape

    def test_forward_accepts_python_float_time(self, unet, x_batch):
        """A plain Python float time must be promoted to a tensor."""
        out = unet(x_batch, 0.5)

        assert out.shape == x_batch.shape

    def test_forward_with_odd_n_points(self):
        """An odd n_points must still produce an output matching the input.

        Time embedding pads to an even feature count internally; this checks
        that path on an odd grid.
        """
        with torch.random.fork_rng():
            torch.manual_seed(SEED)
            model = UNetScoreModel(n_points=DATA_DIM + 1)
        x = torch.randn(4, DATA_DIM + 1)

        out = model(x, torch.rand(4))

        assert out.shape == x.shape

    def test_forward_with_labels_raises(self, unet, x_batch, t_batch):
        """Passing labels must raise NotImplementedError with the full text."""
        y = torch.zeros(x_batch.shape[0])

        with pytest.raises(
            NotImplementedError, match="not supported by UNetScoreModel",
        ):
            unet(x_batch, t_batch, y)

    def test_multiply_inv_sigma_is_applied(self, x_batch, t_batch):
        """A multiply_inv_sigma hook must be applied to the network output."""
        with torch.random.fork_rng():
            torch.manual_seed(SEED)
            model = UNetScoreModel(
                n_points=DATA_DIM,
                multiply_inv_sigma=lambda h, t: torch.zeros_like(h),
            )

        out = model(x_batch, t_batch)

        assert torch.equal(out, torch.zeros_like(out))

    def test_nan_in_output_raises(self, x_batch, t_batch):
        """A non-finite network output must be reported, not propagated."""
        with torch.random.fork_rng():
            torch.manual_seed(SEED)
            model = UNetScoreModel(
                n_points=DATA_DIM,
                multiply_inv_sigma=lambda h, t: torch.full_like(
                    h, float("nan"),
                ),
            )

        with pytest.raises(ValueError, match="NaN"):
            model(x_batch, t_batch)


# ─────────────────────────────────────────────────────────────────────────────
# TestCheckpoint
# ─────────────────────────────────────────────────────────────────────────────


class TestScoreModelCheckpoint:
    """to_checkpoint() / from_checkpoint() contract."""

    def test_to_checkpoint_keys_and_class(self, unet):
        """Checkpoint must expose the three keys and the concrete class.

        multiply_inv_sigma is excluded (non-picklable) and the stored device
        is device-agnostic 'cpu'.
        """
        checkpoint = unet.to_checkpoint()

        assert {"class", "init_kwargs", "fit_state"} <= checkpoint.keys()
        assert checkpoint["class"] is UNetScoreModel
        assert checkpoint["init_kwargs"]["multiply_inv_sigma"] is None
        assert checkpoint["init_kwargs"]["device"] == "cpu"

    def test_to_checkpoint_weights_on_cpu(self, unet):
        """All weight tensors in the checkpoint must be on CPU."""
        checkpoint = unet.to_checkpoint()

        for tensor in checkpoint["fit_state"].values():
            assert tensor.device.type == "cpu"

    def test_to_checkpoint_without_init_kwargs_raises(self):
        """A subclass that omits self._init_kwargs must fail loudly."""
        model = _NoKwargsModel()

        with pytest.raises(AttributeError, match="_init_kwargs"):
            model.to_checkpoint()

    def test_from_checkpoint_round_trip(self, unet):
        """from_checkpoint() must restore weights, class and eval mode."""
        restored = ScoreModel.from_checkpoint(unet.to_checkpoint())

        assert isinstance(restored, UNetScoreModel)
        assert not restored.training
        original_state = unet.state_dict()
        for key, tensor in restored.state_dict().items():
            assert torch.equal(tensor, original_state[key])

    def test_from_checkpoint_missing_keys_raises(self):
        """A checkpoint missing required keys must raise ValueError."""
        with pytest.raises(ValueError, match="missing required keys"):
            ScoreModel.from_checkpoint({"class": UNetScoreModel})

    def test_from_checkpoint_wrong_class_raises(self, unet):
        """A 'class' that is not a ScoreModel subclass must raise TypeError."""
        checkpoint = unet.to_checkpoint()
        checkpoint["class"] = int

        with pytest.raises(TypeError, match="subclass of ScoreModel"):
            ScoreModel.from_checkpoint(checkpoint)

    def test_restore_fit_state_incompatible_architecture_raises(self, unet):
        """Restoring weights into a different architecture must raise."""
        checkpoint = unet.to_checkpoint()
        with torch.random.fork_rng():
            torch.manual_seed(SEED)
            mismatched = UNetScoreModel(n_points=2 * DATA_DIM)

        with pytest.raises(ValueError, match="incompatible"):
            mismatched._restore_fit_state(checkpoint["fit_state"])

    def test_on_load_reattaches_multiply_inv_sigma(self, unet, x_batch):
        """on_load() must graft the diffusion process inverse-sigma operator."""
        vp = VariancePreservingDiffusionProcess()
        vp.fit(x_batch)

        unet.on_load(vp)

        assert unet.multiply_inv_sigma == vp.multiply_inv_sigma
