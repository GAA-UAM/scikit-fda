"""Correctness tests for FunctionalDiffusionGenerator."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from torch.utils.data import TensorDataset

from skfda.ml.generative._diffusion_model import (
    FunctionalDiffusionGenerator,
    _fdatagrid_to_tensor_dataset,
)
from skfda.ml.generative._diffusion_process import (
    DiagonalDiffusionProcess,
    VarianceExplodingDiffusionProcess,
    VariancePreservingDiffusionProcess,
)
from skfda.ml.generative._reverse_diffusion import ReverseDiffusionProcess
from skfda.representation.grid import FDataGrid

from ._constants import SEED

# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ─────────────────────────────────────────────────────────────────────────────

_N_SAMPLES = 20
_N_GRID_POINTS = 32

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def small_grid() -> FDataGrid:
    """20 identical sin(2πt) samples on 32 grid points.

    Expected values are computed from data_matrix rather than hardcoded so
    the oracle stays independent of grid resolution.
    """
    t = np.linspace(0, 1, _N_GRID_POINTS)
    values = np.sin(2 * np.pi * t)                              # shape (M,)
    data_matrix = np.tile(values, (_N_SAMPLES, 1)).reshape(     # (N, M, 1)
        _N_SAMPLES, _N_GRID_POINTS, 1,
    )
    return FDataGrid(data_matrix=data_matrix, grid_points=[t])


@pytest.fixture(scope="module")
def constant_grid() -> FDataGrid:
    """Every sample is f(t) = 3.0; exercises the _scale_=1.0 clamping guard."""
    t = np.linspace(0, 1, _N_GRID_POINTS)
    data_matrix = np.full((_N_SAMPLES, _N_GRID_POINTS, 1), 3.0)
    return FDataGrid(data_matrix=data_matrix, grid_points=[t])


@pytest.fixture(scope="module")
def labels_array() -> np.ndarray:
    """Alternating labels [0, 1, …] of length _N_SAMPLES, dtype float32.

    float32 matches the .float() cast inside _fdatagrid_to_tensor_dataset.
    """
    return np.array([i % 2 for i in range(_N_SAMPLES)], dtype=np.float32)


@pytest.fixture
def fitted_generator(small_grid: FDataGrid) -> FunctionalDiffusionGenerator:
    """FunctionalDiffusionGenerator fitted on small_grid with max_iter=1.

    Function-scoped so each test gets an independent instance; shared mutable
    state (e.g. RNG advancement) would cause test-order-dependent failures.
    """
    gen = FunctionalDiffusionGenerator(max_iter=1, seed=SEED)
    gen.fit(small_grid)
    return gen


@pytest.fixture
def tmp_checkpoint(tmp_path: Path) -> Path:
    """Path to a temporary checkpoint file, fresh per test."""
    return tmp_path / "model.pt"


# ─────────────────────────────────────────────────────────────────────────────
# Module-level test helpers
# ─────────────────────────────────────────────────────────────────────────────


class _EmptyModule(torch.nn.Module):
    """Score model stub with no trainable parameters.

    Triggers the 'at least one trainable parameter' guard in fit().
    forward() is never called; fit() raises before the training loop.
    """

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # noqa: ARG002
        y: torch.Tensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return x


class _StubReverseProcess(ReverseDiffusionProcess):
    """Reverse diffusion stub that always returns torch.zeros_like(x_t).

    Attributes:
        call_count: number of times reverse() has been called.
        grad_enabled_during_call: torch.is_grad_enabled() value at each call.
    """

    def __init__(self) -> None:
        # Skip super().__init__(): ReverseDiffusionProcess requires an
        # integrator argument that is irrelevant for a stub.
        self.call_count: int = 0
        self.grad_enabled_during_call: list[bool] = []

    def reverse(
        self,
        diff_process: object,  # noqa: ARG002
        *,
        score_model: object,  # noqa: ARG002
        x_t: torch.Tensor,
        t_1: torch.Tensor,  # noqa: ARG002
        y: torch.Tensor | None = None,  # noqa: ARG002
        **kwargs: object,  # noqa: ARG002
    ) -> torch.Tensor:
        self.call_count += 1
        self.grad_enabled_during_call.append(torch.is_grad_enabled())
        return torch.zeros_like(x_t)


class _SpyingVP(VariancePreservingDiffusionProcess):
    """VP subclass that records the sample counts passed to fit().

    fit() operates on a clone of the user's process, so per-instance state
    would not survive cloning; counts are accumulated at the class level
    instead. The constructor signature is inherited unchanged so the class
    stays clonable by scikit-learn. Reset ``fit_call_sample_counts`` before
    each use.

    Attribute:
        fit_call_sample_counts: list[int] — per-call sample counts in order.
    """

    fit_call_sample_counts: ClassVar[list[int]] = []

    def fit(self, x: torch.Tensor) -> _SpyingVP:  # type: ignore[override]
        _SpyingVP.fit_call_sample_counts.append(x.shape[0])
        return super().fit(x)


# ─────────────────────────────────────────────────────────────────────────────
# TestInitialization
# ─────────────────────────────────────────────────────────────────────────────


class TestInitialization:
    """Tests for FunctionalDiffusionGenerator.__init__."""

    # ─────────────────────────────────────────────────────────────────────────
    # Default and custom diff_process
    # ─────────────────────────────────────────────────────────────────────────

    def test_default_diff_process_is_none_until_fit(self) -> None:
        """diff_process=None is stored verbatim and resolved to VP at fit().

        Per the scikit-learn convention the constructor must not transform
        its arguments; the default is materialised as the fitted
        ``diff_process_`` attribute instead.
        """
        gen = FunctionalDiffusionGenerator()

        assert gen.diff_process is None
        assert not hasattr(gen, "diff_process_")

    def test_fit_resolves_default_diff_process_to_vp(
        self, small_grid: FDataGrid,
    ) -> None:
        """After fit(), diff_process_ must be a VariancePreservingDiffusion."""
        gen = FunctionalDiffusionGenerator(max_iter=1, seed=SEED)

        gen.fit(small_grid)

        assert isinstance(
            gen.diff_process_, VariancePreservingDiffusionProcess,
        )

    def test_custom_diff_process_is_stored_by_identity(self) -> None:
        """A provided diff_process must be stored as the exact same object."""
        ve = VarianceExplodingDiffusionProcess(
            g_schedule="exponential",
            g_0=0.1,
            g_T=15.0,
        )
        gen = FunctionalDiffusionGenerator(diff_process=ve)

        assert gen.diff_process is ve

    # ─────────────────────────────────────────────────────────────────────────
    # normalize / standardize mutual exclusion
    # ─────────────────────────────────────────────────────────────────────────

    def test_normalize_and_standardize_both_true_raises(
        self, small_grid: FDataGrid,
    ) -> None:
        """normalize=True and standardize=True must raise ValueError at fit.

        Validation happens in fit (not __init__) to keep the constructor free
        of logic, per the scikit-learn convention.
        """
        gen = FunctionalDiffusionGenerator(normalize=True, standardize=True)

        with pytest.raises(ValueError):  # noqa: PT011
            gen.fit(small_grid)

    def test_normalize_false_standardize_false_accepted(self) -> None:
        """Both flags False (raw-data mode) must be accepted without error."""
        gen = FunctionalDiffusionGenerator(normalize=False, standardize=False)

        assert gen.normalize is False
        assert gen.standardize is False

    # ─────────────────────────────────────────────────────────────────────────
    # Verbatim storage of all constructor args
    # ─────────────────────────────────────────────────────────────────────────

    def test_constructor_args_stored_verbatim(self) -> None:
        """Every constructor argument must be stored as the exact value passed.

        Non-default values are used for every parameter to rule out
        coincidental matches with defaults. diff_process and score_model are
        excluded; they are covered by the dedicated identity and None-storage
        tests.
        """
        gen = FunctionalDiffusionGenerator(
            normalize=False,
            standardize=True,
            max_iter=5,
            batch_size=8,
            n_jobs=2,
            device="cpu",
            seed=SEED,
        )

        assert gen.normalize is False
        assert gen.standardize is True
        assert gen.max_iter == 5  # noqa: PLR2004
        assert gen.batch_size == 8  # noqa: PLR2004
        assert gen.n_jobs == 2  # noqa: PLR2004
        assert gen.device == "cpu"
        assert gen.seed == SEED

    # ─────────────────────────────────────────────────────────────────────────
    # score_model parameter
    # ─────────────────────────────────────────────────────────────────────────

    def test_score_model_none_stored_as_none(self) -> None:
        """score_model=None must be stored as None; built lazily in fit()."""
        gen = FunctionalDiffusionGenerator(score_model=None)

        assert gen.score_model is None


# ─────────────────────────────────────────────────────────────────────────────
# TestPreprocess
# ─────────────────────────────────────────────────────────────────────────────


class TestPreprocess:
    """Tests for FunctionalDiffusionGenerator._preprocess."""

    # ─────────────────────────────────────────────────────────────────────────
    # normalize path — _bias_ and _scale_
    # ─────────────────────────────────────────────────────────────────────────

    def test_normalize_bias_is_midpoint(
        self, small_grid: FDataGrid,
    ) -> None:
        """_bias_ must equal (max + min) / 2."""
        gen = FunctionalDiffusionGenerator(normalize=True)
        expected_bias = (
            small_grid.data_matrix.max() + small_grid.data_matrix.min()
        ) / 2

        gen._preprocess(small_grid)  # noqa: SLF001

        assert gen._bias_ == pytest.approx(expected_bias)

    def test_normalize_scale_is_half_range(
        self, small_grid: FDataGrid,
    ) -> None:
        """_scale_ must equal (max - min) / 2."""
        gen = FunctionalDiffusionGenerator(normalize=True)
        expected_scale = (
            small_grid.data_matrix.max() - small_grid.data_matrix.min()
        ) / 2

        gen._preprocess(small_grid)  # noqa: SLF001

        assert gen._scale_ == pytest.approx(expected_scale)

    # ─────────────────────────────────────────────────────────────────────────
    # standardize path — _bias_ and _scale_
    # ─────────────────────────────────────────────────────────────────────────

    def test_standardize_bias_is_mean(
        self, small_grid: FDataGrid,
    ) -> None:
        """_bias_ must equal the global scalar mean."""
        gen = FunctionalDiffusionGenerator(normalize=False, standardize=True)
        expected_bias = float(small_grid.data_matrix.mean())

        gen._preprocess(small_grid)  # noqa: SLF001

        assert gen._bias_ == pytest.approx(expected_bias)

    def test_standardize_scale_is_std(
        self, small_grid: FDataGrid,
    ) -> None:
        """_scale_ must equal the global std (ddof=0)."""
        gen = FunctionalDiffusionGenerator(normalize=False, standardize=True)
        expected_scale = float(small_grid.data_matrix.std())

        gen._preprocess(small_grid)  # noqa: SLF001

        assert gen._scale_ == pytest.approx(expected_scale)

    # ─────────────────────────────────────────────────────────────────────────
    # no-transform path — _bias_ and _scale_
    # ─────────────────────────────────────────────────────────────────────────

    def test_no_transform_bias_zero_scale_one(
        self, small_grid: FDataGrid,
    ) -> None:
        """With both flags False, _bias_=0.0 and _scale_=1.0 (identity)."""
        gen = FunctionalDiffusionGenerator(normalize=False, standardize=False)

        gen._preprocess(small_grid)  # noqa: SLF001

        assert gen._bias_ == 0.0
        assert gen._scale_ == 1.0

    # ─────────────────────────────────────────────────────────────────────────
    # Scale clamping guard
    # ─────────────────────────────────────────────────────────────────────────

    def test_constant_data_scale_clamped_to_one(
        self, constant_grid: FDataGrid,
    ) -> None:
        """When (max - min) / 2 < eps, _scale_ must be clamped to 1.0."""
        gen = FunctionalDiffusionGenerator(normalize=True)

        gen._preprocess(constant_grid)  # noqa: SLF001

        assert gen._scale_ == 1.0

    # ─────────────────────────────────────────────────────────────────────────
    # Return value — type and length
    # ─────────────────────────────────────────────────────────────────────────

    def test_returns_tensor_dataset(
        self, small_grid: FDataGrid,
    ) -> None:
        """_preprocess must return a TensorDataset of length n_samples."""
        gen = FunctionalDiffusionGenerator(normalize=False, standardize=False)

        dataset = gen._preprocess(small_grid)  # noqa: SLF001

        assert isinstance(dataset, TensorDataset)
        assert len(dataset) == small_grid.n_samples

    # ─────────────────────────────────────────────────────────────────────────
    # Data tensor — dtype and shape
    # ─────────────────────────────────────────────────────────────────────────

    def test_data_tensor_is_float32(
        self, small_grid: FDataGrid,
    ) -> None:
        """The data tensor must be float32."""
        gen = FunctionalDiffusionGenerator(normalize=False, standardize=False)

        dataset = gen._preprocess(small_grid)  # noqa: SLF001

        assert dataset.tensors[0].dtype == torch.float32

    def test_data_tensor_shape_is_n_samples_by_n_points(
        self, small_grid: FDataGrid,
    ) -> None:
        """Data tensor must have shape (N, M), not (N, M, 1)."""
        gen = FunctionalDiffusionGenerator(normalize=False, standardize=False)
        expected_shape = (small_grid.n_samples, len(small_grid.grid_points[0]))

        dataset = gen._preprocess(small_grid)  # noqa: SLF001

        assert dataset.tensors[0].shape == expected_shape

    # ─────────────────────────────────────────────────────────────────────────
    # Labeled data — two-tensor dataset
    # ─────────────────────────────────────────────────────────────────────────

    def test_with_labels_dataset_has_two_tensors(
        self, small_grid: FDataGrid, labels_array: np.ndarray,
    ) -> None:
        """Passing y must produce a TensorDataset with exactly two tensors."""
        gen = FunctionalDiffusionGenerator(normalize=False, standardize=False)

        dataset = gen._preprocess(small_grid, y=labels_array)  # noqa: SLF001

        assert len(dataset.tensors) == 2  # noqa: PLR2004
        assert dataset.tensors[1].shape == (small_grid.n_samples,)

    # ─────────────────────────────────────────────────────────────────────────
    # Mutation guard
    # ─────────────────────────────────────────────────────────────────────────

    def test_preprocess_does_not_mutate_original_fdatagrid(
        self, small_grid: FDataGrid,
    ) -> None:
        """_preprocess must not modify the input FDataGrid in place."""
        data_matrix_before = small_grid.data_matrix.copy()
        gen = FunctionalDiffusionGenerator(normalize=True)

        gen._preprocess(small_grid)  # noqa: SLF001

        assert np.array_equal(small_grid.data_matrix, data_matrix_before)


# ─────────────────────────────────────────────────────────────────────────────
# TestFit
# ─────────────────────────────────────────────────────────────────────────────


class TestFit:
    """Tests for FunctionalDiffusionGenerator.fit."""

    # ─────────────────────────────────────────────────────────────────────────
    # scikit-learn convention
    # ─────────────────────────────────────────────────────────────────────────

    def test_fit_returns_self(
        self, small_grid: FDataGrid,
    ) -> None:
        """fit() must return self (scikit-learn chaining convention)."""
        gen = FunctionalDiffusionGenerator(max_iter=1, seed=SEED)

        result = gen.fit(small_grid)

        assert result is gen

    # ─────────────────────────────────────────────────────────────────────────
    # Input validation
    # ─────────────────────────────────────────────────────────────────────────

    def test_fit_raises_for_multidimensional_domain(self) -> None:
        """fit() must raise ValueError for dim_domain > 1."""
        t = np.linspace(0, 1, 4)
        data_matrix = np.zeros((2, 4, 4, 1))  # (N, M1, M2, codomain)
        grid_2d = FDataGrid(data_matrix=data_matrix, grid_points=[t, t])
        gen = FunctionalDiffusionGenerator(max_iter=1, seed=SEED)

        with pytest.raises(ValueError):  # noqa: PT011
            gen.fit(grid_2d)

    def test_fit_raises_for_multidimensional_codomain(self) -> None:
        """fit() must raise ValueError for dim_codomain > 1."""
        t = np.linspace(0, 1, 8)
        data_matrix = np.zeros((2, 8, 2))  # (N, M, codomain=2)
        grid_vector = FDataGrid(data_matrix=data_matrix, grid_points=[t])
        gen = FunctionalDiffusionGenerator(max_iter=1, seed=SEED)

        with pytest.raises(ValueError):  # noqa: PT011
            gen.fit(grid_vector)

    # ─────────────────────────────────────────────────────────────────────────
    # Post-fit attributes
    # ─────────────────────────────────────────────────────────────────────────

    def test_fit_sets_grid_points_attribute(
        self, small_grid: FDataGrid,
    ) -> None:
        """fit() must store the training grid as grid_points_."""
        gen = FunctionalDiffusionGenerator(max_iter=1, seed=SEED)

        gen.fit(small_grid)

        assert hasattr(gen, "grid_points_")
        assert np.array_equal(gen.grid_points_, small_grid.grid_points[0])

    def test_fit_sets_score_model_attribute(
        self, small_grid: FDataGrid,
    ) -> None:
        """After fit(), score_model_ must hold a torch.nn.Module."""
        gen = FunctionalDiffusionGenerator(max_iter=1, seed=SEED)

        gen.fit(small_grid)

        assert hasattr(gen, "score_model_")
        assert isinstance(gen.score_model_, torch.nn.Module)

    def test_score_model_in_eval_mode_after_fit(
        self, small_grid: FDataGrid,
    ) -> None:
        """score_model_ must be in eval mode when fit() returns.

        generate() does not call .eval() itself, so this must be set by fit().
        """
        gen = FunctionalDiffusionGenerator(max_iter=1, seed=SEED)

        gen.fit(small_grid)

        assert gen.score_model_.training is False

    # ─────────────────────────────────────────────────────────────────────────
    # score_model validation
    # ─────────────────────────────────────────────────────────────────────────

    def test_fit_raises_if_custom_score_model_not_nn_module(
        self, small_grid: FDataGrid,
    ) -> None:
        """fit() must raise TypeError when score_model is not nn.Module."""
        gen = FunctionalDiffusionGenerator(score_model="not_a_module")

        with pytest.raises(TypeError):
            gen.fit(small_grid)

    def test_fit_raises_if_score_model_has_no_parameters(
        self, small_grid: FDataGrid,
    ) -> None:
        """fit() must raise ValueError when score model has no parameters."""
        gen = FunctionalDiffusionGenerator(score_model=_EmptyModule())

        with pytest.raises(ValueError, match="trainable parameter"):
            gen.fit(small_grid)

    # ─────────────────────────────────────────────────────────────────────────
    # Labeled data
    # ─────────────────────────────────────────────────────────────────────────

    def test_fit_with_labels_does_raise_not_implemented(
        self, small_grid: FDataGrid, labels_array: np.ndarray,
    ) -> None:
        """fit() must raise not implemented when y labels are supplied."""
        gen = FunctionalDiffusionGenerator(max_iter=1, seed=SEED)

        with pytest.raises(
            NotImplementedError, match="Conditional generation",
        ):
            gen.fit(small_grid, y=labels_array)

    # ─────────────────────────────────────────────────────────────────────────
    # diff_process.fit receives all samples
    # ─────────────────────────────────────────────────────────────────────────

    def test_fit_diff_process_receives_full_dataset(
        self, small_grid: FDataGrid,
    ) -> None:
        """The fitted diff_process clone must receive all N training samples.

        fit() clones the user's process, so the spy records into a shared
        class-level list rather than on the original instance.
        """
        _SpyingVP.fit_call_sample_counts.clear()
        spy = _SpyingVP(beta_schedule="linear", beta_min=0.1, beta_max=10.0)
        gen = FunctionalDiffusionGenerator(
            diff_process=spy,
            max_iter=1,
            batch_size=4,
            seed=SEED,
        )

        gen.fit(small_grid)

        total_samples_seen = sum(_SpyingVP.fit_call_sample_counts)
        assert total_samples_seen == _N_SAMPLES  # 20; bug yields 4

    # ─────────────────────────────────────────────────────────────────────────
    # Reproducibility
    # ─────────────────────────────────────────────────────────────────────────

    def test_fit_is_reproducible_with_same_seed(
        self, small_grid: FDataGrid,
    ) -> None:
        """Same seed must produce identical weights after fit()."""
        gen_a = FunctionalDiffusionGenerator(max_iter=1, seed=SEED)
        gen_b = FunctionalDiffusionGenerator(max_iter=1, seed=SEED)
        gen_a.fit(small_grid)
        gen_b.fit(small_grid)

        for param_a, param_b in zip(
            gen_a.score_model_.parameters(),
            gen_b.score_model_.parameters(),
            strict=True,
        ):
            assert torch.equal(param_a, param_b)

    def test_fit_does_not_mutate_user_diff_process(
        self, small_grid: FDataGrid,
    ) -> None:
        """fit() must clone diff_process and leave the user's instance unfit.

        The fitted state lives on the clone exposed as diff_process_; the
        object passed by the user must not gain an 'M_' attribute.
        """
        vp = VariancePreservingDiffusionProcess(seed=0)
        gen = FunctionalDiffusionGenerator(
            diff_process=vp, max_iter=1, seed=SEED,
        )

        gen.fit(small_grid)

        assert not hasattr(vp, "M_")
        assert gen.diff_process_ is not vp
        assert hasattr(gen.diff_process_, "M_")

    def test_refit_on_different_grid_size_updates_grid(self) -> None:
        """Refitting on a different grid size must update grid_points_.

        A second fit must fully replace the fitted state rather than retain
        the first grid's dimension.
        """
        gen = FunctionalDiffusionGenerator(max_iter=1, seed=SEED)

        t32 = np.linspace(0, 1, 32)
        grid32 = FDataGrid(
            data_matrix=np.tile(np.sin(2 * np.pi * t32), (10, 1)),
            grid_points=[t32],
        )
        t24 = np.linspace(0, 1, 24)
        grid24 = FDataGrid(
            data_matrix=np.tile(np.sin(2 * np.pi * t24), (10, 1)),
            grid_points=[t24],
        )

        gen.fit(grid32)
        gen.fit(grid24)
        result = gen.generate(n_samples=2)

        assert len(gen.grid_points_) == 24  # noqa: PLR2004
        assert result.data_matrix.shape == (2, 24, 1)

# ─────────────────────────────────────────────────────────────────────────────
# TestLossFunction
# ─────────────────────────────────────────────────────────────────────────────

_LOSS_BATCH = 4  # small batch — cheap while still exercising batch reductions


class TestLossFunction:
    """Tests for FunctionalDiffusionGenerator._loss_function."""

    # ─────────────────────────────────────────────────────────────────────────
    # Output value
    # ─────────────────────────────────────────────────────────────────────────

    def test_loss_is_non_negative(
        self, fitted_generator: FunctionalDiffusionGenerator,
    ) -> None:
        """Loss must be non-negative (mean of squared norms)."""
        gen = fitted_generator
        x = torch.randn(_LOSS_BATCH, _N_GRID_POINTS)
        t = torch.full((_LOSS_BATCH,), 0.5)
        generator = torch.Generator()
        generator.manual_seed(SEED)

        loss = gen._loss_function(  # noqa: SLF001
            gen.score_model_, x, t, None, generator,
        )

        assert loss.item() >= 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Output shape
    # ─────────────────────────────────────────────────────────────────────────

    def test_loss_returns_scalar_tensor(
        self, fitted_generator: FunctionalDiffusionGenerator,
    ) -> None:
        """Loss must be a 0-dimensional tensor."""
        gen = fitted_generator
        x = torch.randn(_LOSS_BATCH, _N_GRID_POINTS)
        t = torch.full((_LOSS_BATCH,), 0.5)
        generator = torch.Generator()
        generator.manual_seed(SEED)

        loss = gen._loss_function(  # noqa: SLF001
            gen.score_model_, x, t, None, generator,
        )

        assert loss.shape == torch.Size([])

    # ─────────────────────────────────────────────────────────────────────────
    # Time-dependence
    # ─────────────────────────────────────────────────────────────────────────

    def test_loss_changes_with_different_t(
        self, fitted_generator: FunctionalDiffusionGenerator,
    ) -> None:
        """Loss must differ for different t with all other inputs fixed."""
        gen = fitted_generator
        x = torch.randn(_LOSS_BATCH, _N_GRID_POINTS)

        t_low  = torch.full((_LOSS_BATCH,), 0.1)
        t_high = torch.full((_LOSS_BATCH,), 0.9)

        # Same seed for both generators so z is identical; t is the only diff.
        gen_low  = torch.Generator()
        gen_low.manual_seed(SEED)
        gen_high = torch.Generator()
        gen_high.manual_seed(SEED)

        loss_low = gen._loss_function(  # noqa: SLF001
            gen.score_model_, x, t_low, None, gen_low,
        )
        loss_high = gen._loss_function(  # noqa: SLF001
            gen.score_model_, x, t_high, None, gen_high,
        )

        assert loss_low.item() != loss_high.item(), (
            f"Loss was identical at t=0.1 and t=0.9 ({loss_low.item():.6f}), "
            "suggesting multiply_sigma does not depend on t."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestGenerate
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerate:
    """Tests for FunctionalDiffusionGenerator.generate."""

    # ─────────────────────────────────────────────────────────────────────────
    # Pre-fit guard
    # ─────────────────────────────────────────────────────────────────────────

    def test_generate_raises_if_not_fitted(self) -> None:
        """generate() must raise NotFittedError when called before fit()."""
        gen = FunctionalDiffusionGenerator()

        with pytest.raises(NotFittedError):
            gen.generate(n_samples=5)

    # ─────────────────────────────────────────────────────────────────────────
    # Return type and structure
    # ─────────────────────────────────────────────────────────────────────────

    def test_generate_returns_fdatagrid(
        self, fitted_generator: FunctionalDiffusionGenerator,
    ) -> None:
        """generate() must return an FDataGrid."""
        result = fitted_generator.generate(n_samples=5)

        assert isinstance(result, FDataGrid)

    def test_generate_n_samples_respected(
        self, fitted_generator: FunctionalDiffusionGenerator,
    ) -> None:
        """generate() must produce exactly n_samples observations."""
        result = fitted_generator.generate(n_samples=7)

        assert result.n_samples == 7  # noqa: PLR2004

    def test_generate_dim_domain_and_codomain(
        self, fitted_generator: FunctionalDiffusionGenerator,
    ) -> None:
        """Generated data must have dim_domain=1, dim_codomain=1."""
        result = fitted_generator.generate(n_samples=3)

        assert result.dim_domain == 1
        assert result.dim_codomain == 1

    def test_generate_grid_points_match_training(
        self,
        fitted_generator: FunctionalDiffusionGenerator,
        small_grid: FDataGrid,
    ) -> None:
        """Generated data must be discretized on the same grid as training."""
        result = fitted_generator.generate(n_samples=3)

        assert np.array_equal(result.grid_points[0], small_grid.grid_points[0])

    # ─────────────────────────────────────────────────────────────────────────
    # Gradient tracking
    # ─────────────────────────────────────────────────────────────────────────

    def test_generate_no_gradient_tracking(
        self, fitted_generator: FunctionalDiffusionGenerator,
    ) -> None:
        """The reverse pass must run inside torch.no_grad()."""
        stub = _StubReverseProcess()

        fitted_generator.generate(n_samples=3, reverse_process=stub)

        assert stub.call_count >= 1, "reverse() was never called"
        assert not any(stub.grad_enabled_during_call), (
            "Gradient tracking was enabled during at least one reverse() call;"
            " torch.no_grad() context may have been removed from generate()."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Custom reverse process
    # ─────────────────────────────────────────────────────────────────────────

    def test_custom_reverse_process_is_used(
        self, fitted_generator: FunctionalDiffusionGenerator,
    ) -> None:
        """generate() must delegate to the caller-supplied reverse_process."""
        stub = _StubReverseProcess()

        fitted_generator.generate(n_samples=3, reverse_process=stub)

        assert stub.call_count == 1

    def test_generate_with_labels_raises_for_default_model(
        self, fitted_generator: FunctionalDiffusionGenerator,
    ) -> None:
        """generate(y=...) must raise for the unconditional default model.

        The default UNetScoreModel does not support label conditioning, so
        the generate path must surface the same NotImplementedError as fit.
        """
        y = np.array([0.0, 1.0], dtype=np.float32)

        with pytest.raises(NotImplementedError):
            fitted_generator.generate(n_samples=2, y=y)

    # ─────────────────────────────────────────────────────────────────────────
    # Inverse transform — normalize path
    # ─────────────────────────────────────────────────────────────────────────

    def test_normalize_inverse_transform_applied(self) -> None:
        """generate() must apply the inverse normalization transform.

        Fit on sin(2πt)+5 so _bias_=5.0, _scale_=1.0. Stub returns zeros
        (normalized space), so expected output is 0.0 * 1.0 + 5.0 = 5.0.
        """
        t = np.linspace(0, 1, _N_GRID_POINTS)
        values = np.sin(2 * np.pi * t) + 5.0  # range ≈ [4, 6], _bias_ = 5.0
        data_matrix = np.tile(values, (_N_SAMPLES, 1)).reshape(
            _N_SAMPLES, _N_GRID_POINTS, 1,
        )
        shifted_grid = FDataGrid(data_matrix=data_matrix, grid_points=[t])

        gen = FunctionalDiffusionGenerator(
            normalize=True, max_iter=1, seed=SEED,
        )
        gen.fit(shifted_grid)

        stub = _StubReverseProcess()  # reverse returns zeros
        result = gen.generate(n_samples=2, reverse_process=stub)

        assert np.allclose(result.data_matrix, gen._bias_, atol=1e-5), (
            f"Expected all values ≈ _bias_={gen._bias_:.4f} after inverse "
            f"normalization, got min={result.data_matrix.min():.4f} "
            f"max={result.data_matrix.max():.4f}. "
            "The inverse transform may not have been applied."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Inverse transform — no-transform path
    # ─────────────────────────────────────────────────────────────────────────

    def test_no_transform_generate_does_not_rescale(
        self, small_grid: FDataGrid,
    ) -> None:
        """No rescaling when normalize=False and standardize=False."""
        gen = FunctionalDiffusionGenerator(
            normalize=False, standardize=False, max_iter=1, seed=SEED,
        )
        gen.fit(small_grid)

        stub = _StubReverseProcess()  # reverse returns zeros
        result = gen.generate(n_samples=1, reverse_process=stub)

        assert np.allclose(result.data_matrix, 0.0, atol=1e-6), (
            "Expected all-zero data_matrix when normalize=False and "
            "standardize=False, but values were non-zero. "
            "The rescaling block may have been entered incorrectly."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestGenerateEvolution
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerateEvolution:
    """Tests for FunctionalDiffusionGenerator.generate_evolution.

    The result has exactly len(timesteps) snapshots: result[0] is the initial
    noise, result[-1] is the final denoised state.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Pre-fit guard
    # ─────────────────────────────────────────────────────────────────────────

    def test_evolution_raises_if_not_fitted(self) -> None:
        """generate_evolution() must raise NotFittedError before fit()."""
        gen = FunctionalDiffusionGenerator()

        with pytest.raises(NotFittedError):
            gen.generate_evolution(5, timesteps=np.linspace(1.0, 0.0, 5))

    # ─────────────────────────────────────────────────────────────────────────
    # Return type and structure
    # ─────────────────────────────────────────────────────────────────────────

    def test_evolution_returns_list_of_fdatagrid(
        self, fitted_generator: FunctionalDiffusionGenerator,
    ) -> None:
        """generate_evolution() must return a list of FDataGrid objects."""
        result = fitted_generator.generate_evolution(
            3, timesteps=np.linspace(1.0, 0.0, 5),
        )

        assert isinstance(result, list)
        assert all(isinstance(r, FDataGrid) for r in result)

    def test_evolution_length_matches_timesteps(
        self, fitted_generator: FunctionalDiffusionGenerator,
    ) -> None:
        """The result list must have exactly len(timesteps) elements."""
        timesteps = np.linspace(1.0, 0.0, 6)

        result = fitted_generator.generate_evolution(3, timesteps=timesteps)

        assert len(result) == len(timesteps)

    def test_evolution_each_fdatagrid_has_correct_n_samples(
        self, fitted_generator: FunctionalDiffusionGenerator,
    ) -> None:
        """Every snapshot must contain exactly the requested n_samples."""
        n_samples = 5
        result = fitted_generator.generate_evolution(
            n_samples, timesteps=np.linspace(1.0, 0.0, 4),
        )

        assert all(r.n_samples == n_samples for r in result)

    # ─────────────────────────────────────────────────────────────────────────
    # Inverse transform on post-reverse snapshots
    # ─────────────────────────────────────────────────────────────────────────

    def test_evolution_inverse_transform_applied_to_all_elements(
        self,
    ) -> None:
        """Inverse transform must be applied to every snapshot, not just last.

        result[0] is the initial noise (before any reverse call) and is
        excluded from the assertion; result[1:] are the post-reverse snapshots.
        """
        t = np.linspace(0, 1, _N_GRID_POINTS)
        values = np.sin(2 * np.pi * t) + 5.0   # range ≈ [4, 6], _bias_ = 5.0
        data_matrix = np.tile(values, (_N_SAMPLES, 1)).reshape(
            _N_SAMPLES, _N_GRID_POINTS, 1,
        )
        shifted_grid = FDataGrid(data_matrix=data_matrix, grid_points=[t])

        gen = FunctionalDiffusionGenerator(
            normalize=True, max_iter=1, seed=SEED,
        )
        gen.fit(shifted_grid)

        stub = _StubReverseProcess()  # returns zeros for every call
        result = gen.generate_evolution(
            2, timesteps=np.linspace(1.0, 0.0, 3), reverse_process=stub,
        )

        for idx, snapshot in enumerate(result[1:], start=1):
            assert np.allclose(
                snapshot.data_matrix, gen._bias_, atol=1e-5,
            ), (
                f"result[{idx}].data_matrix is not ≈ "
                f"_bias_={gen._bias_:.4f}; "
                "the inverse transform may not have been applied to snapshot."
            )

    @pytest.mark.parametrize(
        "timesteps",
        [np.array([]), np.array([1.0])],
        ids=["empty", "single"],
    )
    def test_rejects_fewer_than_two_timesteps(
        self,
        fitted_generator: FunctionalDiffusionGenerator,
        timesteps: np.ndarray,
    ) -> None:
        """generate_evolution() needs at least a start and an end timestep.

        Fewer than two entries cannot define a trajectory; the docstring
        promises a snapshot per entry, so silently returning one snapshot
        would be misleading.
        """
        with pytest.raises(ValueError):  # noqa: PT011
            fitted_generator.generate_evolution(2, timesteps=timesteps)


# ─────────────────────────────────────────────────────────────────────────────
# TestSaveLoad
# ─────────────────────────────────────────────────────────────────────────────


class TestSaveLoad:
    """Tests for FunctionalDiffusionGenerator.save and .load."""

    # ─────────────────────────────────────────────────────────────────────────
    # Pre-fit guard
    # ─────────────────────────────────────────────────────────────────────────

    def test_save_raises_if_not_fitted(
        self, tmp_checkpoint: Path,
    ) -> None:
        """save() must raise NotFittedError when called before fit()."""
        gen = FunctionalDiffusionGenerator()

        with pytest.raises(NotFittedError):
            gen.save(tmp_checkpoint)

    # ─────────────────────────────────────────────────────────────────────────
    # File creation
    # ─────────────────────────────────────────────────────────────────────────

    def test_save_creates_file(
        self,
        fitted_generator: FunctionalDiffusionGenerator,
        tmp_checkpoint: Path,
    ) -> None:
        """save() must create a file at the given path."""
        fitted_generator.save(tmp_checkpoint)

        assert tmp_checkpoint.exists()

    # ─────────────────────────────────────────────────────────────────────────
    # Atomic write
    # ─────────────────────────────────────────────────────────────────────────

    def test_save_is_atomic_on_failure(
        self,
        fitted_generator: FunctionalDiffusionGenerator,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failed save must leave neither the target file nor a .tmp file.

        Patches torch.save to write the .tmp file and then raise OSError,
        reproducing a disk-full error after partial I/O.
        """
        target = tmp_path / "model.pt"
        _real_torch_save = torch.save

        def _write_then_fail(
            obj: object, path: object, **kwargs: object,
        ) -> None:
            _real_torch_save(obj, path, **kwargs)
            msg = "simulated disk full after write"
            raise OSError(msg)

        monkeypatch.setattr(torch, "save", _write_then_fail)

        with pytest.raises(OSError):  # noqa: PT011
            fitted_generator.save(target)

        assert not target.exists(), (
            "target file must not exist after a failed save"
        )
        assert not target.with_suffix(".tmp").exists(), (
            ".tmp file must be cleaned up after a failed save"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # load() — format validation
    # ─────────────────────────────────────────────────────────────────────────

    def test_load_raises_for_invalid_checkpoint_format(
        self, tmp_checkpoint: Path,
    ) -> None:
        """load() must raise TypeError when file contains a non-dict object."""
        torch.save("not_a_dict", tmp_checkpoint)

        with pytest.raises(TypeError, match="Invalid checkpoint format"):
            FunctionalDiffusionGenerator.load(tmp_checkpoint)

    def test_load_raises_for_missing_keys(
        self, tmp_checkpoint: Path,
    ) -> None:
        """load() must raise ValueError when required keys are absent."""
        torch.save({"version": 1}, tmp_checkpoint)

        with pytest.raises(ValueError, match="missing required keys"):
            FunctionalDiffusionGenerator.load(tmp_checkpoint)

    # ─────────────────────────────────────────────────────────────────────────
    # Round-trip — attribute preservation
    # ─────────────────────────────────────────────────────────────────────────

    def test_roundtrip_grid_points_preserved(
        self,
        fitted_generator: FunctionalDiffusionGenerator,
        tmp_checkpoint: Path,
    ) -> None:
        """grid_points_ must survive a save → load round-trip unchanged."""
        fitted_generator.save(tmp_checkpoint)
        gen2 = FunctionalDiffusionGenerator.load(tmp_checkpoint)

        assert np.array_equal(gen2.grid_points_, fitted_generator.grid_points_)

    def test_roundtrip_bias_and_scale_preserved(
        self,
        fitted_generator: FunctionalDiffusionGenerator,
        tmp_checkpoint: Path,
    ) -> None:
        """_bias_ and _scale_ must survive a save → load round-trip."""
        fitted_generator.save(tmp_checkpoint)
        gen2 = FunctionalDiffusionGenerator.load(tmp_checkpoint)

        assert gen2._bias_ == fitted_generator._bias_
        assert gen2._scale_ == fitted_generator._scale_

    def test_roundtrip_normalize_flag_preserved(
        self,
        fitted_generator: FunctionalDiffusionGenerator,
        tmp_checkpoint: Path,
    ) -> None:
        """The normalize flag must survive a save → load round-trip."""
        fitted_generator.save(tmp_checkpoint)
        gen2 = FunctionalDiffusionGenerator.load(tmp_checkpoint)

        assert gen2.normalize is True

    # ─────────────────────────────────────────────────────────────────────────
    # Round-trip — generate() works after load
    # ─────────────────────────────────────────────────────────────────────────

    def test_roundtrip_can_generate_after_load(
        self,
        fitted_generator: FunctionalDiffusionGenerator,
        tmp_checkpoint: Path,
    ) -> None:
        """A loaded generator must be able to call generate() without error."""
        fitted_generator.save(tmp_checkpoint)
        gen2 = FunctionalDiffusionGenerator.load(tmp_checkpoint)

        result = gen2.generate(n_samples=3)

        assert isinstance(result, FDataGrid)
        assert result.n_samples == 3  # noqa: PLR2004

    # ─────────────────────────────────────────────────────────────────────────
    # Device override
    # ─────────────────────────────────────────────────────────────────────────

    def test_load_device_override(
        self,
        fitted_generator: FunctionalDiffusionGenerator,
        tmp_checkpoint: Path,
    ) -> None:
        """load(device='cpu') must place all score model parameters on CPU."""
        fitted_generator.save(tmp_checkpoint)
        gen2 = FunctionalDiffusionGenerator.load(tmp_checkpoint, device="cpu")

        for param in gen2.score_model_.parameters():
            assert param.device.type == "cpu", (
                f"Parameter on {param.device} after load(device='cpu'); "
                "device override may not have been applied to the score model."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # RNG state reproducibility
    # ─────────────────────────────────────────────────────────────────────────

    def test_load_restores_rng_state_for_reproducibility(
        self,
        fitted_generator: FunctionalDiffusionGenerator,
        tmp_checkpoint: Path,
    ) -> None:
        """Two loads from the same checkpoint must yield identical outputs."""
        fitted_generator.save(tmp_checkpoint)

        gen_a = FunctionalDiffusionGenerator.load(tmp_checkpoint)
        gen_b = FunctionalDiffusionGenerator.load(tmp_checkpoint)

        result_a = gen_a.generate(n_samples=4)
        result_b = gen_b.generate(n_samples=4)

        assert np.array_equal(result_a.data_matrix, result_b.data_matrix), (
            "generate() produced different outputs from two loads of the same "
            "checkpoint. The RNG state stored in the checkpoint may not have "
            "been correctly restored."
        )

    def test_generate_after_load_continues_saved_stream(
        self,
        fitted_generator: FunctionalDiffusionGenerator,
        tmp_checkpoint: Path,
    ) -> None:
        """A loaded generator must continue the saved RNG stream.

        The checkpoint is taken after one generate() call, so the next draw
        from the original and from the loaded generator must agree. Each
        generate() builds a fresh seed-identical integrator, so the only
        persistent stream is the diffusion process generator restored from
        the checkpoint — this catches a restore that re-seeds from scratch.
        """
        fitted_generator.generate(n_samples=4)  # advance the process RNG
        fitted_generator.save(tmp_checkpoint)
        expected = fitted_generator.generate(n_samples=4)

        gen2 = FunctionalDiffusionGenerator.load(tmp_checkpoint)
        actual = gen2.generate(n_samples=4)

        assert np.array_equal(expected.data_matrix, actual.data_matrix)

    # ─────────────────────────────────────────────────────────────────────────
    # diff_process instance injection
    # ─────────────────────────────────────────────────────────────────────────

    def test_load_with_diff_process_instance_restores_state(
        self, small_grid: FDataGrid, tmp_checkpoint: Path,
    ) -> None:
        """load(diff_process=...) must restore fitted state into the instance.

        DiagonalDiffusionProcess stores a non-picklable callable, so load()
        must graft the checkpoint's numerical state onto the fresh instance.
        """
        def drift_term(t: torch.Tensor) -> torch.Tensor:
            return torch.ones(t.shape[0], _N_GRID_POINTS)

        def diff_term(t: torch.Tensor) -> torch.Tensor:
            return torch.ones(t.shape[0], _N_GRID_POINTS)

        proc = DiagonalDiffusionProcess(
            drift_term=drift_term, diffusion_term=diff_term,
        )
        gen = FunctionalDiffusionGenerator(
            diff_process=proc, max_iter=1, seed=SEED,
        )
        gen.fit(small_grid)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gen.save(tmp_checkpoint)

        # Reconstruct with a fresh process carrying the same callable.
        fresh_proc = DiagonalDiffusionProcess(
            drift_term=drift_term, diffusion_term=diff_term,
        )
        gen2 = FunctionalDiffusionGenerator.load(
            tmp_checkpoint, diff_process=fresh_proc,
        )

        x_test = torch.randn(2, gen2.diff_process_.M_)
        t_test = torch.full((2,), 0.5)
        gen2.diff_process_.mean_cond(x_test, t_test)  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# TestSklearnCompat
# ─────────────────────────────────────────────────────────────────────────────


class TestSklearnCompat:
    """scikit-learn estimator-protocol conformance for the generator."""

    def test_clone_returns_equivalent_unfitted_estimator(self) -> None:
        """clone() must return a new, unfitted estimator with equal params.

        Relies on the constructor storing its arguments verbatim (no defaults
        resolved, no validation) so the clone round-trips through get_params.
        """
        ve = VarianceExplodingDiffusionProcess(g_schedule="exponential")
        gen = FunctionalDiffusionGenerator(
            diff_process=ve, max_iter=3, batch_size=8, seed=7,
        )

        cloned = clone(gen)

        assert cloned is not gen
        assert cloned.max_iter == 3  # noqa: PLR2004
        assert cloned.batch_size == 8  # noqa: PLR2004
        assert cloned.seed == 7  # noqa: PLR2004
        assert type(cloned.diff_process) is type(gen.diff_process)
        assert not hasattr(cloned, "diff_process_")

    def test_set_params_round_trip(self) -> None:
        """get_params/set_params must round-trip and accept overrides."""
        gen = FunctionalDiffusionGenerator(max_iter=3, seed=7)

        gen.set_params(**gen.get_params())  # must not raise
        gen.set_params(max_iter=9)

        assert gen.max_iter == 9  # noqa: PLR2004


# ─────────────────────────────────────────────────────────────────────────────
# TestFromFDataGridToTensorDataset
# ─────────────────────────────────────────────────────────────────────────────


class TestFromFDataGridToTensorDataset:
    """Tests for the module-level utility _fdatagrid_to_tensor_dataset."""

    # ─────────────────────────────────────────────────────────────────────────
    # Number of tensors in the dataset
    # ─────────────────────────────────────────────────────────────────────────

    def test_without_labels_returns_single_tensor_dataset(
        self, small_grid: FDataGrid,
    ) -> None:
        """Without y, the dataset must contain exactly one tensor."""
        ds = _fdatagrid_to_tensor_dataset(small_grid)

        assert len(ds.tensors) == 1

    def test_with_labels_returns_two_tensor_dataset(
        self, small_grid: FDataGrid, labels_array: np.ndarray,
    ) -> None:
        """With y passed, the TensorDataset must have exactly two tensors."""
        ds = _fdatagrid_to_tensor_dataset(small_grid, y=labels_array)

        assert len(ds.tensors) == 2  # noqa: PLR2004

    # ─────────────────────────────────────────────────────────────────────────
    # Data tensor — shape and dtype
    # ─────────────────────────────────────────────────────────────────────────

    def test_data_tensor_shape(
        self, small_grid: FDataGrid,
    ) -> None:
        """Data tensor must have shape (N, M), not (N, M, 1)."""
        ds = _fdatagrid_to_tensor_dataset(small_grid)

        expected_shape = (small_grid.n_samples, len(small_grid.grid_points[0]))
        assert ds.tensors[0].shape == expected_shape

    def test_data_tensor_dtype_float32(
        self, small_grid: FDataGrid,
    ) -> None:
        """The data tensor must be float32."""
        ds = _fdatagrid_to_tensor_dataset(small_grid)

        assert ds.tensors[0].dtype == torch.float32

    # ─────────────────────────────────────────────────────────────────────────
    # Label tensor — values and shape
    # ─────────────────────────────────────────────────────────────────────────

    def test_labels_tensor_matches_input(
        self, small_grid: FDataGrid, labels_array: np.ndarray,
    ) -> None:
        """The label tensor must contain exactly the values passed in y."""
        ds = _fdatagrid_to_tensor_dataset(small_grid, y=labels_array)

        expected = torch.from_numpy(labels_array).float()
        assert torch.allclose(ds.tensors[1], expected)

    def test_labels_tensor_shape(
        self, small_grid: FDataGrid, labels_array: np.ndarray,
    ) -> None:
        """The label tensor must have shape (N,), not (N, 1) or (1, N)."""
        ds = _fdatagrid_to_tensor_dataset(small_grid, y=labels_array)

        assert ds.tensors[1].shape == (small_grid.n_samples,)
