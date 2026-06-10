"""Forward diffusion processes for score-based generative models."""

import pickle
import warnings
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Final, Literal, Protocol, TypedDict

import numpy as np
import torch
from sklearn.utils.validation import check_is_fitted
from torch import Tensor

from ..._utils._sklearn_adapter import BaseEstimator
from .torch_random_numbers import make_torch_rng

FitStateType = int | float | Tensor | torch.device
CheckpointDict = TypedDict(
    "CheckpointDict",
    {
        "class": type["ForwardDiffusionProcess"],
        "init_kwargs": dict[str, Any],
        "fit_state": dict[str, FitStateType],
    },
)

class DiffusionProcess(Protocol):
    """Protocol for a diffusion process used in generative models.

    It defines the drift and diffusion terms of an SDE.
    """
    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """Computes the drift term of the SDE at time t."""
        ...

    def diffusion(self, t: Tensor) -> Tensor:
        """Computes the diffusion term of the SDE at time t."""
        ...

    def diffusion_times_v(self, v: Tensor, t: Tensor) -> Tensor:
        """Computes the diffusion term of the SDE at time t multiplied by v."""
        ...

    def diffusion_gram_times_v(self, v: Tensor, t: Tensor) -> Tensor:
        r"""Compute :math:`G(t)\,G(t)^\top v`, G not assumed symmetric."""
        ...

class CustomDiffusionProcess:
    """Common interface of a diffusion process to be used in generative models.

    It defines the drift and diffusion terms of an SDE.
    """
    def __init__(self,
                 drift: Callable[[Tensor, Tensor], Tensor],
                 diffusion: Callable[[Tensor], Tensor]) -> None:
        """Initialize the diffusion process.

        Args:
            drift: A function that takes the current state x and time t,
                and returns the drift term of the SDE, shape (N, M) or (N,).
            diffusion: A function that takes the current time t and returns
                the diffusion term of the SDE, shape (N,), (N, M) or
                (N, M, M).
        """
        self.f = drift
        self.g = diffusion

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """Computes the drift term of the SDE at time t.

        Args:
            x: The current functional data as a tensor, shape (N, M)
            t: The current time steps as a tensor, shape (N,)

        Returns:
            The drift term as a tensor Shape (N, M) or (N,)
        """
        return self.f(x, t)

    def diffusion(self, t: Tensor) -> Tensor:
        """Computes the diffusion term of the SDE at time t.

        Output shape can be (N, M, M), (N, M) or (N,).

        If (N,) is returned, it represents a scalar
        multiple of the identity matrix. Hence, when used
        its behavior should be the same as if a scalar
        multiple of the identity matrix is returned.

        If (N, M) is returned, it represents only the diagonal
        of the diffusion matrix. Hence, when used its behavior
        should be the same as if a diagonal matrix is returned.

        Args:
            t: The current time steps as a tensor, shape (N,)

        Returns:
            The diffusion term as a tensor Shape (N,), (N, M) or (N, M, M)
        """
        return self.g(t)

    def diffusion_times_v(self, v: Tensor, t: Tensor) -> Tensor:
        """Computes the diffusion term of the SDE at time t multiplied by v.

        This method is used to compute the product of the diffusion term
        with a vector v, which is a common operation in SDE solvers.
        The implementation should be efficient and take into account
        the shape of the diffusion term.

        Args:
            v: The vector to multiply with the diffusion term, shape (N, M)
            t: The current time steps as a tensor, shape (N,)

        Returns:
            The product of the diffusion term with v, shape (N, M)
        """
        diffusion_term = self.diffusion(t)
        if diffusion_term.ndim == 1:
            return diffusion_term.unsqueeze(1) * v

        if diffusion_term.ndim == 2:  # noqa: PLR2004 # Ignore magic value
            return diffusion_term * v

        if diffusion_term.ndim == 3:  # noqa: PLR2004 # Ignore magic value
            return torch.einsum("nij,nj->ni", diffusion_term, v)

        msg = f"Invalid diffusion term shape: {diffusion_term.shape}"
        raise ValueError(msg)

    def diffusion_gram_times_v(self, v: Tensor, t: Tensor) -> Tensor:
        r"""Compute :math:`G(t)\,G(t)^\top v`, G not assumed symmetric.

        Args:
            v: The vector to multiply, shape (N, M)
            t: The time steps as a tensor, shape (N,)

        Returns:
            :math:`G(t)\,G(t)^\top v`, shape (N, M)
        """
        diffusion_term = self.diffusion(t)
        if diffusion_term.ndim == 1:
            return diffusion_term.unsqueeze(1) ** 2 * v
        if diffusion_term.ndim == 2:  # noqa: PLR2004 # Ignore magic value
            return diffusion_term ** 2 * v
        gt_v = torch.einsum("nji,nj->ni", diffusion_term, v)
        return torch.einsum("nij,nj->ni", diffusion_term, gt_v)

class ForwardDiffusionProcess(BaseEstimator):
    r"""Defines the forward diffusion process of a generative model.

    Represents a diffusion process with Gaussian conditional distributions.
    The public operator interface is:

    - ``multiply_sigma(z, t)``: apply the square-root covariance factor to z.
    - ``multiply_cov(h, t)``: apply the covariance matrix to h.
    - ``multiply_inv_sigma(h, t)``: apply the inverse square-root factor to h.

    Subclasses must also implement ``mean_cond`` and
    ``sample_limit_distribution``.

    **Diagonal-mean contract for the Tweedie final step.**
    The generic reverse path in ``SDEReverseDiffusionProcess`` assumes that
    :math:`\boldsymbol{\mu}_t(x) = a(t) \odot x` (diagonal in the canonical
    basis), so that :math:`A(t)^{-1} h = h \mathbin{/} a(1)`.  This holds for
    ``VariancePreservingDiffusionProcess``,
    ``VarianceExplodingDiffusionProcess``,
    and ``DiagonalDiffusionProcess``.  Subclasses whose conditional mean is NOT
    diagonal in the canonical basis must register a custom reverse overload via
    ``SDEReverseDiffusionProcess.reverse.register``, as
    ``CirculantSymmetricMatrixDiffusionProcess`` does; otherwise the Tweedie
    final step will silently produce incorrect results. This is done because
    integrating the SDE with a non-diagonal mean is more expensive, and we
    recommend using a custom reverse method in a diagonal basis
    for better performance.
    """
    def __init__(self, seed: int | None = None) -> None:
        """Initializes the diffusion process.

        Args:
            seed: The random seed for sampling from the limit distribution.
        """
        self.seed = seed

    def fit(self, x: Tensor) -> "ForwardDiffusionProcess":
        """Fits the parameters of the diffusion process to the data.

        This method can be used to fit the parameters of the
        diffusion process to the data. For the most basic, it learns the
        dimensionality and builds the generator for sampling from the
        limit distribution.
        More advanced diffusion process can learn other parameters such as
        the beta(t) schedule for variance-preserving.

        Args:
            x: The input functional data as a tensor, shape (N, M)

        Returns:
            self: The fitted diffusion process.
        """
        self.M_ = x.shape[1]  # Learn the dimension of the data
        # The generator must be in CPU to ensure reproducibility across devices
        self.generator_ = make_torch_rng(self.seed, device="cpu")
        return self

    def _get_fit_state(self) -> dict[str, FitStateType]:
        """Return the fitted attributes as a serializable dictionary.

        The base implementation stores only ``self.M_``.  Subclasses should
        override this to include their own fitted attributes and call
        ``super()._get_fit_state()`` to merge in the base state:

        .. code-block:: python

            def _get_fit_state(self) -> dict:
                state = super()._get_fit_state()
                state["my_param"] = self.my_param_
                return state

        Returns:
            dict: Mapping of attribute names to serializable values.
                All values must be compatible with ``torch.save`` (tensors,
                numpy arrays, Python primitives, or nested structures thereof).

        Raises:
            sklearn.exceptions.NotFittedError: If :meth:`fit` has not been
                called yet.
        """
        check_is_fitted(self, attributes=["M_", "generator_"])

        return {"M": self.M_,
                "generator_state": self.generator_.get_state()}


    def _restore_fit_state(self, data: dict[str, FitStateType]) -> None:
        """Restore fitted attributes from a serialized state dictionary.

        The base implementation restores ``self.M_``.  Subclasses should
        override this to restore their own fitted attributes and call
        ``super()._restore_fit_state(data)`` first:

        .. code-block:: python

            def _restore_fit_state(self, data: dict) -> None:
                super()._restore_fit_state(data)
                self.my_param_ = data["my_param"]

        Args:
            data: A dictionary as produced by :meth:`_get_fit_state`.

        Raises:
            ValueError: If ``"M"`` is absent from *data*
            TypeError: If the stored class is not a valid subclass,
                       or *instance* has the wrong type.
        """
        if "M" not in data:
            msg = (
                "'M' is missing from the fit state dictionary. "
                "Ensure the checkpoint was produced by a compatible process."
            )
            raise ValueError(
                msg,
            )
        if not isinstance(data["M"], int):
            msg = f"Expected 'M' to be an int, got {type(data['M']).__name__}."
            raise TypeError(
                msg,
            )

        if "generator_state" not in data:
            msg = (
                "'generator_state' is missing from the fit state dictionary. "
                "Ensure the checkpoint was produced by a compatible process."
            )
            raise ValueError(
                msg,
            )
        if not isinstance(data["generator_state"], Tensor):
            got = type(data["generator_state"]).__name__
            msg = f"Expected 'generator_state' to be a Tensor, got {got}."
            raise TypeError(
                msg,
            )
        self.M_ = data["M"]
        self.generator_ = torch.Generator(device="cpu")
        self.generator_.set_state(data["generator_state"])

    # -------------------------------------------------------------------------
    # Public checkpoint interface (Rarely override in subclasses)
    # -------------------------------------------------------------------------

    def to_checkpoint(self) -> CheckpointDict:
        """Serialize the full process state to a self-describing checkpoint.

        The checkpoint contains everything needed to reconstruct this instance
        from scratch via :meth:`from_checkpoint`: the concrete class object,
        the constructor arguments, and the fitted state.  Subclasses achieve
        custom serialization by overriding :meth:`_get_fit_state` and
        :meth:`_restore_fit_state` only — this method should not be overridden.

        The returned dict is designed to be stored with ``torch.save`` and
        recovered with ``torch.load(weights_only=False)``.  The ``"class"``
        entry is the class object itself; pickle serializes it as a
        fully-qualified dotted path and resolves it on load automatically,
        so no string registry is needed.

        Returns:
            dict: A dictionary with three keys:

            ``"class"``
                The concrete class of this instance (e.g.
                ``VariancePreservingDiffusionProcess``).
            ``"init_kwargs"``
                Constructor keyword arguments returned by
                :meth:`~sklearn.base.BaseEstimator.get_params`.
            ``"fit_state"``
                Fitted attributes returned by :meth:`_get_fit_state`.

        Raises:
            sklearn.exceptions.NotFittedError: If :meth:`fit` has not been
                called yet (propagated from :meth:`_get_fit_state`).
        """
        return {
            "class": type(self),
            "init_kwargs": self.get_params(),
            "fit_state": self._get_fit_state(),
        }

    @classmethod
    def from_checkpoint(
        cls,
        data: CheckpointDict,
        instance: "ForwardDiffusionProcess | None" = None,
    ) -> "ForwardDiffusionProcess":
        """Reconstruct a fitted diffusion process from a checkpoint dictionary.

        If *instance* is provided and its type matches the stored class, the
        checkpoint's ``init_kwargs`` are ignored and ``_restore_fit_state`` is
        called directly on *instance*. This allows subclasses whose constructor
        arguments are not picklable (e.g. ``DiagonalDiffusionProcess`` with
        lambda callables) to be restored by supplying a pre-built instance
        with those callables already attached.

        Args:
            data:     Checkpoint dict as produced by :meth:`to_checkpoint`.
            instance: Optional pre-built instance to restore state into.
                    Must be of the exact type stored in the checkpoint.

        Raises:
            ValueError: If required keys are missing.
            TypeError: If the stored class is not a
                valid subclass, or *instance* has the wrong type.
        """
        required = {"class", "init_kwargs", "fit_state"}
        missing = required - data.keys()
        if missing:
            msg = f"Checkpoint is missing required keys: {sorted(missing)}."
            raise ValueError(
                msg,
            )

        process_cls = data["class"]
        if not (
            isinstance(process_cls, type)
            and issubclass(process_cls, ForwardDiffusionProcess)
        ):
            msg = (
                "Expected 'class' to be a subclass of"
                f" ForwardDiffusionProcess, got {process_cls!r}."
            )
            raise TypeError(msg)

        if instance is not None:
            if type(instance) is not process_cls:
                msg = (
                    f"Provided instance is of type"
                    f" {type(instance).__name__!r} "
                    f"but the checkpoint contains a {process_cls.__name__!r}. "
                    "Types must match exactly."
                )
                raise TypeError(
                    msg,
                )
            instance._restore_fit_state(data["fit_state"]) # noqa: SLF001 # Ignore accessing private method
            return instance

        process = process_cls(**data["init_kwargs"])
        process._restore_fit_state(data["fit_state"])  # noqa: SLF001 # Ignore accessing private method
        return process

    @abstractmethod
    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """Computes the drift term of the SDE at time t."""
        ...

    @abstractmethod
    def diffusion(self, t: Tensor) -> Tensor:
        """Computes the diffusion term of the SDE at time t."""
        ...

    def diffusion_times_v(self, v: Tensor, t: Tensor) -> Tensor:
        """Computes the diffusion term of the SDE at time t multiplied by v."""
        diffusion_term = self.diffusion(t)
        if diffusion_term.ndim == 1:
            return diffusion_term.unsqueeze(1) * v

        if diffusion_term.ndim == 2: # noqa: PLR2004 # Ignore magic value
            return diffusion_term * v

        if diffusion_term.ndim == 3: # noqa: PLR2004 # Ignore magic value
            return torch.einsum("nij,nj->ni", diffusion_term, v)

        msg = f"Invalid diffusion term shape: {diffusion_term.shape}"
        raise ValueError(msg)

    def diffusion_gram_times_v(self, v: Tensor, t: Tensor) -> Tensor:
        r"""Compute :math:`G(t)\,G(t)^\top v`, G not assumed symmetric.

        Args:
            v: The vector to multiply, shape (N, M)
            t: The time steps as a tensor, shape (N,)

        Returns:
            :math:`G(t)\,G(t)^\top v`, shape (N, M)
        """
        diffusion_term = self.diffusion(t)
        if diffusion_term.ndim == 1:
            return diffusion_term.unsqueeze(1) ** 2 * v
        if diffusion_term.ndim == 2: # noqa: PLR2004 # Ignore magic value
            return diffusion_term ** 2 * v
        gt_v = torch.einsum("nji,nj->ni", diffusion_term, v)
        return torch.einsum("nij,nj->ni", diffusion_term, gt_v)

    @abstractmethod
    def mean_cond(self, x: Tensor, t: Tensor) -> Tensor:
        """Computes the mean of the conditional distribution at time t.

        It computes the mean of the distribution of X(t) given X(0) = x.

        Args:
            x: The current functional data as a tensor, shape (N, M)
            t: The current time steps as a tensor, shape (N,)

        Returns:
            The mean of the conditional distribution at time t, shape (N, M)
        """
        ...

    @abstractmethod
    def sample_limit_distribution(
        self,
        n_samples: int,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """Samples data from the limit distribution of the diffusion process.

        The output shape must be (n_samples, M) where M is the dimension of
        the data. It is learned at the ``fit`` method.

        Args:
            n_samples: The number of samples to generate.
            device: The device on which to create the samples. Default is
                "cpu".

        Returns:
            X: Tensor with samples from the limit distribution,
                shape (n_samples, M)
        """
        ...

    @abstractmethod
    def multiply_sigma(self, z: Tensor, t: Tensor) -> Tensor:
        """Apply the square-root covariance factor to z.

        Args:
            z: Noise vector, same shape as the data (N, M).
            t: Time steps, shape (N,).

        Returns:
            Square-root covariance factor applied to z, shape (N, M).
        """
        ...

    def multiply_cov(self, h: Tensor, t: Tensor) -> Tensor:
        """Apply the covariance operator Cov(t) = sigma(t) sigma(t)^T to h.

        Default implementation calls multiply_sigma twice (valid when sigma is
        symmetric). Subclasses may override for efficiency.

        Args:
            h: Vector, same shape as the data (N, M).
            t: Time steps, shape (N,).

        Returns:
            Cov(t) applied to h, shape (N, M).
        """
        return self.multiply_sigma(self.multiply_sigma(h, t), t)

    @abstractmethod
    def multiply_inv_sigma(self, h: Tensor, t: Tensor) -> Tensor:
        """Apply the inverse square-root covariance factor to h.

        Args:
            h: Vector, same shape as the data (N, M).
            t: Time steps, shape (N,).

        Returns:
            Inverse square-root factor applied to h, shape (N, M).
        """
        ...


class VariancePreservingDiffusionProcess(ForwardDiffusionProcess):
    r"""Implements a variance-preserving diffusion process.

    Based on the following forward SDE:

    .. math::

        d\mathbf{X}(t) = -\tfrac{1}{2}\beta(t)\mathbf{X}(t)dt
        + \sqrt{\beta(t)}\,d\mathbf{W}(t)

    where :math:`\beta(t)` is a time-dependent function controlling
    the noise level.

    The linear and cosine schedules for :math:`\beta(t)` are implemented,
    parameterized by :math:`\beta(0)` and :math:`\beta(T)`. The time horizon
    :math:`T` is fixed to 1, as all schedules depend only on the ratio
    :math:`t/T`.

    For the cosine schedule, :math:`\beta(t)` is clamped to
    :math:`[\beta(0), \beta(T)]` to avoid large values near :math:`T`
    that can cause numerical instability.
    """
    # T is fixed to 1 by convention: the diffusion schedule is
    # fully characterized by beta_0 and beta_T, and all time-dependent
    # expressions depend only on the ratio t/T. Callers are expected
    # to pass t in [0, 1].
    T: Final[int] = 1

    def __init__(
        self,
        beta_schedule: Literal["linear", "cosine"] = "cosine",
        beta_min: float = 0.,
        beta_max: float = 10.,
        seed: int | None = None,
    ) -> None:
        """Initializes the variance-preserving diffusion process.

        Args:
            beta_schedule: The schedule for beta(t). Can be 'linear' or
                        'cosine'. Default is 'cosine'.
            beta_min: The value of beta(0). Default is 0.
            beta_max: The value of beta(T). Default is 10.0.
            seed: The random seed for sampling from the limit distribution.
        """
        super().__init__(seed=seed)

        if beta_schedule not in ("linear", "cosine"):
            msg = f"Unknown beta schedule: {beta_schedule}"
            raise ValueError(msg)
        self.beta_schedule = beta_schedule
        self.beta_min = beta_min
        self.beta_max = beta_max
        # Small constant for numerical stability in cosine schedule
        self.s = torch.tensor(1e-3)

    def _beta_t(self, t: Tensor) -> Tensor:
        """Computes the value of beta(t) at time t.

        Args:
            t: The time steps as a tensor, shape (N,)

        Returns:
            The value of beta(t) as a tensor, shape (N,)
        """
        if self.beta_schedule == "linear":
            beta_t = (
                self.beta_min + (self.beta_max - self.beta_min) * t / self.T
            )
        elif self.beta_schedule == "cosine":
            beta_t = (torch.pi / (self.T * (self.s + 1)) *
                      torch.tan(
                                torch.pi * 0.5 *
                                (t / self.T + self.s) / (self.s + 1),
                                )
                      )
            beta_t = torch.clamp(
                beta_t, min=self.beta_min, max=self.beta_max,
            )
        else:
            msg = f"Unknown beta schedule: {self.beta_schedule}"
            raise ValueError(msg)
        return beta_t

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        r"""Computes the drift term of the SDE at time t.

        For the variance-preserving diffusion process, the drift term is:

        .. math::

            \mathbf{f}(\mathbf{X}(t), t) = -\tfrac{1}{2} \beta(t) \mathbf{X}(t)
        """
        return -0.5 * self._beta_t(t).unsqueeze(1) * x

    def diffusion(self, t: Tensor) -> Tensor:
        r"""Computes the diffusion term of the SDE at time t.

        For the variance-preserving diffusion process, the diffusion term
        is :math:`g(t) = \sqrt{\beta(t)}`.
        """
        return torch.sqrt(self._beta_t(t))

    def mean_cond(self, x: Tensor, t: Tensor) -> Tensor:
        r"""Computes the mean of the conditional distribution at time t.

        For the variance-preserving diffusion process:

        .. math::

            \boldsymbol{\mu}_t =
            \exp\!\Bigl(-\tfrac{1}{2}\int_0^t \beta(s)\,ds\Bigr)\,\mathbf{x}

        Args:
            x: The data to condition on as a tensor, shape (N, M)
            t: The time steps as a tensor, shape (N,)

        Returns:
            mu_t: The mean of the diffusion process at time t, shape (N,M)

        Examples:
            Boundary conditions — identity at t = 0, zero at t = 1 (cosine):

            >>> import torch
            >>> from skfda.ml.generative import (
            ...     VariancePreservingDiffusionProcess,
            ... )
            >>> vp = VariancePreservingDiffusionProcess(beta_schedule="cosine")
            >>> x = torch.ones(3, 8)
            >>> torch.allclose(vp.mean_cond(x, torch.zeros(3)), x)
            True
            >>> vp.mean_cond(x, torch.ones(3)).abs().max().item() < 1e-5
            True
        """
        if self.beta_schedule == "linear":

            integral_beta = (
                self.beta_min * t
                + 0.5 * (self.beta_max - self.beta_min) * (t ** 2) / self.T
            )

            mu_t = torch.exp(-0.5 * integral_beta)  # Shape (N,)

        elif self.beta_schedule == "cosine":
            f_t = torch.cos(
                ((t / self.T + self.s) / (1 + self.s)) * (torch.pi / 2),
            ) ** 2
            f_0 = torch.cos((self.s / (1 + self.s)) * (torch.pi / 2)) ** 2

            mu_t = torch.sqrt(f_t / f_0)  # Shape (N,)
        return mu_t.unsqueeze(-1) * x

    def _cov(self, t: Tensor) -> Tensor:
        r"""Returns the scalar conditional variance at time t, shape (N,).

        For the variance-preserving process:
        :math:`\sigma_t^2 = 1 - \exp(-\int_0^t \beta(s) ds)`
        """
        if self.beta_schedule == "linear":
            integral_beta = (
                self.beta_min * t
                + 0.5 * (self.beta_max - self.beta_min) * (t ** 2) / self.T
            )
            return 1 - torch.exp(-integral_beta)

        if self.beta_schedule == "cosine":
            f_t = torch.cos(
                ((t / self.T + self.s) / (1 + self.s)) * (torch.pi / 2),
            ) ** 2
            f_0 = torch.cos((self.s / (1 + self.s)) * (torch.pi / 2)) ** 2
            return 1 - (f_t / f_0)

        msg = f"Unknown beta schedule: {self.beta_schedule}"
        raise ValueError(msg)

    def multiply_sigma(self, z: Tensor, t: Tensor) -> Tensor:
        r"""Scale z by :math:`\sigma_t = \sqrt{1 - e^{-\int_0^t\beta\,ds}}`.

        Examples:
            Cosine schedule boundary values — sigma_0 = 0, sigma_T = 1:

            >>> import torch
            >>> from skfda.ml.generative import (
            ...     VariancePreservingDiffusionProcess,
            ... )
            >>> vp = VariancePreservingDiffusionProcess(beta_schedule="cosine")
            >>> z = torch.ones(3, 8)
            >>> t0, t1 = torch.zeros(3), torch.ones(3)
            >>> vp.multiply_sigma(z, t0).abs().max().item() < 1e-5
            True
            >>> torch.allclose(vp.multiply_sigma(z, t1), z, atol=1e-5)
            True
        """
        return torch.sqrt(self._cov(t)).unsqueeze(1) * z

    def multiply_cov(self, h: Tensor, t: Tensor) -> Tensor:
        r"""Scale h by :math:`\sigma_t^2 = 1 - e^{-\int_0^t\beta(s)ds}`."""
        return self._cov(t).unsqueeze(1) * h

    def multiply_inv_sigma(self, h: Tensor, t: Tensor) -> Tensor:
        r"""Scale h by :math:`1/\sigma_t`, clamped for numerical stability.

        Examples:
            Inverse is a left inverse of ``multiply_sigma``:

            >>> import torch
            >>> from skfda.ml.generative import (
            ...     VariancePreservingDiffusionProcess,
            ... )
            >>> vp = VariancePreservingDiffusionProcess(beta_schedule="cosine")
            >>> z = torch.randn(3, 8)
            >>> t = torch.full((3,), 0.5)
            >>> h = vp.multiply_sigma(z, t)
            >>> torch.allclose(vp.multiply_inv_sigma(h, t), z, atol=1e-5)
            True
        """
        inv_sigma = 1.0 / torch.sqrt(torch.clamp(self._cov(t), min=1e-6))
        return inv_sigma.unsqueeze(1) * h

    def sample_limit_distribution(
            self,
            n_samples: int,
            device: torch.device | str = "cpu",
    ) -> Tensor:
        r"""Samples data from the final distribution of the diffusion process.

        For the variance-preserving diffusion process, the final distribution
        at time T is given by a standard normal distribution:
        :math:`\mathbf{X}(T) \sim \mathcal{N}(0, I)`

        Args:
            n_samples: The number of samples to generate.
            grid_size: The number of discretization points of the
                functional data. Refers to the M dimension of the data.
            device: The device on which to create the samples.
                    Default is "cpu".

        Returns:
            X: Tensor with samples from the final distribution.
            Shape (n_samples, grid_size)

        Examples:
            Raises ``NotFittedError`` before fitting:

            >>> from skfda.ml.generative import (
            ...     VariancePreservingDiffusionProcess,
            ... )
            >>> VariancePreservingDiffusionProcess().sample_limit_distribution(
            ...     n_samples=5,
            ... )  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            sklearn.exceptions.NotFittedError: ...

            Shape after fit:

            >>> import torch
            >>> vp = VariancePreservingDiffusionProcess(seed=0)
            >>> _ = vp.fit(torch.zeros(10, 32))
            >>> vp.sample_limit_distribution(n_samples=5).shape
            torch.Size([5, 32])
        """
        check_is_fitted(self, attributes=["M_", "generator_"])
        # Noise is generated in CPU to ensure reproducibility cross-device
        return torch.randn(
            (n_samples, self.M_), device="cpu", generator=self.generator_,
        ).to(device, non_blocking=True)


class VarianceExplodingDiffusionProcess(ForwardDiffusionProcess):
    r"""Implements a variance-exploding diffusion process.

    Based on the following forward SDE:

    .. math::

        d\mathbf{X}(t) = g(t)\,d\mathbf{W}(t)

    where :math:`g(t)` is a time-dependent function controlling
    the noise level.

    The linear and exponential schedules for :math:`g(t)` are implemented,
    parameterized by :math:`g(0)` and :math:`g(T)`. The time horizon
    :math:`T` is fixed to 1, as all schedules depend only on the ratio
    :math:`t/T`.
    """
    # T is fixed to 1 by convention: the diffusion schedule is
    # fully characterized by g_0 and g_T, and all time-dependent
    # expressions depend only on the ratio t/T. Callers are expected
    # to pass t in [0, 1].
    T: Final[int] = 1

    def __init__(
        self,
        g_schedule: Literal["linear", "exponential"] = "exponential",
        g_0: float = 0.1,
        g_T: float = 15.,  # noqa: N803 # Ignore uppercase variable
        seed: int | None = None,
    ) -> None:
        """Initializes the variance-exploding diffusion process.

        Args:
            g_schedule: The schedule for g(t). Can be 'linear' or
                        'exponential'. Default is 'exponential'.
            g_0: The value of g(0).
            g_T: The value of g(T).
            seed: The random seed for sampling from the limit distribution.

        """
        super().__init__(seed=seed)

        if g_schedule not in ("linear", "exponential"):
            msg = f"Unknown g schedule: {g_schedule}"
            raise ValueError(msg)

        self.g_schedule = g_schedule
        self.g_0 = g_0
        self.g_T = g_T

    def drift(self, x: Tensor, t: Tensor) -> Tensor: # noqa: ARG002 # Ignore unused arguments
        r"""Computes the drift term of the SDE at time t.

        For the variance-exploding diffusion process, the drift term is zero:
        :math:`\mathbf{f}(\mathbf{X}(t), t) = 0`
        """
        return torch.zeros_like(x)

    def diffusion(self, t: Tensor) -> Tensor:
        r"""Computes the diffusion term of the SDE at time t.

        For the variance-exploding diffusion process:

        - linear: :math:`g(t) = g(0) + (g(T) - g(0)) \cdot t / T`
        - exponential: :math:`g(t) = g(0) \cdot (g(T) / g(0))^{t/T}`
        """
        if self.g_schedule == "linear":
            return self.g_0 + (self.g_T - self.g_0) * t / self.T
        if self.g_schedule == "exponential":
            return self.g_0 * torch.pow(self.g_T / self.g_0, t / self.T)
        msg = f"Unknown g schedule: {self.g_schedule}"
        raise ValueError(msg)

    def mean_cond(self, x: Tensor, t: Tensor) -> Tensor:  # noqa: ARG002 # Ignore unused argument
        r"""Computes the mean of the conditional distribution at time t.

        For the variance-exploding process the mean equals the initial
        condition (zero drift):

        .. math::

            \boldsymbol{\mu}_t = \mathbf{x}

        Args:
            x: The data to condition on as a tensor, shape (N, M)
            t: The time steps as a tensor, shape (N,)

        Returns:
            mu_t: The mean of the diffusion process at time t, shape (N,M)

        Examples:
            Mean equals the initial condition for all t (zero drift):

            >>> import torch
            >>> from skfda.ml.generative import (
            ...     VarianceExplodingDiffusionProcess,
            ... )
            >>> ve = VarianceExplodingDiffusionProcess()
            >>> x = torch.randn(4, 16)
            >>> torch.allclose(ve.mean_cond(x, torch.rand(4)), x)
            True
        """
        return x

    def _cov(self, t: Tensor) -> Tensor:
        r"""Returns the scalar conditional variance at time t, shape (N,).

        For the variance-exploding process:

        .. math::

            \sigma_t^2 = \int_0^t g^2(s)\,ds

        When :math:`g_0 = g_T` (constant diffusion), the integral reduces
        to :math:`g_0^2\,t`.
        """
        if self.g_schedule == "linear":
            return (
                self.g_0**2 * t
                + self.g_0 * (self.g_T - self.g_0) * t ** 2 / self.T
                + (self.g_T - self.g_0)**2 * t ** 3 / (3 * self.T ** 2)
            )
        if self.g_schedule == "exponential":
            if self.g_0 != self.g_T:
                log_term: float = float(np.log(self.g_T / self.g_0))
                cov: Tensor = (
                    ((self.g_0 ** 2) * self.T / (2 * log_term))
                    * (((self.g_T / self.g_0) ** (2 * t / self.T)) - 1)
                )
                return cov
            return self.g_0 ** 2 * t
        msg = f"Unknown g schedule: {self.g_schedule}"
        raise ValueError(msg)

    def multiply_sigma(self, z: Tensor, t: Tensor) -> Tensor:
        r"""Scale z by :math:`\sigma_t = \sqrt{\int_0^t g^2(s)\,ds}`.

        Examples:
            No noise injected at t = 0:

            >>> import torch
            >>> from skfda.ml.generative import (
            ...     VarianceExplodingDiffusionProcess,
            ... )
            >>> ve = VarianceExplodingDiffusionProcess()
            >>> z = torch.ones(3, 8)
            >>> ve.multiply_sigma(z, torch.zeros(3)).abs().max().item() < 1e-6
            True
        """
        return torch.sqrt(self._cov(t)).unsqueeze(1) * z

    def multiply_cov(self, h: Tensor, t: Tensor) -> Tensor:
        r"""Scale h by :math:`\sigma_t^2 = \int_0^t g^2(s)\,ds`."""
        return self._cov(t).unsqueeze(1) * h

    def multiply_inv_sigma(self, h: Tensor, t: Tensor) -> Tensor:
        r"""Scale h by :math:`1/\sigma_t`, clamped for numerical stability."""
        inv_sigma = 1.0 / torch.sqrt(torch.clamp(self._cov(t), min=1e-6))
        return inv_sigma.unsqueeze(1) * h

    def sample_limit_distribution(
        self,
        n_samples: int,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        r"""Samples data from the final distribution of the diffusion process.

        For the variance-exploding diffusion process, the final distribution
        at time T is given by a normal distribution with mean 0 and covariance
        matrix given by the integral of g(s) from 0 to T.

        Args:
            n_samples: The number of samples to generate.
            grid_size: The number of discretization points of the
                functional data. Refers to the M dimension of the data.
            device: The device on which to create the samples.
                    Default is "cpu".

        Returns:
            X: Tensor with samples from the final distribution.
            Shape (n_samples, grid_size)
        """
        check_is_fitted(self, attributes=["M_", "generator_"])
        # Noise is sampled in CPU to ensure reproducibility across devices,
        # then moved to the target device
        noise = torch.randn(
            (n_samples, self.M_), device="cpu", generator=self.generator_,
        ).to(device, non_blocking=True)
        t_end = torch.full((n_samples,), self.T, device=device)
        return self.multiply_sigma(noise, t_end)

# Helper function for interpolation of the integrals of D(t) and F(t) in
# the DiagonalDiffusionProcess class.
def _batch_linear_interp_1d(
    t: Tensor, t_grid: Tensor, f_grid: Tensor,
) -> Tensor:
    """Vectorized 1D linear interpolation with stability handling.

    Args:
        t: query points, shape (N,) - MUST be within [t_grid[0], t_grid[-1]]
        t_grid: time coordinates of data points, shape (K,) - must be sorted
        f_grid: y-coordinates for M functions, shape (K, M)

    Returns:
        Interpolated values, shape (N, M)
    """
    if t.device != t_grid.device:
        msg = (
            f"Query times are on {t.device} but precomputed grids are on "
            f"{t_grid.device}. Fit and query must use the same device."
        )
        raise ValueError(
            msg,
        )
    if t_grid.is_contiguous() is False:
        msg = "t_grid must be a contiguous tensor."
        raise ValueError(msg)
    if t.is_contiguous() is False:
        msg = "t must be a contiguous tensor."
        raise ValueError(msg)
    indices = torch.searchsorted(t_grid, t, right=False)
    indices = torch.clamp(indices, 1, len(t_grid) - 1)
    t0 = t_grid[indices - 1]
    t1 = t_grid[indices]
    y0 = f_grid[indices - 1, :]
    y1 = f_grid[indices, :]

    dt = t1 - t0
    eps = 1e-10

    # Avoid division by zero: when dt is tiny, use midpoint value
    weight = torch.where(
        torch.abs(dt) > eps,
        (t - t0) / dt,
        torch.tensor(0.5, dtype=t.dtype, device=t.device),
    )
    # Linear interpolation: y = y0 + weight * (y1 - y0)
    return y0 + weight.unsqueeze(-1) * (y1 - y0)


def _validate_diagonal_callable(
    x: Tensor, fn: Callable[[Tensor], Tensor], name: str,
) -> None:
    """Validate that a diagonal SDE callable returns a shape-compatible tensor.

    Called inside fit() after self.M_ is set, once per callable (drift_term,
    diffusion_term). Probes the callable with a zero time tensor of the
    correct batch size and device, then checks shape and device of the output.

    Valid output shapes:
        (N, 1)    — single value per sample, broadcast to all M dimensions
        (N, M)    — per-dimension value per sample

    Args:
        x:    Training data tensor, shape (N, M). Used to extract N, M,
              device.
        fn:   Callable to validate. Signature: (t: Tensor) -> Tensor.
        name: Human-readable name for error messages (e.g. "drift_term",
        "diffusion_term").

    Raises:
        ValueError: If the output shape or device is incompatible with x.
        TypeError: If the output is not a torch.Tensor.
    """
    n, m = x.shape[0], x.shape[1]
    t_probe = torch.zeros(n, device=x.device, dtype=x.dtype)
    out = fn(t_probe)
    # ── type check ───────────────────────────────────────────────────────────
    if not isinstance(out, torch.Tensor):
        got = type(out).__name__
        msg = f"{name} must return a torch.Tensor, got {got}."
        raise TypeError(
            msg,
        )
    # ── shape check ──────────────────────────────────────────────────────────
    valid_shape = (
        (out.ndim == 2 and out.shape == (n, 1))   or # noqa: PLR2004 # Ignore magic value
        (out.ndim == 2 and out.shape == (n, m))      # noqa: PLR2004 # Ignore magic value
    )
    if not valid_shape:
        msg = (
            f"{name} returned shape {tuple(out.shape)}. "
            f"Expected one of: ({n}, 1), or ({n}, {m})."
        )
        raise ValueError(
            msg,
        )
    if out.dtype != x.dtype:
        msg = (
            f"{name} returned dtype {out.dtype}, but x has dtype {x.dtype}. "
            f"Both must have the same dtype."
        )
        raise ValueError(
            msg,
        )
    # ── device check ─────────────────────────────────────────────────────────
    if out.device != x.device:
        msg = (
            f"{name} returned a tensor on device {out.device}, "
            f"but x is on {x.device}. Both must be on the same device."
        )
        raise ValueError(
            msg,
        )


def _picklable_or_none(obj: object, name: str) -> object:
    """Return *obj* if picklable, otherwise warn and return None.

    Used by to_checkpoint() to produce a torch.save-compatible checkpoint
    even when a callable is not picklable. The None sentinel signals to
    from_checkpoint() that a pre-built instance must be supplied by the
    caller to restore those arguments.

    Args:
        obj:  The object to probe.
        name: Human-readable name for the warning message
        (e.g. ``"drift_term"``).

    Returns:
        *obj* unchanged if picklable, ``None`` otherwise.
    """
    try:
        pickle.dumps(obj)
    except Exception:  # noqa: BLE001
        warnings.warn(
            f"'{name}' is not picklable and will be stored as None. "
            "Supply a pre-built instance via the diff_process argument "
            "when loading this checkpoint.",
            UserWarning,
            stacklevel=3,
        )
        return None
    else:
        return obj

class DiagonalDiffusionProcess(ForwardDiffusionProcess):
    r"""Implements a diffusion process with diagonal drift and diffusion terms.

    Based on the following forward SDE:

    .. math::

        d\mathbf{X}(t) = \mathbf{D}(t)\mathbf{X}(t)\,dt
        + \mathbf{G}(t)\,d\mathbf{W}(t)

    where :math:`\mathbf{D}(t)` and :math:`\mathbf{G}(t)` are diagonal
    matrices whose entries may vary across both time and frequency components.

    Unlike the variance-preserving and variance-exploding processes, the
    mean and covariance of the conditional distribution are computed
    numerically. Integration is performed once on a fixed time grid at
    fit time and values are interpolated for arbitrary query times,
    amortizing the integration cost over the lifetime of the object.

    The time horizon :math:`T` is fixed to 1, as all expressions depend
    only on the ratio :math:`t/T`.
    """
    T: Final[int] = 1
    _PSEUDOINV_THRESHOLD: Final[float] = 1e-3

    def __init__(
            self,
            drift_term: Callable[[Tensor], Tensor],
            diffusion_term: Callable[[Tensor], Tensor],
            n_integration_points: int = 1000,
            seed: int | None = None,
    ) -> None:
        """Initializes the diagonal diffusion process.

        Args:
            drift_term: A function that takes the current time t and
                    returns the value of the drift term D(t) as a tensor,
                    shape (N, M) or (N,1)
            diffusion_term: A function that takes the current time t and
                    returns the value of the diffusion term G(t), shape
                    (N, M) or (N,1).
            n_integration_points: The number of points to use for numerical
                integration when computing the mean and covariance of the
                conditional distribution. Default is 1000.
            seed: The random seed for sampling from the limit distribution.
        """
        super().__init__(seed=seed)

        self.drift_term = drift_term
        self.diffusion_term = diffusion_term
        self.n_integration_points = n_integration_points

    def fit(self, x: Tensor) -> "DiagonalDiffusionProcess":
        """Fits the parameters of the diffusion process to the data.

        This method is used to learn the dimensionality of the data
        in order to sample from the limit distribution.

        Args:
            x: The input functional data as a tensor, shape (N, M)

        Returns:
            self: The fitted diffusion process.
        """
        super().fit(x)  # Learn the dimension of the data
        if self.n_integration_points < 2: # noqa: PLR2004 # Ignore magic value
            msg = (
                f"n_integration_points must be at least 2 (need at least"
                f" two grid points per interval); got"
                f" {self.n_integration_points}."
            )
            raise ValueError(
                    msg,
                )
        _validate_diagonal_callable(x, self.drift_term,  "drift_term")
        _validate_diagonal_callable(x, self.diffusion_term, "diffusion_term")
        self.device_ = x.device  # Learn the device of the data
        self._precompute_d_integral()
        self._precompute_f_integral()
        return self

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """Computes the drift term of the SDE at time t.

        For this process, the drift term is D(t) * x,
        where D(t) is a diagonal matrix.
        """
        return self.drift_term(t) * x

    def diffusion(self, t: Tensor) -> Tensor:
        """Computes the diffusion term of the SDE at time t.

        For this process, the diffusion term is g(t),
        where g(t) is a diagonal matrix.
        """
        return self.diffusion_term(t)

    def _precompute_d_integral(self) -> None:
        r"""Precompute :math:`\int_0^t D(u)\,du` on the integration grid.

        Stores the result in ``precomputed_d_grid`` (shape
        ``(n_integration_points, M)``) indexed by ``precomputed_d_grid__T``.
        """
        t = torch.linspace(0, self.T, self.n_integration_points,
                           device=self.device_)
        d_at_t = self.drift_term(t)
        if d_at_t.dim() == 1:
            d_at_t = d_at_t.unsqueeze(-1)
        if d_at_t.shape[-1] == 1:
            d_at_t = d_at_t.expand(-1, self.M_)

        d_int = torch.cumulative_trapezoid(d_at_t, t, dim=0)
        d_int = torch.cat(
            [d_int.new_zeros((1, d_at_t.shape[1])), d_int], dim=0,
        )
        self.precomputed_d_grid_T_ = t
        self.precomputed_d_grid_ = d_int

    def _integrate_d(self, t: Tensor) -> Tensor:
        r"""Return :math:`\int_0^t D(u)\,du` for each dimension, shape (N, M).

        Interpolates from the precomputed grid; assumes ``t`` is in
        :math:`(0, T]`.

        Args:
            t: Query times, shape (N,) or (N, 1), values in (0, T].

        Returns:
            :math:`\int_0^t D_i(u)\,du` for each sample and dimension,
            shape (N, M).
        """
        return _batch_linear_interp_1d(
                    t.reshape(-1),
                    self.precomputed_d_grid_T_,
                    self.precomputed_d_grid_,
        )

    def _precompute_f_integral(self) -> None:
        r"""Precompute :math:`f_i(t)` on the integration grid.

        For each dimension :math:`i`:

        .. math::

            f_i(t) = \int_0^t g_i(s)^2\,
            \exp\!\Bigl(-2\int_0^s D_i(u)\,du\Bigr)\,ds

        The result is stored in ``precomputed_sigma_f_grid`` (shape
        ``(n_integration_points, M)``), indexed by
        ``precomputed_sigma_t_grid``.
        """
        t = torch.linspace(0, self.T, self.n_integration_points,
                           device=self.device_)

        gt = self.diffusion_term(t)
        if isinstance(gt, (int, float)):
            gt = torch.full_like(t, gt)
        if gt.dim() == 1:
            gt = gt.unsqueeze(-1)
        if gt.shape[-1] == 1:
            gt = gt.expand(-1, self.M_)
        h = (gt**2) * torch.exp(-2 * self._integrate_d(t))
        f_cum = torch.cumulative_trapezoid(h, t, dim=0)
        f_cum = torch.cat([f_cum.new_zeros((1, f_cum.shape[1])), f_cum], dim=0)
        self.precomputed_sigma_t_grid_ = t
        self.precomputed_sigma_f_grid_ = f_cum

    def mean_cond(self, x: Tensor, t: Tensor) -> Tensor:
        r"""Computes the mean of the conditional distribution at time t.

        Because :math:`D(t)` is diagonal the matrix exponential reduces to
        an elementwise operation:

        .. math::

            \boldsymbol{\mu}_t =
            \exp\!\Bigl(\int_0^t D(u)\,du\Bigr) \odot \mathbf{x}

        Args:
            x: The data to condition on as a tensor, shape (N, M)
            t: The time steps as a tensor, shape (N,)

        Returns:
            The mean of the conditional distribution at time t, shape (N, M)
        """
        return torch.exp(self._integrate_d(t)) * x

    def _cov(self, t: Tensor) -> Tensor:
        r"""Returns the diagonal of :math:`\text{Cov}[X(t)]`, shape (N, M).

        Since the drift :math:`D(t)` and diffusion :math:`g(t)` are diagonal,
        the covariance is also diagonal. Denoting the :math:`i`-th diagonal
        entry of :math:`D(t)` as :math:`D_i(t)`, the covariance of
        :math:`X_i(t)` is:

        .. math::

            \text{Cov}[X_i(t)] = \exp\!\left(2\int_0^t D_i(u)\,du\right)
            \cdot F_i(t)

        where :math:`F_i(t) = \int_0^t g_i(s)^2
        \exp\!\left(-2\int_0^s D_i(u)\,du\right) ds`.

        Both integrals are precomputed on a fixed grid and interpolated for
        arbitrary query times.

        Args:
            t: The current time, shape (N,) or (N, 1).

        Returns:
            Diagonal of :math:`\text{Cov}[X(t)]`, shape (N, M).
        """
        f_t = _batch_linear_interp_1d(
            t.reshape(-1),
            self.precomputed_sigma_t_grid_,
            self.precomputed_sigma_f_grid_,
        )  # shape (N, M)
        exp_2dt = torch.exp(2 * self._integrate_d(t))  # shape (N, M)
        return (exp_2dt * f_t).clamp(min=0)  # shape (N, M)

    def multiply_sigma(self, z: Tensor, t: Tensor) -> Tensor:
        r"""Scale z by :math:`\sigma_t`, the per-dimension conditional std."""
        return torch.sqrt(self._cov(t)) * z

    def multiply_cov(self, h: Tensor, t: Tensor) -> Tensor:
        r"""Scale h by the diagonal covariance :math:`\mathrm{Cov}[X(t)]`."""
        return self._cov(t) * h

    def multiply_inv_sigma(self, h: Tensor, t: Tensor) -> Tensor:
        r"""Scale h by :math:`1/\sigma_t` using a pseudoinverse.

        Directions where :math:`\sqrt{\mathrm{Cov}_i} < 10^{-3}` get zero
        weight instead of a large inverse. This preserves
        :math:`\sigma \cdot \sigma^{-1} = I` in directions with actual noise
        injection and correctly ignores directions with no noise (e.g.
        positive-eigenvalue modes under VP clamping).
        """
        sigma = torch.sqrt(self._cov(t))  # shape (N, M)
        mask = sigma > self._PSEUDOINV_THRESHOLD
        inv_sigma = torch.zeros_like(sigma)
        inv_sigma[mask] = 1.0 / sigma[mask]
        return inv_sigma * h

    def _get_fit_state(self) -> dict[str, FitStateType]:
        """Return the fitted attributes as a serializable dictionary.

        Extends the base state with ``device``, which is learned from the
        training data in :meth:`fit`. The precomputed integral grids are
        intentionally excluded: they are derived quantities and are
        recomputed from ``drift_term``, ``diffusion_term``, ``M``, and
        ``device`` in :meth:`_restore_fit_state`, keeping the stored
        state minimal.

        Returns:
            dict: Base state plus ``{"device": self.device_}``.

        Raises:
            sklearn.exceptions.NotFittedError: If :meth:`fit` has not been
                called yet.
        """
        state = super()._get_fit_state()
        state["device"] = self.device_
        return state


    def _restore_fit_state(self, data: dict[str, FitStateType]) -> None:
        """Restore fitted attributes and rebuild all precomputed grids.

        Calls the base implementation to restore ``M``, then restores
        ``device`` and recomputes the integral grids via
        :meth:`_precompute_d_integral` and :meth:`_precompute_f_integral`.
        ``n_integration_points`` is already set by ``__init__`` (via
        ``init_kwargs``) before this method is called, so it does not
        appear in *data*.

        Args:
            data: A dictionary as produced by :meth:`_get_fit_state`.

        Raises:
            ValueError: If ``"device"`` is absent from *data* or is not a
                ``torch.device`` with type ``"cpu"`` or ``"cuda"``.
        """
        if self.drift_term is None or self.diffusion_term is None:
            msg = (
                "Cannot restore fit state: drift_term and/or diffusion_term "
                "are None. The checkpoint was saved with non-picklable "
                "callables. Supply a pre-built DiagonalDiffusionProcess "
                "with the correct callables via the diff_process argument "
                "when loading."
            )
            raise ValueError(
                msg,
            )

        super()._restore_fit_state(data)
        if "device" not in data:
            msg = (
                "'device' is missing from the fit state dictionary. "
                "Ensure the checkpoint was produced by a compatible process."
            )
            raise ValueError(
                msg,
            )
        if (
            not isinstance(data["device"], torch.device)
            or data["device"].type not in ("cpu", "cuda")
        ):
            msg = (
                "Expected 'device' to be a torch.device with type 'cpu' or "
                f"'cuda', got {data['device']!r}."
            )
            raise ValueError(
                msg,
            )

        self.device_ = data["device"]
        self._precompute_d_integral()
        self._precompute_f_integral()

    def to_checkpoint(self) -> CheckpointDict:
        """Serialize the full process state, validating callables first.

        Extends :meth:`ForwardDiffusionProcess.to_checkpoint` by checking
        that ``drift_term`` and ``diffusion_term`` are picklable before
        attempting serialization. If either is a lambda or a local closure,
        a :exc:`ValueError` is raised immediately with a clear description
        of the fix required.

        Returns:
            dict: Checkpoint dictionary as described in the base class.

        Raises:
            ValueError: If ``drift_term`` or ``diffusion_term``
            is not picklable.
            sklearn.exceptions.NotFittedError: If :meth:`fit` has not been
                called yet (propagated from :meth:`_get_fit_state`).
        """
        checkpoint = super().to_checkpoint()
        checkpoint["init_kwargs"]["drift_term"] = (
            _picklable_or_none(self.drift_term, "drift_term")
        )
        checkpoint["init_kwargs"]["diffusion_term"] = (
            _picklable_or_none(self.diffusion_term, "diffusion_term")
        )
        return checkpoint

    def sample_limit_distribution(
        self,
        n_samples: int,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        r"""Samples data from the final distribution of the diffusion process.

        The limit distribution has mean zero and covariance given by
        :math:`\text{Cov}[X(T)]` evaluated at ``T``.

        Args:
            n_samples: The number of samples to generate.
            device: The device on which to create the samples.
                Default is "cpu".

        Returns:
            Tensor of shape (n_samples, M) sampled from the limit distribution.

        Raises:
            sklearn.exceptions.NotFittedError: If :meth:`fit` has not been
                called yet.
        """
        check_is_fitted(self, attributes=["M_", "generator_"])
        # Noise is sampled always on CPU to allow reproducibility cross-device.
        noise = torch.randn(
            (n_samples, self.M_), device="cpu", generator=self.generator_,
        ).to(device, non_blocking=True)
        t_end = torch.full((n_samples,), self.T, device=device)
        return self.multiply_sigma(noise, t_end)


def _duplicate_symmetric_eigenvalues(
    lambda_half: Callable[[Tensor], Tensor],
    m: int,
    device: torch.device | str = "cpu",
) -> Callable[[Tensor], Tensor]:
    r"""Wraps a half-eigenvalue callable into a full eigenvalue callable.

    When the drift or diffusion term is specified directly in the Fourier
    domain (``fourier_drift=True`` or ``fourier_diffusion=True``), only the
    :math:`M/2 + 1` unique eigenvalues need to be provided. The remaining
    eigenvalues are recovered by symmetric reflection, mirroring the
    structure of the cosine basis:

    .. math::

        \boldsymbol{\lambda} = [
            \lambda_0,\ \lambda_1,\ \ldots,\ \lambda_{M/2},\
            \lambda_{M/2-1},\ \ldots,\ \lambda_1
        ]

    This is the Fourier-domain analogue of
    :func:`_duplicate_symmetric_row_and_get_eigenvalues`, used when
    eigenvalues are supplied directly rather than derived from circulant
    row entries.

    Args:
        lambda_half: Callable that takes time steps of shape ``(N,)`` and
            returns the :math:`M/2 + 1` unique eigenvalues, shape
            ``(N, M/2+1)``.
        m: Size of the circulant matrix.
        device: Device on which to run computations. Default is ``'cpu'``.

    Returns:
        A callable that takes time steps of shape ``(N,)`` and returns
        the full :math:`M` eigenvalues after symmetric reflection,
        shape ``(N, M)``.
    """
    def lambdas(t: Tensor) -> Tensor:
        n = t.shape[0]
        lambda_half_val = lambda_half(t)

        lambdas_val = torch.zeros(
            (n, m), device=device, dtype=lambda_half_val.dtype,
        )
        half = (
            lambda_half_val if lambda_half_val.ndim == 2 # noqa: PLR2004 # Ignore magic value
            else lambda_half_val.unsqueeze(-1)
        )
        lambdas_val[:, :m//2 + 1] = half

        # Fixed symmetry slicing for Odd/Even M
        lambdas_val[:, m//2 + 1:] = torch.flip(
            lambdas_val[:, 1:(m + 1)//2], dims=[1],
        )

        return lambdas_val
    return lambdas

def _duplicate_symmetric_row_and_get_eigenvalues(
    c_half: Callable[[Tensor], Tensor],
    m: int,
    device: torch.device | str = "cpu",
) -> Callable[[Tensor], Tensor]:
    r"""Wraps a half-row callable into a full eigenvalue callable.

    A circulant symmetric matrix of size :math:`M` is fully determined by its
    :math:`M/2 + 1` unique entries :math:`[c_0, c_1, \ldots, c_{M/2}]`.
    The full first row is recovered by symmetric reflection:

    .. math::

        \mathbf{c} = [c_0,\ c_1,\ \ldots,\ c_{M/2},\ c_{N/2-1},\ \ldots,\ c_1]

    The eigenvalues of the circulant matrix are then the real DFT of
    :math:`\mathbf{c}`, which coincides with the projection onto the cosine
    basis :math:`\mathbf{Q}` from :func:`_get_cosine_basis`:

    .. math::

        \boldsymbol{\lambda}(t) = \mathbf{Q}^\top \mathbf{c}(t)

    This function wraps ``c_half`` so that the returned callable accepts a
    batch of time steps and returns the corresponding eigenvalues, avoiding
    the need to reconstruct and diagonalize the full matrix at every call.

    Args:
        c_half: Callable that takes time steps of shape ``(N,)`` and returns
            the :math:`M/2 + 1` unique first-row entries, shape ``(N, M/2+1)``.
        m: Size of the circulant matrix.
        device: Device on which to run computations. Default is ``'cpu'``.

    Returns:
        A callable that takes time steps of shape ``(N,)`` and returns the
        :math:`M` real eigenvalues of the circulant matrix, shape ``(N, M)``.
    """
    def lambdas(t: Tensor) -> Tensor:
        n = t.shape[0]
        c_val = c_half(t)

        row = torch.zeros((n, m), device=device, dtype=c_val.dtype)
        row[:, :m//2 + 1] = c_val if c_val.ndim == 2 else c_val.unsqueeze(-1) # noqa: PLR2004 # Ignore magic value
        row[:, m//2 + 1:] = torch.flip(row[:, 1:(m + 1)//2], dims=[1])

        results: Tensor = torch.fft.fft(row).real
        return results
    return lambdas

def _get_cosine_basis(
    m: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    r"""Returns real orthogonal Fourier basis for circulant symmetric matrices.

    Constructs the :math:`M \times M` orthogonal matrix :math:`\mathbf{Q}`
    whose columns diagonalize any circulant symmetric matrix of size :math:`M`.
    The columns are ordered by frequency:

    .. math::

        Q_{jk} = \begin{cases}
            \dfrac{1}{\sqrt{M}}
                & k = 0 \text{ (DC)} \\[6pt]
            \sqrt{\dfrac{2}{M}}\cos\!\left(\dfrac{2\pi j k}{M}\right)
                & k = 1, \ldots, \lfloor(M-1)/2\rfloor \\[6pt]
            \dfrac{(-1)^j}{\sqrt{M}}
                & k = M/2,\ M \text{ even (Nyquist)}
        \end{cases}

    so that for any circulant symmetric matrix :math:`\mathbf{C}`,

    .. math::

        \mathbf{C} = \mathbf{Q}\,\boldsymbol{\Lambda}\,\mathbf{Q}^\top

    where :math:`\boldsymbol{\Lambda}` is diagonal.

    Args:
        m: Size of the matrix.
        device: Device on which to create the matrix. Default is ``'cpu'``.
        dtype: Dtype of the output tensor. Default is ``torch.float32``.

    Returns:
        q_mat: Real orthogonal Fourier basis, shape ``(m, m)``.
    """
    with torch.no_grad():
        # Ensure m is an integer tensor for calculations
        m_tensor = torch.tensor(m, device=device, dtype=dtype)
        indices = torch.arange(m, device=device, dtype=dtype)

        q_mat = torch.zeros((m, m), device=device, dtype=dtype)

        # --- 1. DC Component (k=0) ---
        # Corresponds to lambda[0]
        q_mat[:, 0] = 1.0 / torch.sqrt(m_tensor)

        # --- 2. Harmonic Frequencies (Pairs) ---
        # We iterate k from 1 up to (m-1)//2.
        # This generates matching pairs at columns k and m-k.
        # Example m=8: k=1, 2, 3. (Cols 1&7, 2&6, 3&5 filled).
        # Example m=9: k=1, 2, 3, 4. (Cols 1&8, 2&7, 3&6, 4&5 filled).

        limit = (m - 1) // 2
        if limit > 0:
            k_vals = torch.arange(1, limit + 1, device=device, dtype=dtype)
            angles = (
                2 * torch.pi * indices.unsqueeze(1) * k_vals.unsqueeze(0)
            ) / m_tensor
            # Flip so col m-1 carries k=1, col m-2 carries k=2, etc.
            flip_angles = torch.flip(angles, dims=[1])

            normalization = torch.sqrt(2.0 / m_tensor)

            # Fill the 'left' side (k) with Cosines
            q_mat[:, 1:limit+1] = normalization * torch.cos(angles)

            # Fill the 'right' side (m-k) with Sines
            # We map k=1 to col m-1, k=2 to col m-2, etc.
            q_mat[:, m - limit:] = normalization * torch.sin(flip_angles)

        # --- 3. Nyquist Component (k=m/2) ---
        # Only exists if m is even. Corresponds to lambda[m/2].
        if m % 2 == 0:
            # This is the alternating vector [1, -1, 1, -1...]
            q_mat[:, m // 2] = (
                torch.cos(torch.pi * indices) / torch.sqrt(m_tensor)
                )
    return q_mat

def _fft_to_eigenspace(x: Tensor) -> Tensor:
    r"""Project x to the circulant eigenspace (x @ _get_cosine_basis(m)).

    Replaces the :math:`O(NM^2)` matrix multiplication with an
    :math:`O(NM\log M)` FFT. The cosine basis ``Q`` is the normalized
    real DFT matrix, so the projection equals: rfft → separate Re/Im
    → normalize → repack into ``Q``'s column order.

    Args:
        x: Input tensor, shape (..., M).

    Returns:
        Eigenspace coordinates, shape (..., M).
    """
    m = x.shape[-1]
    X = torch.fft.rfft(x, n=m)  # (..., m//2+1) complex
    inv_sqrt_m = m ** -0.5
    sqrt_2_over_m = (2.0 / m) ** 0.5
    limit = (m - 1) // 2

    out = torch.empty_like(x)
    out[..., 0] = X[..., 0].real * inv_sqrt_m

    if limit > 0:
        harmonics = X[..., 1:limit + 1]  # FFT[1..limit], shape (..., limit)
        out[..., 1:limit + 1] = harmonics.real * sqrt_2_over_m
        # Columns m-limit..m-1 correspond to k=limit..1 (reversed)
        out[..., m - limit:m] = (
            -torch.flip(harmonics, dims=[-1]).imag * sqrt_2_over_m
        )

    if m % 2 == 0:
        out[..., m // 2] = X[..., m // 2].real * inv_sqrt_m

    return out

def _eigenspace_to_fft(y: Tensor) -> Tensor:
    r"""Project from circulant eigenspace back to original space (y @ Q.T).

    Replaces the :math:`O(NM^2)` matrix multiplication with an
    :math:`O(NM\log M)` IFFT.

    Args:
        y: Eigenspace coordinates, shape (..., M).

    Returns:
        Original-space tensor, shape (..., M).
    """
    m = y.shape[-1]
    sqrt_m = float(m) ** 0.5
    sqrt_m_over_2 = (m / 2.0) ** 0.5
    limit = (m - 1) // 2
    n_freqs = m // 2 + 1

    ctype = torch.complex64 if y.dtype == torch.float32 else torch.complex128
    fft_coeffs = torch.zeros(
        *y.shape[:-1], n_freqs, dtype=ctype, device=y.device,
    )

    fft_coeffs[..., 0] = y[..., 0] * sqrt_m

    if limit > 0:
        re_part = y[..., 1:limit + 1] * sqrt_m_over_2
        # Im(FFT[k]) = -y[m-k]*sqrt(m/2); sin cols m-limit..m-1 → k=limit..1
        sin_cols = y[..., m - limit:m]  # shape (..., limit), k=limit..1
        # flip → k=1..limit
        im_part = -torch.flip(sin_cols, dims=[-1]) * sqrt_m_over_2
        fft_coeffs[..., 1:limit + 1] = torch.view_as_complex(
            torch.stack([re_part, im_part], dim=-1).contiguous(),
        )

    if m % 2 == 0:
        fft_coeffs[..., m // 2] = y[..., m // 2] * sqrt_m

    results: Tensor = torch.fft.irfft(fft_coeffs, n=m)
    return results

class CirculantSymmetricMatrixDiffusionProcess(ForwardDiffusionProcess):
    r"""Implements a diffusion process with circulant symmetric matrices.

    Based on the following forward SDE:

    .. math::

        d\mathbf{X}(t) = \mathrm{D}(t)\mathbf{X}(t)\,dt
        + \mathrm{G}(t)\,d\mathbf{W}(t)

    where :math:`\mathrm{D}(t)` and :math:`\mathrm{G}(t)` are circulant
    symmetric matrices parameterized by their :math:`M/2 + 1` unique
    coefficients. For :math:`M = 6`, the structure is:

    .. math::

        \mathrm{C}(t) = \begin{bmatrix}
            c_0 & c_1 & c_2 & c_3 & c_2 & c_1 \\
            c_1 & c_0 & c_1 & c_2 & c_3 & c_2 \\
            c_2 & c_1 & c_0 & c_1 & c_2 & c_3 \\
            c_3 & c_2 & c_1 & c_0 & c_1 & c_2 \\
            c_2 & c_3 & c_2 & c_1 & c_0 & c_1 \\
            c_1 & c_2 & c_3 & c_2 & c_1 & c_0
        \end{bmatrix}

    Circulant symmetric matrices are diagonalized by the real orthogonal
    Fourier basis, so all computations are carried out in the eigenspace
    where drift and diffusion act diagonally on frequency components.
    This makes the process equivalent to a :class:`DiagonalDiffusionProcess`
    in the Fourier domain, with the additional structure that the eigenvalues
    are derived from the circulant coefficients.

    As with :class:`DiagonalDiffusionProcess`, the mean and covariance of
    the conditional distribution are computed numerically on a fixed time
    grid at fit time and interpolated for arbitrary query times.

    The time horizon :math:`T` is fixed to 1, as all expressions depend
    only on the ratio :math:`t/T`.
    """
    T: Final[int] = 1

    def __init__(
            self,
            drift_term: Callable[[Tensor], Tensor],
            diffusion_term: Callable[[Tensor], Tensor],
            *,
            fourier_drift: bool = False,
            fourier_diffusion: bool = False,
            n_integration_points: int = 1000,
            seed: int | None = None,
    ) -> None:
        """Initializes the circulant symmetric matrix diffusion process.

        Args:
            drift_term: A function that takes the current time t and returns
                the value of M/2 + 1 coefficients needed to build the circulant
                symmetric matrix. Depending on the value of fourier drift, the
                coefficients are either the eigenvalues of the drift term in
                the Fourier basis (if fourier_drift is True) or the first
                M/2 + 1 elements of the first row of the circulant matrix
                (if fourier_drift is False).
            diffusion_term: A function returning M/2+1 coefficients for the
                circulant diffusion matrix. If ``fourier_diffusion`` is True,
                the coefficients are eigenvalues; otherwise they are the first
                M/2+1 elements of the first row. Default is False.
            fourier_drift: If True, ``drift_term`` returns eigenvalues in the
                Fourier basis; otherwise it returns the first M/2+1 row
                elements. Default is False.
            fourier_diffusion: If True, ``diffusion_term`` returns eigenvalues
                in the Fourier basis; otherwise the first M/2+1 row elements.
                Default is False.
            n_integration_points: The number of points to use for numerical
                integration when computing the mean and covariance of the
                conditional distribution. Default is 1000.
            seed: The random seed for sampling from the limit distribution.
        """
        super().__init__()
        self.drift_term = drift_term
        self.diffusion_term = diffusion_term
        self.fourier_drift = fourier_drift
        self.fourier_diffusion = fourier_diffusion
        self.n_integration_points = n_integration_points
        self.seed = seed

    def _build_diagonal_process(self) -> DiagonalDiffusionProcess:
        """Builds the underlying DiagonalDiffusionProcess in the Fourier basis.

        This is called during fitting after the eigenvalues have been computed
        from the provided drift and diffusion terms. The diagonal process will
        be fitted on the data projected to the Fourier basis, ensuring that all
        computations of mean, covariance, and sampling are consistent with the
        circulant structure.

        Returns:
            An instance of DiagonalDiffusionProcess in the Fourier basis.
        """
        if self.fourier_drift:
            self.lambdas_ = _duplicate_symmetric_eigenvalues(
                lambda_half=self.drift_term,
                m=self.M_,
                device=self.device_,
            )
        else:
             self.lambdas_ = _duplicate_symmetric_row_and_get_eigenvalues(
                c_half=self.drift_term,
                m=self.M_,
                device=self.device_,
            )

        if self.fourier_diffusion:
            self.diag_gt_ = _duplicate_symmetric_eigenvalues(
                lambda_half=self.diffusion_term,
                m=self.M_,
                device=self.device_,
            )
        else:
            self.diag_gt_ = _duplicate_symmetric_row_and_get_eigenvalues(
                c_half=self.diffusion_term,
                m=self.M_,
                device=self.device_,
            )

        return DiagonalDiffusionProcess(
            drift_term=self.lambdas_,
            diffusion_term=self.diag_gt_,
            n_integration_points=self.n_integration_points,
        )


    def fit(self, x: Tensor) -> "CirculantSymmetricMatrixDiffusionProcess":
        """Fits the parameters of the diffusion process to the data.

        This method is used to learn the dimensionality of the data
        in order to sample from the limit distribution and
        to compute the matrix B(t) of the drift term.

        Args:
            x: The input functional data as a tensor, shape (N, M)

        Returns:
            self: The fitted diffusion process.
        """
        super().fit(x)  # Learn the dimension of the data
        self.device_ = x.device  # Learn the device of the data
        self.diagonal_process_ = self._build_diagonal_process()

        # Fit in the eigenbasis
        self.diagonal_process_.fit(_fft_to_eigenspace(x))

        # Precompute the cosine basis matrix
        self.q_mat_ = _get_cosine_basis(self.M_, self.device_)

        return self

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        r"""Computes the drift term of the SDE at time t.

        Computes :math:`B(t)\,x = Q\,\operatorname{diag}(\lambda(t))\,Q^\top x`
        via FFT in :math:`O(NM\log M)`.
        """
        check_is_fitted(self, attributes=["lambdas_"])
        lambda_t = self.lambdas_(t)  # shape (N, M)
        return _eigenspace_to_fft(lambda_t * _fft_to_eigenspace(x))

    def diffusion(self, t: Tensor) -> Tensor:
        r"""Computes the diffusion term of the SDE at time t.

        Returns the circulant matrix
        :math:`G(t) = Q\,\operatorname{diag}(g(t))\,Q^\top` of shape
        ``(N, M, M)``. For efficient matrix-vector products use
        ``diffusion_times_v``, which exploits the circulant structure via
        FFT in :math:`O(NM\log M)`.

        Args:
            t: The time steps as a tensor, shape (N,)

        Returns:
            Circulant diffusion matrix, shape (N, M, M)
        """
        check_is_fitted(self, attributes=["diag_gt_"])

        diffusion_term = self.diagonal_process_.diffusion(t)   # (N, M)
        g_diag = torch.diag_embed(diffusion_term)              # (N, M, M)
        return self.q_mat_ @ g_diag @ self.q_mat_.T   # (N, M, M)

    def diffusion_times_v(self, v: Tensor, t: Tensor) -> Tensor:
        r"""Compute :math:`G(t)\,v` via FFT in :math:`O(NM\log M)`.

        Equivalent to
        :math:`Q\,\operatorname{diag}(g(t))\,Q^\top v` without forming
        the full matrix.

        Args:
            v: The vector to multiply, shape (N, M)
            t: The time steps as a tensor, shape (N,)

        Returns:
            The result of G(t) v, shape (N, M)
        """
        diffusion_term = self.diagonal_process_.diffusion(t)  # shape (N, M)
        return _eigenspace_to_fft(diffusion_term * _fft_to_eigenspace(v))

    def diffusion_gram_times_v(self, v: Tensor, t: Tensor) -> Tensor:
        r"""Compute :math:`G(t)\,G(t)^\top v` via FFT in :math:`O(NM\log M)`.

        :math:`G` is symmetric for circulant processes, so
        :math:`G\,G^\top = G^2`. Eigenvalues are squared directly,
        requiring only two FFTs.

        Args:
            v: The vector to multiply, shape (N, M)
            t: The time steps as a tensor, shape (N,)

        Returns:
            :math:`G(t)\,G(t)^\top v`, shape (N, M)
        """
        diffusion_term = self.diagonal_process_.diffusion(t)  # shape (N, M)
        return _eigenspace_to_fft(diffusion_term ** 2 * _fft_to_eigenspace(v))

    def mean_cond(self, x: Tensor, t: Tensor) -> Tensor:
        r"""Computes the mean of the conditional distribution at time t.

        Using the diagonalisation :math:`Q^\top x = y`:

        .. math::

            \mathbb{E}[X(t) \mid X(0)=x]
            = Q\,\mathbb{E}[Y(t) \mid Y(0)=Q^\top x]

        computed via FFT in :math:`O(NM\log M)`.

        Args:
            x: The initial condition X_0, shape (N, M)
            t: The current time, shape (N,)

        Returns:
            mean_t: The mean of the conditional distribution, shape (N, M).
        """
        y = _fft_to_eigenspace(x)
        mean_eigen = self.diagonal_process_.mean_cond(y, t)  # shape (N, M)
        return _eigenspace_to_fft(mean_eigen)

    def _cov(self, t: Tensor) -> Tensor:
        r"""Returns the full covariance matrix.

        :math:`Q\,\text{diag}(c_Y(t))\,Q^\top`

        Private introspection helper for visualization; hot paths
        should call ``multiply_sigma``, ``multiply_cov``, or
        ``multiply_inv_sigma``.

        Args:
            t: The current time, shape (N,) or (N, 1).

        Returns:
            Covariance matrix, shape (N, M, M).
        """
        cov_y_t = self.diagonal_process_._cov(t)  # noqa: SLF001 # Ignore private method access
        cov_y_diag = torch.diag_embed(cov_y_t)
        return self.q_mat_ @ cov_y_diag @ self.q_mat_.T

    # --- Fast operator overrides via FFT ---

    def multiply_sigma(self, z: Tensor, t: Tensor) -> Tensor:
        r"""See base class. Uses FFT for :math:`O(NM\log M)` complexity."""
        return _eigenspace_to_fft(
            self.diagonal_process_.multiply_sigma(_fft_to_eigenspace(z), t),
        )

    def multiply_cov(self, h: Tensor, t: Tensor) -> Tensor:
        r"""See base class. Uses FFT for :math:`O(NM\log M)` complexity."""
        return _eigenspace_to_fft(
            self.diagonal_process_.multiply_cov(_fft_to_eigenspace(h), t),
        )

    def multiply_inv_sigma(self, h: Tensor, t: Tensor) -> Tensor:
        r"""See base class. Uses FFT for :math:`O(NM\log M)` complexity."""
        return _eigenspace_to_fft(
            self.diagonal_process_.multiply_inv_sigma(
                _fft_to_eigenspace(h),
                t,
            ),
        )

    def _get_fit_state(self) -> dict[str, FitStateType]:
        """Return the fitted attributes as a serializable dictionary.

        Extends the base state with ``device``. All other fitted attributes
        (``lambdas``, ``diag_gt``, ``diagonal_process``, ``q_mat``) are
        derived quantities and are rebuilt from the init kwargs plus ``M``
        and ``device`` in :meth:`_restore_fit_state`, so they are not stored.

        Returns:
            dict: Base state plus ``{"device": self.device_}``.

        Raises:
            sklearn.exceptions.NotFittedError: If :meth:`fit` has not been
                called yet.
        """
        state = super()._get_fit_state()
        state["device"] = self.device_
        return state


    def _restore_fit_state(self, data: dict[str, FitStateType]) -> None:
        """Restore fitted attributes and rebuild all derived components.

        Calls the base implementation to restore ``M``, then restores
        ``device`` and delegates to :meth:`fit` with a dummy tensor of the
        correct shape. This rebuilds ``lambdas``, ``diag_gt``,
        ``diagonal_process``, and ``q_mat`` from the callables already
        present in the instance (set by ``__init__`` via ``init_kwargs``
        before this method is called).

        Args:
            data: A dictionary as produced by :meth:`_get_fit_state`.

        Raises:
            ValueError: If ``"device"`` is absent or has the wrong type.
        """
        if self.drift_term is None or self.diffusion_term is None:
            msg = (
                "Cannot restore fit state: drift_term and/or diffusion_term"
                " are None. The checkpoint was saved with non-picklable"
                " callables. Supply a pre-built"
                " CirculantSymmetricMatrixDiffusionProcess with the correct"
                " callables via the diff_process argument when loading."
            )
            raise ValueError(
                msg,
            )
        super()._restore_fit_state(data)
        if "device" not in data:
            msg = (
                "'device' is missing from the fit state dictionary. "
                "Ensure the checkpoint was produced by a compatible process."
            )
            raise ValueError(
                msg,
            )
        if (
            not isinstance(data["device"], torch.device)
            or data["device"].type not in ("cpu", "cuda")
        ):
            msg = (
                "Expected 'device' to be a torch.device with type 'cpu' or "
                f"'cuda', got {data['device']!r}."
            )
            raise ValueError(
                msg,
            )
        self.device_ = data["device"]
        # All derived attributes depend only on the callables (already
        # restored via init_kwargs), M, and device. A dummy zero tensor
        # is sufficient — fit() does not use the actual data values here.
        x_toy = torch.zeros((1, self.M_), device=self.device_)
        self.diagonal_process_ = self._build_diagonal_process()
        self.diagonal_process_.fit(_fft_to_eigenspace(x_toy))

        self.q_mat_ = _get_cosine_basis(self.M_, self.device_)



    def to_checkpoint(self) -> CheckpointDict:
        """Serialize the full process state, validating callables first.

        Extends :meth:`ForwardDiffusionProcess.to_checkpoint` by checking
        that ``drift_term`` and ``diffusion_term`` are picklable before
        attempting serialization. If either is a lambda or local closure, a
        :exc:`ValueError` is raised with a clear description of the fix.

        Returns:
            dict: Checkpoint dictionary as described in the base class.

        Raises:
            ValueError: If ``drift_term`` or ``diffusion_term`` is not
                picklable.
            sklearn.exceptions.NotFittedError: If :meth:`fit` has not been
                called yet (propagated from :meth:`_get_fit_state`).
        """
        checkpoint = super().to_checkpoint()
        checkpoint["init_kwargs"]["drift_term"] = (
            _picklable_or_none(self.drift_term, "drift_term")
        )
        checkpoint["init_kwargs"]["diffusion_term"] = (
            _picklable_or_none(self.diffusion_term, "diffusion_term")
        )
        return checkpoint

    def sample_limit_distribution(
        self,
        n_samples: int,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        r"""Samples data from the final distribution of the diffusion process.

        Samples :math:`Y \sim \mathcal{N}(0,\,\text{diag}(c_Y(T)))` in
        eigenspace then maps back via IFFT.

        Args:
            n_samples: The number of samples to generate.
            device: The device on which to create the samples.
                    Default is "cpu".

        Returns:
            Tensor of shape (n_samples, M) sampled from the limit distribution.

        Raises:
            sklearn.exceptions.NotFittedError: If :meth:`fit` has not been
                called yet.
        """
        check_is_fitted(self, attributes=["M_", "generator_"])

        # Noise is generated always on CPU to ensure consistent sampling
        # regardless of the device of the process.
        noise = torch.randn(
            (n_samples, self.M_), device="cpu", generator=self.generator_,
        ).to(device, non_blocking=True)
        t_end = torch.full((n_samples,), self.T, device=device)
        return self.multiply_sigma(noise, t_end)

