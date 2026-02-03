from typing import Callable, Protocol

from ...representation import FData

import torch
from torch import Tensor

from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import partial

@dataclass
class DiffusionState:
    """Class to hold the state of a diffusion process at a given time step.

    Attributes:
        t: The current time step as a tensor, shape (N,)
        x: The current functional data as a tensor, shape (N, M)
        mu_t: The mean of the diffusion process at time t, shape (N, M)
        sigma_t: The square root of the covariance matrix of the diffusion
                  process at time t, shape (N, M, M)
        inv_sigma_t: The inverse of the square root of the covariance matrix
                     of the diffusion process at time t, shape (N, M, M)
    """
    t: Tensor
    x: Tensor
    mu_t: Tensor
    sigma_t: Tensor
    inv_sigma_t: Tensor

class DiffusionProcess(Protocol):
    """Common interface of a diffusion process to be used in generative models.

    It defines the necessary methods to be used in the FunDiffusion class to
    generate synthetic functional data.
    """
    T: int # Final time of the diffusion process

    # TODO(): Decide whether we want to forward from t_0 != 0
    def forward(self, x: Tensor, t: Tensor) -> DiffusionState:
        """Applies the forward diffusion process to the input data.

        Args:
            x: The input functional data as a tensor, shape (N, M)
            t: The time steps at which to apply the diffusion, shape (N,)

        Returns:
            :class:`DiffusionState` with the diffusion state at time t.
        """
        ...

    # TODO(): Decide what to do with inv_sigma_t for score_model
    # For training it can be done using DiffusionState but for sampling
    # we don't call the forward method so we need to compute it somehow else
    def inv_sigma_t(self, t: Tensor) -> Tensor:
        """Computes the inverse of a sqrt of the covariance matrix at time t.

        It can have shape (N, M, M), (N,M) or (N,).

        If (N, M, M) is returned, it represents the full inverse covariance
        matrix.

        If (N, M) is returned, it represents only the values of the diagonal
        of the matrix. Hence the behavior is the same as if a diagonal matrix
        is used.

        If (N,) is returned, it represents a scalar multiple of the identity
        matrix. Hence the behavior is the same as if a scalar multiple of the
        identity matrix is used.

        Args:
            t: The time steps as a tensor, shape (N,)

        Returns:
            inv_sigma_t: The inverse of the square root of the covariance
                         matrix at time t, shape (N, M, M) or (N, M) or (N,).
        """
        ...


    # TODO(): Decide whether the labels should be generated, passed or neither
    def sample_final_distribution(self, n_samples: int) -> Tensor:
        """Samples data from the final distribution of the diffusion process.

        Args:
            n_samples: The number of samples to generate.

        Returns:
            X: Tensor with samples from the final distribution.
        """
        ...

    # TODO(): Decide whether to return DiffusionState or just x at t_0
    def backward(self,
                 score_model: ScoreNet,
                 x: Tensor,
                 t: Tensor | float,
                 t_0: Tensor | float = 0.0,
                 y: Tensor | None = None) -> DiffusionState:
        """Applies the backward diffusion process to the perturbed data.

        Args:
            score_model: The score network used to estimate the score
                         function of the diffusion process.
            x: The perturbed data at time t as a tensor. Shape (N, M)
            t: The time at which the data is perturbed as a tensor with shape
               (N,) or a float. In the latter case, the behavior is the same as
                if a tensor with shape (N,) with all entries equal to t is
                passed.
            t_0: The time till which we want to reverse the diffusion process
                as a tensor with shape (N,) or a float. In the latter case,
                the behavior is the same as if a tensor with shape (N,)
                with all entries equal to t_0 is passed.
            y: Optional tensor with the class labels of the data, shape (N,)
               Default is `None`.
        Returns:
            :class:`DiffusionState` with the diffusion state at time t_0.
        """
        ...


class EulerMaruyamaDiffusionProcess(ABC):
    """Abstract base class for diffusion processes using Euler-Maruyama.

    This class defines the interface for diffusion processes that use the
    Euler-Maruyama method for simulating the backward diffusion process.
    It requires subclasses to implement the drift and diffusion coefficient.
    """
    T: int # Final time of the diffusion process
    # TODO(): Decide whether this is a class attribute or an argument in backward
    integration_steps: int # Number of integration steps for Euler-Maruyama

    # TODO(): Do I need to write here forward and sample from final distribution again?
    @abstractmethod
    def forward(self, x: Tensor, t: Tensor) -> DiffusionState:
        """Applies the forward diffusion process using Euler-Maruyama.

        Args:
            x: The input functional data as a tensor, shape (N, M)
            t: The time steps at which to apply the diffusion, shape (N,)

        Returns:
            :class:`DiffusionState` with the diffusion state at time t.
        """
        ...
    @abstractmethod
    def sample_final_distribution(self, n_samples: int, y: Tensor | None = None) -> Tensor:
        """Samples data from the final distribution of the diffusion process.

           If y is given and n_samples doesn't match the number of labels in y,
           samples should be generated accordingly to the labels in y.

        Args:
            n_samples: The number of samples to generate.
            y: Optional tensor with the class labels of the data.
               Default is `None`.

        Returns:
            X: Tensor with samples from the final distribution.
        """
        ...

    # TODO: Decide whether to accept the commented shapes
    # or just (N, M) despite losing some efficiency
    @abstractmethod
    def backward_drift_coef(self, score_model: ScoreNet, x: Tensor, t: Tensor, y: Tensor | None = None) -> Tensor:
        """Computes the drift coefficient of the SDE at time t.

        The output shape must be (N, M)

        Args:
            x: The current functional data as a tensor, shape (N, M)
            t: The current time steps as a tensor, shape (N,)
            y: Optional tensor with the class labels of the data, shape (N,)
               Default is `None`.

        Returns:
            The drift coefficient as a tensor Shape (N, M)

        """
        ...

    # TODO: Decide whether to acept the commented shapes
    # or just (N, M, M) despite losing some efficiency
    @abstractmethod
    def backward_diffusion_coef(self, t: Tensor) -> Tensor:
        """Computes the backward diffusion coefficient of the SDE at time t.

        The output shape must be (N, M, M).

        Args:
            t: The current time steps as a tensor, shape (N,)

        Returns:
            The diffusion coefficient as a tensor, shape (N, M, M).
        """
        ...

    def backward(self,
                 score_model: ScoreNet,
                 x: Tensor,
                 t: Tensor | float,
                 t_0: Tensor | float = 0.0,
                 y: Tensor | None = None) -> Tensor:
        """Applies the backward diffusion process using Euler-Maruyama.

        Args:
            score_model: The score-based model used to estimate the score
                         function.
            x: The perturbed data at time t as a tensor. Shape (N, M)
            t: The time at which the data is perturbed as a tensor with shape
               (N,) or a float. In the latter case, the behavior is the same as
                if a tensor with shape (N,) with all entries equal to t is
                passed.
            t_0: The time till which we want to reverse the diffusion process
                as a tensor with shape (N,) or a float. In the latter case,
                the behavior is the same as if a tensor with shape (N,)
                with all entries equal to t_0 is passed.
            y: Optional tensor with the class labels of the data, shape (N,)
               Default is `None`.

        Returns:
            :class:`DiffusionState` with the diffusion state at time t_0.
        """
        device = x.device
        N, M = x.shape
        if isinstance(t, float):
            t = torch.full((N,), t, device=device)
        if isinstance(t_0, float):
            t_0 = torch.full((N,), t_0, device=device)

        backward_drift = partial(self.backward_drift_coef,
                                 score_model=score_model,
                                 y=y)

        backward_diffusion = self.backward_diffusion_coef
        # TODO():Adapt this to the implemented tool.
        return euler_maruyama_integration(
            x,
            t,
            t_0,
            backward_drift,
            backward_diffusion,
            self.integration_steps,
        )


def euler_maruyama_integration(
    x: Tensor,
    t_0: Tensor,
    t_end: Tensor,
    drift: Callable[[Tensor, Tensor], Tensor],
    diffusion: Callable[[Tensor], Tensor],
    n_steps: int,
) -> Tensor:
    """Performs Euler-Maruyama integration of an SDE.

    Args:
        x: The initial data at time t as a tensor, shape (N, M)
        t_0: The initial time steps as a tensor, shape (N,)
        t_end: The final time steps as a tensor, shape (N,)
        drift: The drift function of the SDE.
        diffusion: The diffusion coefficient function of the SDE.
        n_steps: The number of integration steps.

    Returns:
        The integrated data at time t_0 as a tensor, shape (N, M)
    """
    N, M = x.shape
    device = x.device
    times = torch.linspace(t_0, t_end, n_steps + 1, device=device)
    dt = times[1] - times[0] # Can be negative
    if isinstance(dt, float):
        dt = torch.tensor(dt, device=device)
    sqrt_dt = torch.sqrt(torch.abs(dt))

    # Brownian increment: dW ~ N(0, |dt|)
    # Note: sqrt(|dt|) because variance is always positive
    dw = torch.randn((n_steps, N, M), dtype=x.dtype, device=device) * sqrt_dt
    x_t = x.clone()
    for n in range(n_steps):
        t = times[n]

        drift_t = drift(x_t, t)  # Shape (N, M)
        diffusion_t = diffusion(t)  # Shape (N, M, M)

        diff_score = torch.einsum("bi,bij->bj", dw[n], diffusion_t)  # (N, M)
        x_t = x_t + drift_t * dt + diff_score  # Shape (N, M)

    return x_t
