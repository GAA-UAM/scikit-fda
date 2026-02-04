from typing import Callable, Protocol, Final, Literal

from ...representation import FData
from score_model import ScoreNet

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
    # TODO(): Decide whether T should be Final or not
    # Decide whether other T different than 1 should be allowed
    T: Final[int] # Final time of the diffusion process

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
    # For now we assume that neither.
    def sample_final_distribution(self, n_samples: int, device: torch.device | str = "cpu") -> Tensor:
        """Samples data from the final distribution of the diffusion process.

        Args:
            n_samples: The number of samples to generate.
            device: The device on which to create the samples. Default is "cpu".

        Returns:
            X: Tensor with samples from the final distribution.
        """
        ...

    # TODO(): Decide whether to return DiffusionState or just x at t_0
    def backward(self,
                 score_model: ScoreNet,
                 x: Tensor,
                 t: Tensor | float | None = None,
                 t_0: Tensor | float = 0.0,
                 y: Tensor | None = None) -> DiffusionState:
        """Applies the backward diffusion process to the perturbed data.

        Args:
            score_model: The score network used to estimate the score
                         function of the diffusion process.
            x: The perturbed data at time t as a tensor. Shape (N, M)
            t: The time at which the data is perturbed as a 0-dimensional
               tensor or a float. Default is `None`. If `None`, it is assumed
                to be the final time T of the diffusion process.
            t_0: The time till which we want to reverse the diffusion process
                as a 0-dimensional tensor or a float.
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
    def sample_final_distribution(self, n_samples: int, device: torch.device | str = "cpu") -> Tensor:
        """Samples data from the final distribution of the diffusion process.

        Args:
            n_samples: The number of samples to generate.
            device: The device on which to create the samples. Default is "cpu".

        Returns:
            X: Tensor with samples from the final distribution.
        """
        ...

    # TODO: Decide whether to accept the commented shapes
    # or just (N, M) despite losing some efficiency
    @abstractmethod
    def backward_drift_coef(self,
                            score_model: ScoreNet, 
                            x: Tensor,
                            t: Tensor,
                            y: Tensor | None = None) -> Tensor:
        """Computes the drift coefficient of the SDE at time t.

        The output shape must be (N, M)

        Args:
            score_model: The score-based model used to estimate the score
                         function.
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

        The output shape can be (N, M, M), (N, M), (N,) or scalar.

        In the case (N, M) is returned, it represents only the diagonal
        of the diffusion matrix.
        In the case (N,) is returned, it represents a scalar multiple of
        the identity matrix.
        In the case of a scalar the behavior is the same as in the case
        of (N,) with all values equal to that scalar.

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
            t: The time at which the data is perturbed as a 0-dimensional
               tensor or a float.
            t_0: The time till which we want to reverse the diffusion process
                as a 0-dimensional tensor or a float.
            y: Optional tensor with the class labels of the data, shape (N,)
               Default is `None`.

        Returns:
            :class:`DiffusionState` with the diffusion state at time t_0.
        """
        device = x.device

        if isinstance(t, float):
            t = torch.tensor(t, device=device)
        if isinstance(t_0, float):
            t_0 = torch.tensor(t_0, device=device)
        if t.ndim != 0 or t_0.ndim != 0:
            raise ValueError("t and t_0 must be either floats or 0-dimensional tensors")

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
    # Note: sqrt(|dt|) because variance is always positive
    sqrt_dt = torch.sqrt(torch.abs(dt))

    # Brownian increment: dW ~ N(0, |dt|)
    dw = torch.randn((n_steps, N, M), dtype=x.dtype, device=device) * sqrt_dt
    x_t = x.clone()
    for n in range(n_steps):
        t = times[n]

        drift_t = drift(x_t, t)  # Shape (N, M)
        diffusion_t = diffusion(t)  # Shape (N, M, M)
        if diffusion_t.dim() in (0, 1, 2):
            # Scalar, (N,) or (N,M) shape
            # Element-wise multiplication
            diff_score = diffusion_t * dw[n]
        else:
            # Full (N, M, M) shape
            # Matrix-vector multiplication
            diff_score = torch.einsum("bi,bij->bj", dw[n], diffusion_t)
        x_t = x_t + drift_t * dt + diff_score  # Shape (N, M)

    return x_t


class VariancePreservingDiffusionProcess(EulerMaruyamaDiffusionProcess):
    r"""Implements a variance-preserving diffusion process.

    Based on the following forward SDE equation:
    :math:`d\mathbf{X}(t) = -\frac{1}{2}\beta(t)\mathbf{X}(t)dt + \sqrt{\beta(t)}d\mathbf{W}(t)`

    where :math:`\beta(t)` is a time-dependent function controlling
    the noise level.

    The linear and cosine schedules for :math:`\beta(t)` are implemented.
    In addition the :math:`\beta(0)` and :math:`\beta(T)` values can be set.
    In this class :math;`T` is taken to be 1.

    When using `cosine` schedule the values of :math:`\beta(0)` and
    :math:`\beta(T)` are ignored.
    """
    # TODO(): Decide whether to allow other T values
    T: Final[int] = 1

    def __init__(self,
                M:int,
                beta_schedule: Literal["linear", "cosine"] = "linear",
                beta_0: float = 0.01,
                beta_T: float = 3.0,
                integration_steps: int = 1000,
                device: torch.device | str = 'cpu',
                ):
        """Initializes the variance-preserving diffusion process.

        Args:
            M: The dimension of the data. Needed to sample
            from the final distribution.
            beta_schedule: The schedule for beta(t). Can be 'linear' or
                        'cosine'. Default is 'linear'.
            beta_0: The value of beta(0). Used only if `beta_schedule`
                    is 'linear'. Default is 0.01.
            beta_T: The value of beta(T). Used only if `beta_schedule`
                    is 'linear'. Default is 3.0.
            integration_steps: The number of integration steps for the
                                Euler-Maruyama method. Default is 1000.
            device: The device to run the computations on. Default is 'cpu'.
        """
        if beta_schedule not in ("linear", "cosine"):
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        # TODO(): Decide if I should pass as a class attribute or argument
        # in the sample_final_distribution method
        self.M = M
        self.beta_schedule = beta_schedule
        self.beta_0 = beta_0
        self.beta_T = beta_T
        self.integration_steps = integration_steps
        self.device = device

    def _beta_t(self, t: Tensor) -> Tensor:
        """Computes the value of beta(t) at time t.

        Args:
            t: The time steps as a tensor, shape (N,)

        Returns:
            The value of beta(t) as a tensor, shape (N,)
        """
        if self.beta_schedule == "linear":
            beta_t = self.beta_0 + (self.beta_T - self.beta_0) * t / self.T
        elif self.beta_schedule == "cosine":
            # TODO(): What should I do with eps here?
            s = 0.008
            eps = 1e-5
            beta_t = torch.pi / (self.T * (s + 1))* torch.tan(torch.pi*0.5 * (t / self.T + s) / (s + 1 + eps))
            beta_t = torch.clamp(beta_t, min=0.0, max=1.0)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
        return beta_t

    def backward_drift_coef(self,
                            score_model: ScoreNet,
                            x: Tensor,
                            t: Tensor,
                            y: Tensor | None = None,
                            ) -> Tensor:
        r"""Computes the drift coefficient of the backward SDE at time t.

        If the forward SDE is defined as:
        :math:`d\mathbf{X}(t) = -\frac{1}{2}\beta(t)\mathbf{X}(t)dt + \sqrt{\beta(t)}d\mathbf{W}(t)`

        Then the backward SDE is defined as:
        :math:`d\mathbf{X}(t) = \left[-\frac{1}{2}\beta(t)\mathbf{X}(t) - \beta(t)\nabla_{\mathbf{X}}\log p_t(\mathbf{X}(t)|y)\right]dt + \sqrt{\beta(t)}d\mathbf{W}(t)`

        Hence, the drift coefficient is:
        :math:`-\frac{1}{2}\beta(t)\mathbf{X}(t) - \beta(t)\nabla_{\mathbf{X}}\log p_t(\mathbf{X}(t)|y)`

        Args:
            score_model: The score-based model used to estimate the score
                         function.
            x: The current functional data as a tensor, shape (N, M)
            t: The current time steps as a tensor, shape (N,)
            y: Optional tensor with the class labels of the data, shape (N,)
               Default is `None`.

        Returns:
            The drift coefficient as a tensor Shape (N, M)
        """
        beta = self._beta_t(t)  # Shape (N,)
        score = score_model(x, t, y)  # Shape (N, M)

        return -0.5 * beta.unsqueeze(1) * x - beta.unsqueeze(1) * score  # Shape (N, M)

    def backward_diffusion_coef(self, t: Tensor) -> Tensor:
        r"""Computes the diffusion coefficient of the backward SDE at time t.

        If the forward SDE is defined as:
        :math:`d\mathbf{X}(t) = -\frac{1}{2}\beta(t)\mathbf{X}(t)dt + \sqrt{\beta(t)}d\mathbf{W}(t)`

        Then the backward SDE is defined as:
        :math:`d\mathbf{X}(t) = \left[-\frac{1}{2}\beta(t)\mathbf{X}(t) - \beta(t)\nabla_{\mathbf{X}}\log p_t(\mathbf{X}(t)|y)\right]dt + \sqrt{\beta(t)}d\mathbf{W}(t)`

        Hence, the diffusion coefficient is:
        :math:`\sqrt{\beta(t)}`

        Args:
            score_model: The score-based model used to estimate the score
                         function.
            x: The current functional data as a tensor, shape (N, M)
            t: The current time steps as a tensor, shape (N,)
            y: Optional tensor with the class labels of the data, shape (N,)
               Default is `None`.

        Returns:
            The diffusion coefficient as a tensor Shape (N,)
        """
        beta = self._beta_t(t)  # Shape (N,)
        return torch.sqrt(beta)  # Shape (N,)

    def inv_sigma_t(self, t: Tensor) -> Tensor:
        r"""Computes the inverse of the square root of the covariance matrix at time t.

        For the variance-preserving diffusion process, the covariance matrix
        at time t is given by:
        :math:`\sigma_t^2 =\int_0^t \beta(s) \exp(\int_s^t\beta(u) du) ds =
        \left(1 - \exp(-\int_0^t \beta(s) ds)\right)`

        Hence, the inverse of the square root of the covariance matrix is
        given by the square root of the inverse of :math:`\sigma_t^2`.

        Args:
            t: The time steps as a tensor, shape (N,)

        Returns:
            inv_sigma_t: The inverse of the square root of the covariance
                         matrix at time t, shape (N,)
        """
        # Compute the integral of beta from 0 to t using the trapezoidal rule
        sigma_t = self._sigma_t(t)  # Shape (N,)
        return 1.0 / sigma_t  # Shape (N,)

    def _sigma_t(self, t: Tensor) -> Tensor:
        r"""Computes the square root of the covariance matrix at time t.

        For the variance-preserving diffusion process, the covariance matrix
        at time t is given by:
        :math:`\sigma_t^2 =\int_0^t \beta(s) \exp(-\int_s^t\beta(u) du) ds =
        \left(1 - \exp(-\int_0^t \beta(s) ds)\right)`

        In the case of a linear schedule for beta(t), the integral can be computed
        analytically as:
        :math:`\int_0^t \beta(s) ds = \beta_0 * t + \frac{(beta_T - beta_0) * t^2}{2T}`
        Hence sigma_t^2 can be computed as:
        :math:`\sigma_t^2 = \left(1 - \exp(-(\beta_0 * t +  \frac{(beta_T - beta_0) * t^2}{2T}))\right)`

        In the case of a cosine schedule for beta(t), the integral is given by:
        :math:`\int_0^t \beta(s) ds = -\ln\left(\frac{f(t)}{f(0)}\right)`
        where :math:`f(t) = \cos^2\left(\frac{\pi}{2}\frac{t/s + 1}{1 + s}\right)`
        Hence sigma_t^2 can be computed as:
        :math:`\sigma_t^2 = \left(1 - \frac{f(t)}{f(0)}\right)`

        Args:
            t: The time steps as a tensor, shape (N,)

        Returns:
            inv_sigma_t: The inverse of the square root of the covariance
                         matrix at time t, shape (N,)
        """
        if self.beta_schedule == "linear":
            beta_0 = self.beta_0
            beta_T = self.beta_T
            integral_beta = beta_0 * t + 0.5 * (beta_T - beta_0) * (t ** 2) / self.T
            return torch.sqrt(1 - torch.exp(-integral_beta))  # Shape (N,)
        elif self.beta_schedule == "cosine":
            s = 0.008
            f_t = torch.cos(((t / self.T + s) / (1 + s)) * (torch.pi / 2)) ** 2
            f_0 = torch.cos((s / (1 + s)) * (torch.pi / 2)) ** 2
            sigma_t_squared = 1 - f_t / f_0  # Shape (N,)
            return torch.sqrt(sigma_t_squared)  # Shape (N,)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")

    def _mu_t(self, t: Tensor) -> Tensor:
        r"""Computes the mean of the diffusion process at time t.

        For the variance-preserving diffusion process, the mean at time t is
        given by:
        :math:`\mu_t = \exp(-0.5 * \int_0^t \beta(s) ds)`

        Args:
            t: The time steps as a tensor, shape (N,)

        Returns:
            mu_t: The mean of the diffusion process at time t, shape (N,)
        """
        if self.beta_schedule == "linear":
            beta_0 = self.beta_0
            beta_T = self.beta_T
            integral_beta = beta_0 * t + 0.5 * (beta_T - beta_0) * (t ** 2) / self.T
            return torch.exp(-0.5 * integral_beta)  # Shape (N,)
        elif self.beta_schedule == "cosine":
            s = 0.008
            f_t = torch.cos(((t / self.T + s) / (1 + s)) * (torch.pi / 2)) ** 2
            f_0 = torch.cos((s / (1 + s)) * (torch.pi / 2)) ** 2
            return torch.sqrt(f_t / f_0)  # Shape (N,)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")

    def forward(self, x: Tensor, t: Tensor) -> DiffusionState:
        r"""Applies the forward diffusion process to the input data.

        For the variance-preserving diffusion process, the forward diffusion
        process is given by:
        :math:`\mathbf{X}(t) = \mu_t \mathbf{X}(0) + \sigma_t \mathbf{Z}`
        where :math:`\mathbf{Z} \sim \mathcal{N}(0, I)`
        
        Read _mu_t and _sigma_t methods for more details.

        Args:
            x: The input functional data as a tensor, shape (N, M)
            t: The time steps at which to apply the diffusion, shape (N,)
        Returns:
            :class:`DiffusionState` with the diffusion state at time t.
        """
        mu_t = self._mu_t(t)  # Shape (N,)
        sigma_t = self._sigma_t(t)  # Shape (N,)
        z = torch.randn_like(x)  # Shape (N, M)
        x_t = mu_t.unsqueeze(1) * x + sigma_t.unsqueeze(1) * z # Shape (N, M)
        inv_sigma_t = 1.0 / sigma_t  # Shape (N,)
        return DiffusionState(
            t=t,
            x=x_t,
            mu_t=mu_t,
            sigma_t=sigma_t,
            inv_sigma_t=inv_sigma_t,
        )

    # TODO(): Decide how to acces the value of M.
    def sample_final_distribution(self, n_samples: int) -> Tensor:
        r"""Samples data from the final distribution of the diffusion process.

        For the variance-preserving diffusion process, the final distribution
        at time T is given by a standard normal distribution:
        :math:`\mathbf{X}(T) \sim \mathcal{N}(0, I)`

        Args:
            n_samples: The number of samples to generate.

        Returns:
            X: Tensor with samples from the final distribution.
            Shape (n_samples, M)
        """
        return torch.randn((n_samples, self.M), device=self.device)
