from typing import Callable, Protocol, Final, Literal

from ..._utils._sklearn_adapter import BaseEstimator

import torch
from torch import Tensor

from abc import ABC, abstractmethod

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


class CustomDiffusionProcess(DiffusionProcess):
    """Common interface of a diffusion process to be used in generative models.

    It defines the drift and diffusion terms of an SDE.
    """
    def __init__(self,
                 drift: Callable[[Tensor, Tensor], Tensor],
                 diffusion: Callable[[Tensor], Tensor]):
        """Initializes the diffusion process with the given drift and diffusion functions.

        Args:
            drift: A function that takes the current state x and time t, 
            and returns the drift term of the SDE, shape (N, M) or (N,).
            diffusion: A function that takes the current time t and 
            returns the diffusion term of the SDE, shape (N,), (N, M) or (N, M, M).
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


# TODO(): Decide whether an abstrac class or Protocol is better for this
class ForwardDiffusionProcess(DiffusionProcess, BaseEstimator):
    """Defines the forward diffusion process of a generative model.

    This method represents a diffusion process with gaussian
    conditional distributions. It defines the methods
    mean_cond and sigma_cond of the conditional distribution at time t.
    where mean is mean of the distribution and sigma is a square root
    of the covariance matrix.

    It also defines a method to compute the inverse of the square root
    of the covariance matrix at time t.

    Also defines the method to sample from the limiting distribution.
    """
    @abstractmethod
    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """Computes the drift term of the SDE at time t."""
        ...

    @abstractmethod
    def diffusion(self, t: Tensor) -> Tensor:
        """Computes the diffusion term of the SDE at time t."""
        ...

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
    def sigma_cond(self, t: Tensor) -> Tensor:
        """Computes the square root of the covariance matrix of the conditional distribution at time t.

        It computes the square root of the covariance matrix of the
        distribution of X(t) given X(0) = x.

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
            The square root of the covariance matrix of the conditional
            distribution at time t, shape (N,), (N, M) or (N, M, M)
        """
        ...

    @abstractmethod
    def inv_sigma_cond(self, t: Tensor) -> Tensor:
        """Computes the inverse of the square root of the covariance matrix of the conditional distribution at time t.

        It computes the inverse of the square root of the covariance matrix of the
        distribution of X(t) given X(0) = x.

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
            The inverse of the square root of the covariance matrix of the conditional
            distribution at time t, shape (N,), (N, M) or (N, M, M)
        """
        ...

    @abstractmethod
    def sample_limit_distribution(
        self,
        n_samples: int,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """Samples data from the limit distribution of the diffusion process.

        The output shape must be (n_samples, M) where M is the dimension of the data.
        It is learned at the fit method.

        Args:
            n_samples: The number of samples to generate.
            device: The device on which to create the samples. Default is "cpu".

        Returns:
            X: Tensor with samples from the limit distribution, shape (n_samples, M)
        """
        ...

    def fit(self, x: Tensor) -> "ForwardDiffusionProcess":
        """Fits the parameters of the diffusion process to the data.

        This method can be used to fit the parameters of the
        diffusion process to the data.For the most basic, it learns the
        dimensionality of the data for sampling from limit distribution.
        More advanced diffusion process can learn other parameters such as
        the beta(t) schedule for variance-preserving.

        Args:
            x: The input functional data as a tensor, shape (N, M)

        Returns:
            self: The fitted diffusion process.
        """
        self.M = x.shape[1]  # Learn the dimension of the data
        return self


class VariancePreservingDiffusionProcess(ForwardDiffusionProcess):
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

    def __init__(
        self,
        beta_schedule: Literal["linear", "cosine"] = "linear",
        beta_0: float = 0.001,
        beta_T: float = 10.,
        integration_steps: int = 1000,
    ):
        """Initializes the variance-preserving diffusion process.

        Args:
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
        self.beta_schedule = beta_schedule
        self.beta_0 = beta_0
        self.beta_T = beta_T
        self.integration_steps = integration_steps
        self.M = None  # Will be set in fit method

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
            beta_t = (torch.pi / (self.T * (s + 1)) *
                      torch.tan(
                                torch.pi * 0.5 *
                                (t / self.T + s) / (s + 1 + eps),
                                )
                      )
            beta_t = torch.clamp(beta_t, min=0.0, max=1.0)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
        return beta_t

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        r"""Computes the drift term of the SDE at time t.

        For the variance-preserving diffusion process, the drift term is given by:
        :math:`\mathbf{f}(\mathbf{X}(t), t) = -\frac{1}{2} \beta(t) \mathbf{X}(t)`
        """
        return -0.5 * self._beta_t(t).unsqueeze(1) * x

    def diffusion(self, t: Tensor) -> Tensor:
        r"""Computes the diffusion term of the SDE at time t.

        For the variance-preserving diffusion process, the diffusion term is given by:
        :math:`g(t) = \sqrt{\beta(t)}`
        """
        return torch.sqrt(self._beta_t(t))

    def mean_cond(self, x: Tensor, t: Tensor) -> Tensor:
        r"""Computes the mean of the conditional distribution at time t.

        For the variance-preserving diffusion process, the mean of the
        conditional distribution is given by:
        :math:`\mu_t = \exp(-0.5 * \int_0^t \beta(s) ds) * x`

        Args:
            x: The data to condition on as a tensor, shape (N, M)
            t: The time steps as a tensor, shape (N,)

        Returns:
            mu_t: The mean of the diffusion process at time t, shape (N,M)
        """
        if self.beta_schedule == "linear":
            beta_0 = self.beta_0
            beta_T = self.beta_T

            integral_beta = beta_0 * t + 0.5 * (beta_T - beta_0) * (t ** 2) / self.T

            mu_t = torch.exp(-0.5 * integral_beta)  # Shape (N,)

        elif self.beta_schedule == "cosine":
            s = torch.tensor(0.008)
            f_t = torch.cos(((t / self.T + s) / (1 + s)) * (torch.pi / 2)) ** 2
            f_0 = torch.cos((s / (1 + s)) * (torch.pi / 2)) ** 2

            mu_t = torch.sqrt(f_t / f_0)  # Shape (N,)
        return mu_t.unsqueeze(1) * x

    def inv_sigma_cond(self, t: Tensor) -> Tensor:
        r"""Computes the inverse of the square root of the covariance matrix at time t.

        For the variance-preserving diffusion process, the covariance matrix
        at time t is given by:
        :math:`\sigma_t^2 =\int_0^t \beta(s) \exp(\int_s^t\beta(u) du) ds =
        \left(1 - \exp(-\int_0^t \beta(s) ds)\right)`

        Hence, the inverse of the square root of the covariance matrix is
        given by the square root of the inverse of :math:`\sigma_t^2`.

        For stability reasons, we clamp the values of :math:`\sigma_t` to be
        at least 1e-4 to avoid division by zero or extremely large
        values of inv_sigma_t, which can cause instability in the
        training process.

        Args:
            t: The time steps as a tensor, shape (N,)

        Returns:
            inv_sigma_t: The inverse of the square root of the covariance
                         matrix at time t, shape (N,)
        """
        # Compute the integral of beta from 0 to t using the trapezoidal rule
        sigma_t = self.sigma_cond(t)  # Shape (N,)
        sigma_t_safe = torch.clamp(sigma_t, min=1e-5)  # Avoid division by zero
        return 1.0 / sigma_t_safe  # Shape (N,)

    def sigma_cond(self, t: Tensor) -> Tensor:
        r"""Computes the square root of the covariance matrix at time t.

        For the variance-preserving diffusion process, the covariance matrix
        at time t is given by:
        :math:`\sigma_t^2 =\int_0^t \beta(s) \exp(-\int_s^t\beta(u) du) ds =
        \left(1 - \exp(-\int_0^t \beta(s) ds)\right)`

        In the case of a linear schedule for beta(t), the integral can be computed
        analytically as:/home/diego-linux/Documentos/Universidad/Quinto/TFG_INFO/TFG_Info/code/SDE_B_Method_C/metrics.py
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

        if self.beta_schedule == "cosine":
            s = torch.tensor(0.008)
            f_t = torch.cos(((t / self.T + s) / (1 + s)) * (torch.pi / 2)) ** 2
            f_0 = torch.cos((s / (1 + s)) * (torch.pi / 2)) ** 2

            sigma_t_squared = 1 - f_t / f_0  # Shape (N,)

            return torch.sqrt(sigma_t_squared)  # Shape (N,)

        raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")

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
            grid_size: The number of discretization points of the functional data.
                          Refers to the M dimension of the data.
            device: The device on which to create the samples.
                    Default is "cpu".

        Returns:
            X: Tensor with samples from the final distribution.
            Shape (n_samples, grid_size)
        """
        # TODO(): Decide wheteher to check if M is None or if it has an attribute.
        if not hasattr(self, "M"):
            raise ValueError("The diffusion process must be fitted to the data before sampling from the final distribution. Call the fit method with the training data.")
        if self.M is None:
            raise ValueError("The diffusion process must be fitted to the data before sampling from the final distribution. Call the fit method with the training data.")
        return torch.randn((n_samples, self.M), device=device)
