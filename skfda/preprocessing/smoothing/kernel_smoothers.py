import abc
import warnings
from typing import Callable, Optional

from ...misc import kernels
from ...misc.hat_matrix import (
    HatMatrix,
    KNeighborsHatMatrix,
    LocalLinearRegressionHatMatrix,
    NadarayaWatsonHatMatrix,
)
from ...typing._base import GridPointsLike
from ...typing._numpy import NDArrayFloat
from . import KernelSmoother
from ._linear import _LinearSmoother

warnings.warn(
    "The \"kernel_smoothers\" module is deprecated. "
    "Use the \"KernelSmoother\" class instead",
    DeprecationWarning,
)


class _DeprecatedLinearKernelSmoother(_LinearSmoother):

    def __init__(
        self,
        *,
        smoothing_parameter: Optional[float] = None,
        kernel: Callable[[NDArrayFloat], NDArrayFloat] = kernels.normal,
        weights: Optional[NDArrayFloat] = None,
        output_points: Optional[GridPointsLike] = None,
    ):
        self.smoothing_parameter = smoothing_parameter
        self.kernel = kernel
        self.weights = weights
        self.output_points = output_points

        warnings.warn(
            f"Class \"{type(self)}\" is deprecated. "
            "Use the \"KernelSmoother\" class instead",
            DeprecationWarning,
        )

    def _hat_matrix(
        self,
        input_points: GridPointsLike,
        output_points: GridPointsLike,
    ) -> NDArrayFloat:

        return KernelSmoother(
            kernel_estimator=self._get_kernel_estimator(),
            weights=self.weights,
            output_points=output_points,
        )._hat_matrix(
            input_points=input_points,
            output_points=output_points,
        )

    @abc.abstractmethod
    def _get_kernel_estimator(self) -> HatMatrix:
        raise NotImplementedError


class NadarayaWatsonSmoother(_DeprecatedLinearKernelSmoother):
    """Nadaraya-Watson smoother (deprecated)."""

    def _get_kernel_estimator(self) -> HatMatrix:
        return NadarayaWatsonHatMatrix(
            bandwidth=self.smoothing_parameter,
            kernel=self.kernel,
        )


class LocalLinearRegressionSmoother(_DeprecatedLinearKernelSmoother):
    """Local linear regression smoother (deprecated)."""

    def _get_kernel_estimator(self) -> HatMatrix:
        return LocalLinearRegressionHatMatrix(
            bandwidth=self.smoothing_parameter,
            kernel=self.kernel,
        )


class KNeighborsSmoother(_DeprecatedLinearKernelSmoother):
    """Local linear regression smoother (deprecated)."""

    def __init__(
        self,
        *,
        smoothing_parameter: Optional[int] = None,
        kernel: Callable[[NDArrayFloat], NDArrayFloat] = kernels.uniform,
        weights: Optional[NDArrayFloat] = None,
        output_points: Optional[GridPointsLike] = None,
    ):
        super().__init__(
            smoothing_parameter=smoothing_parameter,
            kernel=kernel,
            weights=weights,
            output_points=output_points,
        )

    def _get_kernel_estimator(self) -> HatMatrix:
        return KNeighborsHatMatrix(
            n_neighbors=(
                int(self.smoothing_parameter)
                if self.smoothing_parameter is not None
                else None
            ),
            kernel=self.kernel,
        )
