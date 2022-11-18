"""Shift registration of functional data by least squares."""
from __future__ import annotations

import warnings
from typing import Callable, Optional, Tuple, TypeVar, Union

import numpy as np
from sklearn.utils.validation import check_is_fitted
from typing_extensions import Literal

from ...misc._math import inner_product
from ...misc.metrics._lp_norms import l2_norm
from ...misc.validation import check_fdata_dimensions
from ...representation import FData, FDataGrid
from ...representation.extrapolation import ExtrapolationLike
from ...typing._base import GridPointsLike
from ...typing._numpy import ArrayLike, NDArrayFloat
from .base import InductiveRegistrationTransformer

SelfType = TypeVar("SelfType", bound="LeastSquaresShiftRegistration[FData]")
T = TypeVar("T", bound=FData)
TemplateFunction = Callable[[FDataGrid], FDataGrid]


class LeastSquaresShiftRegistration(
    InductiveRegistrationTransformer[T, T],
):
    r"""Register data using shift alignment by least squares criterion.

    Realizes the registration of a set of curves using a shift aligment
    :footcite:`ramsay+silverman_2005_functional_shift`.
    Let :math:`\{x_i(t)\}_{i=1}^{N}` be a functional dataset, calculates
    :math:`\delta_{i}` for each sample such that :math:`x_i(t + \delta_{i})`
    minimizes the least squares criterion:

    .. math::
        \text{REGSSE} = \sum_{i=1}^{N} \int_{\mathcal{T}}
        [x_i(t + \delta_i) - \hat\mu(t)]^2 ds

    Estimates each shift parameter :math:`\delta_i` iteratively by
    using a modified Newton-Raphson algorithm, updating the template
    :math:`\mu` in each iteration as is described in detail in
    :footcite:`ramsay+silverman_2005_functional_newton-raphson`.

    Method only implemented for univariate functional data.

    Args:
        max_iter: Maximun number of iterations.
            Defaults sets to 5. Generally 2 or 3 iterations are sufficient to
            obtain a good alignment.
        tol: Tolerance allowable. The process will stop if
            :math:`\max_{i}|\delta_{i}^{(\nu)}-\delta_{i}^{(\nu-1)}|<tol`.
            Default sets to 1e-2.
        template: Template to use in the
            least squares criterion. If template="mean" it is use the
            functional mean as in the original paper. The template can be a
            callable that will receive an FDataGrid with the samples and will
            return another FDataGrid as a template, such as any of the means or
            medians of the module `skfda.explotatory.stats`.
            If the template is an FData is used directly as the final
            template to the registration, if it is a callable or "mean" the
            template is computed iteratively constructing a temporal template
            in each iteration.
            In :footcite:`ramsay+silverman_2005_functional_newton-raphson`
            is described in detail this procedure. Defaults to "mean".
        extrapolation: Controls the
            extrapolation mode for points outside the :term:`domain` range.
            By default uses the method defined in the data to be transformed.
            See the `extrapolation` documentation to obtain more information.
        step_size: Parameter to adjust the rate of
            convergence in the Newton-Raphson algorithm, see
            :footcite:`ramsay+silverman_2005_functional_newton-raphson`.
            Defaults to 1.
        restrict_domain: If True restricts the :term:`domain`
            to avoid the need of using extrapolation, in which
            case only the fit_transform method will be available, as training
            and transformation must be done together. Defaults to False.
        initial: Array with an initial estimation
            of shifts. Default uses a list of zeros for the initial shifts.
        grid_points: Set of points where the
            functions are evaluated to obtain the discrete
            representation of the object to integrate. If None is
            passed it calls numpy.linspace in FDataBasis and uses the
            `grid_points` in FDataGrids.

    Attributes:
        template\_: Template :math:`\mu` learned during the fitting
            used to the transformation.
        deltas\_: List of shifts :math:`\delta_i` applied
            during the last transformation.
        n_iter\_: Number of iterations performed during the last
            transformation.

    Note:
        Due to the use of derivatives for the estimation of the shifts, the
        samples to be registered may be smooth for the correct convergence of
        the method.

    Examples:
        >>> from skfda.preprocessing.registration import (
        ...     LeastSquaresShiftRegistration,
        ... )
        >>> from skfda.datasets import make_sinusoidal_process
        >>> from skfda.representation.basis import FourierBasis


        Registration and creation of dataset in discretized form:

        >>> fd = make_sinusoidal_process(n_samples=10, error_std=0,
        ...                              random_state=1)
        >>> reg = LeastSquaresShiftRegistration(extrapolation="periodic")
        >>> fd_registered = reg.fit_transform(fd)
        >>> fd_registered
        FDataGrid(...)

        Shifts applied during the transformation

        >>> reg.deltas_.round(3)
        array([-0.131,  0.188,  0.026,  0.033, -0.109,  0.115, ..., -0.062])


        Registration of a dataset in basis form using the
        transformation previosly fitted. The result is a dataset in
        discretized form, as it is not possible to express shifted functions
        exactly as a basis expansion:

        >>> fd = make_sinusoidal_process(n_samples=2, error_std=0,
        ...                              random_state=2)
        >>> fd_basis = fd.to_basis(FourierBasis())
        >>> reg.transform(fd_basis)
        FDataGrid(...)


    References:
        .. footbibliography::

    """

    def __init__(
        self,
        max_iter: int = 5,
        tol: float = 1e-2,
        template: Union[Literal["mean"], FData, TemplateFunction] = "mean",
        extrapolation: Optional[ExtrapolationLike] = None,
        step_size: float = 1,
        restrict_domain: bool = False,
        initial: Union[Literal["zeros"], ArrayLike] = "zeros",
        grid_points: Optional[GridPointsLike] = None,
    ) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.template = template
        self.restrict_domain = restrict_domain
        self.extrapolation = extrapolation
        self.step_size = step_size
        self.initial = initial
        self.grid_points = grid_points

    def _compute_deltas(
        self,
        fd: FData,
        template: Union[Literal["mean"], FData, TemplateFunction],
    ) -> Tuple[NDArrayFloat, FDataGrid]:
        """Compute the shifts to perform the registration.

        Args:
            fd: Functional object to be registered.
            template: Template to align the
                the samples. "mean" to compute the mean iteratively as in
                the original paper, an FData with the templated calculated or
                a callable wich constructs the template.

        Returns:
            A tuple with an array of deltas and an FDataGrid with the template.

        """
        check_fdata_dimensions(
            fd,
            dim_domain=1,
            dim_codomain=1,
        )

        domain_range = fd.domain_range[0]

        # Initial estimation of the shifts
        if self.initial == "zeros":
            delta = np.zeros(fd.n_samples)
        else:
            delta = np.asarray(self.initial)

            if len(delta) != fd.n_samples:
                raise ValueError(
                    f"The length of the initial shift ({len(delta)}) must "
                    f"be the same than the number of samples ({fd.n_samples})",
                )

        # Auxiliar array to avoid multiple memory allocations
        delta_aux = np.empty(fd.n_samples)

        # Computes the derivate of originals curves in the mesh points
        fd_deriv = fd.derivative()

        # Second term of the second derivate estimation of REGSSE. The
        # first term has been dropped to improve convergence (see references)
        d2_regsse = l2_norm(fd_deriv)**2

        # We need the discretized derivative to compute the inner product later
        fd_deriv = fd_deriv.to_grid(grid_points=self.grid_points)

        max_diff = self.tol + 1
        self.n_iter_ = 0

        # Newton-Rhapson iteration
        while max_diff > self.tol and self.n_iter_ < self.max_iter:

            # Computes the new values shifted
            x = fd.shift(delta, grid_points=self.grid_points)

            if isinstance(template, str):
                assert template == "mean"
                template_iter = x.mean()
            elif isinstance(template, FData):
                template_iter = template.to_grid(grid_points=x.grid_points)
            else:  # Callable
                template_iter = template(x)

            # Updates the limits for non periodic functions ignoring the ends
            if self.restrict_domain:
                # Calculates the new limits
                a = domain_range[0] - min(float(np.min(delta)), 0)
                b = domain_range[1] - max(float(np.max(delta)), 0)

                restricted_domain = (
                    max(a, template_iter.domain_range[0][0]),
                    min(b, template_iter.domain_range[0][1]),
                )

                template_iter = template_iter.restrict(restricted_domain)

                x = x.restrict(restricted_domain)
                fd_deriv = fd_deriv.restrict(restricted_domain)
                d2_regsse = l2_norm(fd_deriv)**2

            # Calculates x - mean
            x -= template_iter

            d1_regsse = inner_product(x, fd_deriv)

            # Updates the shifts by the Newton-Rhapson iteration
            # Same as delta = delta - step_size * d1_regsse / d2_regsse
            delta_aux[:] = d1_regsse
            delta_aux[:] /= d2_regsse
            delta_aux[:] *= self.step_size
            delta[:] -= delta_aux

            # Updates convergence criterions
            max_diff = np.abs(delta_aux, out=delta_aux).max()
            self.n_iter_ += 1

        return delta, template_iter

    def fit_transform(self, X: T, y: object = None) -> T:

        deltas, template = self._compute_deltas(X, self.template)

        self.deltas_ = deltas
        self.template_ = template

        shifted = X.shift(
            self.deltas_,
            restrict_domain=self.restrict_domain,
            extrapolation=self.extrapolation,
            grid_points=self.grid_points,
        )
        shifted.argument_names = None  # type: ignore[assignment]
        return shifted

    def fit(
        self: SelfType,
        X: FData,
        y: object = None,
    ) -> SelfType:

        # If the template is an FData, fit doesnt learn anything
        if isinstance(self.template, FData):
            self.template_ = self.template

        else:
            _, template = self._compute_deltas(X, self.template)

            self.template_ = template

        return self

    def transform(self, X: FData, y: object = None) -> FDataGrid:

        if self.restrict_domain:
            raise AttributeError(
                "transform is not available when "
                "restrict_domain=True, fitting and "
                "transformation should be done together. Use "
                "an extrapolation method with "
                "restrict_domain=False or fit_predict",
            )

        # Check is fitted
        check_is_fitted(self)

        deltas, _ = self._compute_deltas(X, self.template_)
        self.deltas_ = deltas

        shifted = X.shift(
            deltas,
            restrict_domain=self.restrict_domain,
            extrapolation=self.extrapolation,
            grid_points=self.grid_points,
        )
        shifted.argument_names = None  # type: ignore[assignment]
        return shifted

    def inverse_transform(self, X: FData, y: object = None) -> FDataGrid:
        """
        Apply the inverse transformation.

        Applies the opossite shift used in the last call to `transform`.

        Args:
            X: Functional dataset to be transformed.
            y: not used, present for API consistency by convention.

        Returns:
            Functional data registered.

        Examples:
            Creates a synthetic functional dataset.

            >>> from skfda.preprocessing.registration import (
            ...     LeastSquaresShiftRegistration,
            ... )
            >>> from skfda.datasets import make_sinusoidal_process
            >>> fd = make_sinusoidal_process(error_std=0, random_state=1)
            >>> fd.extrapolation = 'periodic'

            Dataset registration and centering.

            >>> reg = LeastSquaresShiftRegistration()
            >>> fd_registered = reg.fit_transform(fd)
            >>> fd_centered = fd_registered - fd_registered.mean()

            Reverse the translation applied during the registration.

            >>> reg.inverse_transform(fd_centered)
            FDataGrid(...)

        """
        deltas = getattr(self, "deltas_", None)

        if deltas is None:
            raise AttributeError(
                "Data must be previously transformed to learn"
                " the inverse transformation",
            )
        elif len(X) != len(deltas):
            raise ValueError(
                "Data must contain the same number of samples "
                "than the dataset previously transformed",
            )

        return X.shift(
            -deltas,
            restrict_domain=self.restrict_domain,
            extrapolation=self.extrapolation,
            grid_points=self.grid_points,
        )


class ShiftRegistration(LeastSquaresShiftRegistration[T]):
    """Deprecated name for LeastSquaresShiftRegistration."""

    def __init__(
        self,
        max_iter: int = 5,
        tol: float = 1e-2,
        template: Union[Literal["mean"], FData, TemplateFunction] = "mean",
        extrapolation: Optional[ExtrapolationLike] = None,
        step_size: float = 1,
        restrict_domain: bool = False,
        initial: Union[Literal["zeros"], ArrayLike] = "zeros",
        grid_points: Optional[GridPointsLike] = None,
    ) -> None:
        warnings.warn(
            "ShiftRegistration has been renamed. "
            "Use LeastSquaresShiftRegistration instead.",
            DeprecationWarning,
        )
        super().__init__(
            max_iter=max_iter,
            tol=tol,
            template=template,
            extrapolation=extrapolation,
            step_size=step_size,
            restrict_domain=restrict_domain,
            initial=initial,
            grid_points=grid_points,
        )
