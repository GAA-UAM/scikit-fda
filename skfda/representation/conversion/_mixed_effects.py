# -*- coding: utf-8 -*-
r"""
Mixed effects converters
========================

This module contains the classes for converting irregular data to basis
representation using the mixed effects model.

The use of the mixed effects model for conversion of irregularly sampled
functional data to a basis representation is detailed in
:footcite:t:`james_2018_sparsenessfda`. In the following, we provide a brief
overview of the model for 1-dimensional functional data.

The mixed effects model for functional data
-------------------------------------------

Let :math:`\{x_i(t)\}_{i=1}^N` be a functional dataset where each :math:`x_i`
is a function from :math:`[a, b]` to :math:`\mathbb{R}` and we have the
measurements of :math:`x_i(t)` at :math:`M_i` points of the domain
:math:`\mathbf{t}_i = (t_{i1}, t_{i2}, \dots, t_{iM_i})`.
That is, we have the irregularly sampled data:
:math:`\{x_i(\mathbf{t}_i))\}_{i=1}^N`, where
:math:`x_i(\mathbf{t}_i) = (x_i(t_{i1}), x_i(t_{i2}), \dots, x_i(t_{iM_i}))`.
Let :math:`\{\phi_b\}_{b=1}^B` be the basis that we want to express the
data in. We denote by :math:`\pmb{\phi}(t)` the vector of evaluations
:math:`(\phi_1(t), \phi_2(t), \dots, \phi_B(t))`.

The mixed effects model assumes the data comes from the model (for each
:math:`1\leq i \leq N` and :math:`a\leq t \leq b`):

.. math::
    x_i(t) = \pmb{\phi}(t)^T (\pmb{\beta} + \pmb{\gamma}_i) + \epsilon_i(t),

where :math:`\pmb{\beta}\in\mathbb{R}^B` is an unknown constant vector
called the fixed effects (we will call it the **mean**);
:math:`\{\pmb{\gamma}_i\}_{i=1}^N\subseteq\mathbb{R}^B` are unknown
random vectors called the **random effects** and they are assumed to be
independent and identically with a normal distribution of mean 0 and
covariance matrix :math:`\pmb{\Gamma}` (which we call **covariance** for
short); and :math:`\epsilon_i(t)` is a random noise term that is assumed to
have a normal distribution with mean 0 and variance :math:`\sigma^2` (which we
call **sigmasq**). We assume that
:math:`\{\epsilon_i(t)\}_{i,t}\cup\{\pmb{\gamma}_i\}_i` are independent.

In order to work with this model and the data available, we denote (for each 
:math:`1 \leq i \leq N`):

.. math::

    \pmb{x}_i = \left(\begin{array}{c}
    x_i(t_{i1}) \\
    x_i(t_{i2}) \\
    \vdots \\
    x_i(t_{iM_i})
    \end{array}\right),
    \qquad
    \pmb{\Phi}_i = \left(\begin{array}{c}
    \pmb{\phi}(t_{i1})^T \\
    \pmb{\phi}(t_{i2})^T \\
    \vdots \\
    \pmb{\phi}(t_{iM_i})^T
    \end{array}\right),
    \qquad
    \pmb{\epsilon}_i = \left(\begin{array}{c}
    \epsilon_i(t_{i1}) \\
    \epsilon_i(t_{i2}) \\
    \vdots \\
    \epsilon_i(t_{iM_i})
    \end{array}\right),

and we have that our model can be written as (for each
:math:`1 \leq i \leq N`):

.. math::

    \pmb{x}_i = \pmb{\Phi}_i (\pmb{\beta} + \pmb{\gamma}_i) + \pmb{\epsilon}_i.

We call :math:`\pmb{x}_i` the *i-th* **values** *vector*, and
:math:`\pmb{\Phi}_i` the *i-th* **basis evaluations** *matrix*.


Fitting the model
-----------------

The model is fitted by maximizing its likelihood to get the MLE (Maximum
Likelihood Estimates) of :math:`\pmb{\beta}`, :math:`\pmb{\Gamma}` and
:math:`\sigma`, and then computing the random effects
(:math:`\{\pmb{\gamma}_i\}_i`) with their least squares linear estimators.

The MLE are computed using either the EM algorithm
(:class:`EMMixedEffectsConverter`,
:footcite:t:`laird+lange+stram_1987_emmixedeffects`), or by minimizing the
profile loglikelihood of the model with generic numerical optimization
(:class:`MinimizeMixedEffectsConverter`, :footcite:t:`Lindstrom_1988`).

The examples
:ref:`sphx_glr_auto_examples_plot_irregular_to_basis_mixed_effects.py` and
:ref:`sphx_glr_auto_examples_plot_irregular_mixed_effects_robustness.py`
illustrate the basic usage of these converters.


References
----------

.. footbibliography::

"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Callable, List, Literal, Protocol

import numpy as np
import scipy
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted
from typing_extensions import Self

from ...misc.lstsq import solve_regularized_weighted_lstsq
from ...representation import FDataBasis, FDataIrregular
from ...representation.basis import Basis
from ...typing._numpy import NDArrayFloat
from ._to_basis import _ToBasisConverter

_SCIPY_MINIMIZATION_METHODS = [
    "BFGS",  # no hessian
    "Powell",  # no jacobian
    "L-BFGS-B",
    "trust-constr",
    "Nelder-Mead",  # no jacobian
    "COBYLA",  # no jacobian
    "SLSQP",
    "CG",  # no hessian
    "TNC",

    # The following methods require jacobian and we do not provide it

    # "trust-ncg",
    # "trust-exact",
    # "trust-krylov",
    # "dogleg",
    # "Newton-CG",
]

_EM_MINIMIZATION_METHODS = [
    "params",
    "squared-error",
    "loglikelihood",
]


def _get_values_list(
    fdatairregular: FDataIrregular,
) -> List[NDArrayFloat]:
    """Get the values vectors for the mixed-effects model.

    Args:
        fdatairregular: Irregular data.

    Returns:
        List of values vectors (one vector per functional datum). If the
        codomain is multidimensional, the vectors are flattened so that each
        measurement's values are contiguous.

    Examples:
        >>> fdata = FDataIrregular(
        ...     start_indices=[0, 1, 5],
        ...     values=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ...     points=list(range(9)),
        ... )
        >>> _get_values_list(fdata)
        [array([ 1]), array([ 2, 3, 4, 5]), array([ 6, 7, 8, 9])]
        >>> fdata_multidim = FDataIrregular(
        ...     start_indices=[0, 1, 3],
        ...     values=np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        ...     points=list(zip(range(5), range(5))),
        ... )
        >>> _get_values_list(fdata_multidim)
        [array([ 1, 2]), array([ 3, 4, 5, 6]), array([ 7,  8,  9, 10])]
    """
    return np.split(
        fdatairregular.values.reshape(-1),
        fdatairregular.start_indices[1:] * fdatairregular.dim_codomain,
    )


def _get_basis_evaluations_list(
    fdatairregular: FDataIrregular,
    basis: Basis,
) -> List[NDArrayFloat]:
    """Get the matrix of basis evaluations for the mixed-effects model.

    Args:
        fdatairregular: Irregular data.
        basis: Basis to evaluate.

    Returns:
        A list of matrices (one matrix per functional datum).

        In the case of 1-dimensional codomain, each matrix is
        of shape (n_points, n_basis), where n_points is the number of points
        of the functional datum and n_basis is the number of basis functions.
        The i-th row of the matrix is the evaluation of the basis functions at
        the i-th point of the functional datum.

        In the case of p-dimensional codomain, each matrix is
        of shape (n_points * dim_codomain, n_basis) (where n_points is the
        number of points of the functional datum).
        The (i*dim_codomain + j)-th row of the matrix is the j-th coordinate of
        the evaluation of the basis functions at the i-th point of the
        functional datum.

    Examples:
        >>> from skfda.representation.basis import (
        ...     MonomialBasis, VectorValuedBasis,
        ... )
        >>> basis = MonomialBasis(n_basis=2)
        >>> fdata = FDataIrregular(
        ...     start_indices=[0, 1, 5],
        ...     values=list(range(7)),
        ...     points=list(range(7)),
        ... )
        >>> _get_basis_evaluations_list(fdata, basis)
        [array([[ 1, 0]]),
         array([[ 1, 1],
                [ 1, 2],
                [ 1, 3],
                [ 1, 4]]),
         array([[ 1, 5],
                [ 1, 6]])]
        >>> monomial_2 = MonomialBasis(n_basis=2, domain_range=(0, 10))
        >>> monomial_3 = MonomialBasis(n_basis=3, domain_range=(0, 10))
        >>> vector_basis = VectorValuedBasis([monomial_2, monomial_3])
        >>> fdata = FDataIrregular(
        ...     start_indices=[0, 1, 4],
        ...     values=list(zip(range(6), range(6))),
        ...     points=list(range(6)),
        ... )
        >>> _get_basis_evaluations_list(fdata, vector_basis)
          [array([[ 1.,  0.,  0.,  0.,  0.],
                  [ 0.,  0.,  1.,  0.,  0.]]), array([[ 1.,  1.,  0.,  0.,  0.],
                  [ 0.,  0.,  1.,  1.,  1.],
                  [ 1.,  2.,  0.,  0.,  0.],
                  [ 0.,  0.,  1.,  2.,  4.],
                  [ 1.,  3.,  0.,  0.,  0.],
                  [ 0.,  0.,  1.,  3.,  9.]]), array([[  1.,  4.,  0.,  0.,  0.],
                  [  0.,  0.,  1.,  4.,  16.],
                  [  1.,  5.,  0.,  0.,   0.],
                  [  0.,  0.,  1.,  5.,  25.]])]
    """
    return np.split(
        basis(fdatairregular.points).reshape(basis.n_basis, -1).T,
        fdatairregular.start_indices[1:] * fdatairregular.dim_codomain,
    )


def _minimize(
    fun: Callable[[NDArrayFloat], float],
    x0: NDArrayFloat,
    minimization_method: str | None = None,
) -> scipy.optimize.OptimizeResult:
    """Minimize a scalar function of one or more variables.

    Uses ``scipy.optimize.minimize``.

    Args:
        fun: Function to minimize.
        x0: Starting point for the minimization.
        minimization_method: ``scipy.optimize.minimize`` method to use for
            minimization.
    """
    if minimization_method is None:
        minimization_method = _SCIPY_MINIMIZATION_METHODS[0]
    elif minimization_method not in _SCIPY_MINIMIZATION_METHODS:
        raise ValueError(
            f"Invalid minimize method: \"{minimization_method}\". "
            f"Supported methods are {_SCIPY_MINIMIZATION_METHODS}."
        )

    result = scipy.optimize.minimize(
        fun=fun,
        x0=x0,
        method=minimization_method,
        options={
            # "disp": True,
            # "maxiter": 1000,
        },
    )
    return result  # even if it failed

def _sum_mahalanobis(
    r_list: List[NDArrayFloat],
    cov_mat_list: List[NDArrayFloat],
    r_list2: List[NDArrayFloat] | None = None,
) -> NDArrayFloat:
    """sum_k ( r_list[k]^T @ cov_mat_list[k]^{-1} @ r_list2[k] )

    Arguments:
        r_list: List of residuals (could be matrices).
        cov_mat_list: List of covariance matrices.
        r_list2: List of residuals (right side) -- if None, r_list is used.

    Returns:
        sum_k ( r_list[k]^T @ cov_mat_list[k]^{-1} @ r_list2[k] )
    """
    if r_list2 is None:
        r_list2 = r_list
    return sum(
        r1.T @ solve_regularized_weighted_lstsq(
            cov_mat,
            r2,
            lstsq_method="cholesky",
        )
        for r1, cov_mat, r2 in zip(r_list, cov_mat_list, r_list2)
    )  # type: ignore


class _MixedEffectsParams(Protocol):
    """Params of the mixed effects model for irregular data."""

    @property
    def covariance(self) -> NDArrayFloat:
        """Covariance of the mixed effects."""

    @property
    def sigmasq(self) -> float:
        """Variance of the noise term."""

    @property
    def covariance_div_sigmasq(self) -> NDArrayFloat:
        """Covariance of the random effects divided by sigmasq."""

    @property
    def mean(self) -> NDArrayFloat:
        """Fixed effects."""


@dataclass(frozen=True)
class _MixedEffectsParamsResult:
    """Result of the fitting of a mixed effects model for irregular data."""
    mean: NDArrayFloat
    covariance: NDArrayFloat
    sigmasq: float

    @property
    def covariance_div_sigmasq(self) -> NDArrayFloat:
        """Covariance of the random effects divided by sigmasq."""
        return self.covariance / self.sigmasq


def _initial_params(
    dim_effects: int,  # TODO add X: FDataIrregular, basis: Basis ?
) -> _MixedEffectsParams:
    """Generic initial parameters."""
    return _MixedEffectsParamsResult(
        mean=np.zeros(dim_effects),
        covariance=np.eye(dim_effects),
        sigmasq=1,
    )


class _MixedEffectsModel:
    """Mixed effects model.

    Class representing the mixed effects model for irregular data.

    Model:

    values[k] = basis_evaluations[k] @ (mean + random_effects[k]) + error[k]

    Args:
        values: List of the values of each curve.
        basis_evaluations: List of the basis evaluations corresponding to the
            points where the curves are evaluated.
    """

    values: List[NDArrayFloat]
    basis_evaluations: List[NDArrayFloat]
    n_measurements: int
    _profile_loglikelihood_additive_constants: float

    def __init__(
        self,
        fdatairregular: FDataIrregular,
        basis: Basis,
    ) -> None:
        self.values = _get_values_list(fdatairregular)
        self.basis_evaluations = _get_basis_evaluations_list(
            fdatairregular, basis,
        )
        self.n_measurements = len(fdatairregular.points)
        self._profile_loglikelihood_additive_constants = (
            + self.n_measurements / 2 * np.log(self.n_measurements)
            - self.n_measurements / 2 * np.log(2 * np.pi)
            - self.n_measurements / 2
        )

    def _dim_effects(self) -> int:
        """Dimension of the mixed and of the fixed effects."""
        return self.basis_evaluations[0].shape[1]

    def partial_residuals(
        self,
        mean: NDArrayFloat,
    ) -> List[NDArrayFloat]:
        """Residuals of the mixed effects model.

        r[k] = value[k] - basis_evaluations[k] @ mean
        """
        return [
            value - basis_evaluation @ mean
            for value, basis_evaluation in zip(
                self.values, self.basis_evaluations,
            )
        ]

    def values_covariances_div_sigmasq(
        self,
        cov_div_sigmasq: NDArrayFloat,
    ) -> List[NDArrayFloat]:
        """Covariance of the values divided by sigmasq.

        values_covariances_div_sigmasq[k] = (
            I + basis_evaluations[k] @ cov_div_sigmasq @ basis_evaluations[k].T
        )

        Used for the model from Lindstrom & Bates (1988).
        """
        return [
            np.eye(basis_evaluation.shape[0])
            + basis_evaluation @ cov_div_sigmasq @ basis_evaluation.T
            for basis_evaluation in self.basis_evaluations
        ]

    def values_covariances(
        self,
        sigmasq: float,
        random_effects_covariance: NDArrayFloat,
    ) -> List[NDArrayFloat]:
        """Covariance of the values.

        values_covariances[k] = (
            sigmasq * I
            + basis_evaluations[k] @ random_effects_covariance
              @ basis_evaluations[k].T
        )

        Args:
            sigmasq: Variance of the noise term.
            random_effects_covariance: Covariance of the random effects.
        """

        return [
            sigmasq * np.eye(basis_evaluation.shape[0])
            + basis_evaluation @ random_effects_covariance @ basis_evaluation.T
            for basis_evaluation in self.basis_evaluations
        ]

    def _random_effects_estimate(
        self,
        random_effects_covariance: NDArrayFloat,
        values_covariances: List[NDArrayFloat],
        partial_residuals: List[NDArrayFloat],
    ) -> NDArrayFloat:
        """Estimates of the random effects (generalized least squares).

        random_effects_estimate[k] = (
            random_effects_covariance @ basis_evaluations[k].T
            @ values_covariances[k]^{-1} @ partial_residuals[k]
        )

        Args:
            random_effects_covariance: Covariance of the random effects.
            values_covariances: Covariances of the values.
            partial_residuals: List of: value - basis_evaluation @ mean.
        """
        return np.array([
            random_effects_covariance @ basis_eval.T @ solve_regularized_weighted_lstsq(
                value_cov,
                r,
                lstsq_method="cholesky",
            )
            for basis_eval, value_cov, r in zip(
                self.basis_evaluations,
                values_covariances,
                partial_residuals,
            )
        ])

    def random_effects_estimate(
        self,
        params: _MixedEffectsParams,
    ) -> NDArrayFloat:
        """Estimates of the random effects (generalized least squares)."""
        return self._random_effects_estimate(
            random_effects_covariance=params.covariance,
            values_covariances=self.values_covariances(
                params.sigmasq, params.covariance,
            ),
            partial_residuals=self.partial_residuals(params.mean),
        )

    def profile_loglikelihood(
        self,
        params: _MixedEffectsParams,
    ) -> float:
        """Profile loglikelihood."""
        partial_residuals = self.partial_residuals(params.mean)
        values_covariances = self.values_covariances_div_sigmasq(
            params.covariance_div_sigmasq,
        )

        # slogdet_V_list = [np.linalg.slogdet(V) for V in V_list]
        # if any(slogdet_V[0] <= 0 for slogdet_V in slogdet_V_list):
        #     return -np.inf
        # TODO remove check sign?

        # sum_logdet_V: float = sum(
        #     slogdet_V[1] for slogdet_V in slogdet_V_list
        # )
        sum_logdet_V: float = sum(
            np.linalg.slogdet(V)[1] for V in values_covariances
        )
        sum_mahalanobis = _sum_mahalanobis(
            partial_residuals, values_covariances,
        )
        log_sum_mahalanobis: float = np.log(sum_mahalanobis)  # type: ignore

        return (
            - sum_logdet_V / 2
            - self.n_measurements / 2 * log_sum_mahalanobis
            + self._profile_loglikelihood_additive_constants
        )

    @property
    def n_samples(self) -> int:
        """Number of samples of the irregular dataset."""
        return len(self.values)


class MixedEffectsConverter(_ToBasisConverter[FDataIrregular], ABC):
    """Abstract class for mixed effects to-basis-converters.

    Args:
        basis: Basis to convert to.

    Parameters:
        result_: Bunch containing the result of the fitting of the model.
            Contains the parameters:

            - model: Fitted mixed effects model.
            - fitted_params: Fitted parameters of the mixed effects model.
            - minimize_result: Result of the ``scipy.optimize.minimize`` call,
                if this function was used.
            - success: Whether the fitting was successful.
            - message: Message of the fitting.
            - nit: Number of iterations of the fitting.
    """

    result_: Bunch  # not None after fitting

    def __init__(
        self,
        basis: Basis,
    ) -> None:
        super().__init__(basis)

    def transform(
        self,
        X: FDataIrregular,
    ) -> FDataBasis:
        """Transform X to FDataBasis using the fitted converter."""
        check_is_fitted(self)

        model = _MixedEffectsModel(X, self.basis)
        mean = self.result_.fitted_params.mean
        gamma_estimates = model.random_effects_estimate(
            self.result_.fitted_params,
        )

        coefficients = mean[np.newaxis, :] + gamma_estimates

        return FDataBasis(
            basis=self.basis,
            coefficients=coefficients,
            dataset_name=X.dataset_name,
            argument_names=X.argument_names,
            coordinate_names=X.coordinate_names,
            sample_names=X.sample_names,
            extrapolation=X.extrapolation,
        )


class MinimizeMixedEffectsConverter(MixedEffectsConverter):
    """Mixed effects to-basis-converter using ``scipy.optimize.minimize``.

    Minimizes the profile loglikelihood of the mixed effects model as proposed
    by :footcite:t:`Lindstrom_1988`.
    """

    @dataclass(frozen=True)
    class Params:
        """Private class for the parameters of the minimization.

        Args:
            sqrt_cov_div_sigmasq:
                (sqrt_cov_div_sigmasq @ sqrt_cov_div_sigmasq.T) is the Cholesky
                decomposition of covariance/sigmasq.
            has_mean: Whether the mean is fixed or estimated with ML estimator.
            mean: Fixed effects (can be none).
            model: Mixed effects model to use for the estimation of the mean in
                case mean=None (will be None otherwise).
        """

        sqrt_cov_div_sigmasq: NDArrayFloat
        _mean: NDArrayFloat | None
        _model: _MixedEffectsModel | None

        def __init__(
            self,
            sqrt_cov_div_sigmasq: NDArrayFloat,
            mean: NDArrayFloat | None,
            model: _MixedEffectsModel | None = None,
        ) -> None:
            if mean is None:
                assert model is not None, "model is required if mean is None"

            # must use object.__setattr__ due to frozen=True
            object.__setattr__(
                self, "sqrt_cov_div_sigmasq", sqrt_cov_div_sigmasq)
            object.__setattr__(self, "_mean", mean)
            object.__setattr__(self, "_model", model)

        @property
        def mean(self) -> NDArrayFloat:
            """Estimate the fixed effects (mean of the coefficients)."""
            if self._mean is not None:
                return self._mean
            assert self._model is not None, "Model is required"
            values_covariances = self._model.values_covariances_div_sigmasq(
                self.covariance_div_sigmasq,
            )
            return solve_regularized_weighted_lstsq(
                coefs=_sum_mahalanobis(
                    self._model.basis_evaluations,
                    values_covariances,
                    self._model.basis_evaluations,
                ),
                result=_sum_mahalanobis(
                    self._model.basis_evaluations,
                    values_covariances,
                    self._model.values,
                ),
                lstsq_method="cholesky",
            )

        @property
        def covariance_div_sigmasq(self) -> NDArrayFloat:
            """Covariance of the random effects divided by sigmasq."""
            return self.sqrt_cov_div_sigmasq @ self.sqrt_cov_div_sigmasq.T

        @property
        def covariance(self) -> NDArrayFloat:
            """Covariance of the random effects."""
            return self.covariance_div_sigmasq * self.sigmasq

        @property
        def sigmasq(self) -> float:
            """Variance of the noise term."""
            assert self._model is not None, "Model is required"
            return _sum_mahalanobis(
                self._model.partial_residuals(self.mean),
                self._model.values_covariances_div_sigmasq(
                    self.covariance_div_sigmasq,
                ),
            ) / self._model.n_measurements  # type: ignore

        @classmethod
        def from_vec(
            cls,
            vec: NDArrayFloat,
            dim_effects: int,
            model: _MixedEffectsModel | None = None,
            has_mean: bool = True,
        ) -> Self:
            """Create Params from vectorized parameters."""
            mean = vec[:dim_effects] if has_mean else None
            sqrt_cov_vec_len = dim_effects * (dim_effects + 1) // 2
            sqrt_cov_div_sigmasq = np.zeros((dim_effects, dim_effects))
            sqrt_cov_div_sigmasq[np.tril_indices(dim_effects)] = (
                vec[-sqrt_cov_vec_len:]
            )
            return cls(
                mean=mean,
                sqrt_cov_div_sigmasq=sqrt_cov_div_sigmasq, 
                model=model,
            )

        def to_vec(self) -> NDArrayFloat:
            """Vectorize parameters."""
            return np.concatenate([
                self._mean if self._mean is not None else np.array([]),
                self.sqrt_cov_div_sigmasq[
                    np.tril_indices(self.sqrt_cov_div_sigmasq.shape[0])
                ],
            ])

    def fit(
        self,
        X: FDataIrregular,
        y: object = None,
        *,
        initial_params: (
            MinimizeMixedEffectsConverter.Params | NDArrayFloat | None
        ) = None,
        minimization_method: str | None = None,
        has_mean: bool = True,
    ) -> Self:
        """Fit the model.

        Args:
            X: irregular data to fit.
            y: ignored.
            initial_params: initial params of the model.
            minimization_method: ``scipy.optimize.minimize`` method to be used
                for the minimization of the loglikelihood of the model.
            has_mean: Whether the mean is a fixed parameter to be optimized or
                estimated with ML estimator from the covariance parameters.

        Returns:
            self after fit
        """
        dim_effects = self.basis.n_basis
        model = _MixedEffectsModel(X, self.basis)
        n_samples = X.n_samples
        if isinstance(initial_params, MinimizeMixedEffectsConverter.Params):
            initial_params_vec = initial_params.to_vec()
        elif initial_params is not None:
            initial_params_vec = initial_params
        else:
            initial_params_generic = _initial_params(dim_effects)
            initial_params_vec = MinimizeMixedEffectsConverter.Params(
                sqrt_cov_div_sigmasq=np.linalg.cholesky(
                    initial_params_generic.covariance,
                ),
                mean=initial_params_generic.mean if has_mean else None,
                model=model,
            ).to_vec()

        if minimization_method is None:
            minimization_method = _SCIPY_MINIMIZATION_METHODS[0]

        def objective_function(params_vec: NDArrayFloat) -> float:
            return - model.profile_loglikelihood(
                params=MinimizeMixedEffectsConverter.Params.from_vec(
                    params_vec,
                    dim_effects,
                    model=self,
                    has_mean=has_mean,
                )
            ) / n_samples

        minimize_result = _minimize(
            fun=objective_function,
            x0=initial_params_vec,
            minimization_method=minimization_method,
        )
        params = MinimizeMixedEffectsConverter.Params.from_vec(
            minimize_result.x,
            dim_effects=dim_effects,
            model=model,
            has_mean=has_mean,
        )
        fitted_params = _MixedEffectsParamsResult(
            mean=params.mean,
            covariance=params.covariance,
            sigmasq=params.sigmasq,
        )
        self.result_ = Bunch(
            model=model,
            fitted_params=fitted_params,
            minimize_result=minimize_result,
            success=minimize_result.success,
            message=minimize_result.message,
            **(
                {"nit": minimize_result.nit}
                if "nit" in minimize_result.keys()
                else {}
            ),
        )

        return self


class EMMixedEffectsConverter(MixedEffectsConverter):
    """Mixed effects to-basis-converter using the EM algorithm.

    Minimizes the profile loglikelihood of the mixed effects model with the EM
    algorithm as proposed by
    :footcite:t:`laird+lange+stram_1987_emmixedeffects`.
    """
    @dataclass(frozen=True)
    class Params:
        """Mixed effects parameters for the EM algorithm."""
        sigmasq: float
        covariance: NDArrayFloat

        @property
        def covariance_div_sigmasq(self) -> NDArrayFloat:
            """Covariance of the mixed effects."""
            return self.covariance / self.sigmasq

        def to_vec(self) -> NDArrayFloat:
            """Vectorize parameters."""
            return np.concatenate([
                np.array([self.sigmasq]),
                self.covariance[np.tril_indices(self.covariance.shape[0])],
            ])

        @classmethod
        def from_vec(
            cls,
            vec: NDArrayFloat,
            dim_effects: int,
        ) -> EMMixedEffectsConverter.Params:
            """Create Params from vectorized parameters."""
            sigmasq = vec[0]
            covariance = np.zeros((dim_effects, dim_effects))
            covariance[np.tril_indices(dim_effects)] = vec[1:]
            return cls(sigmasq=sigmasq, covariance=covariance)

        def len_vec(self) -> int:
            """Length of the vectorized parameters."""
            dim_effects = self.covariance.shape[0]
            return 1 + dim_effects * (dim_effects + 1) // 2

    def _mean(
        self,
        model: _MixedEffectsModel,
        values_covariances_list: List[NDArrayFloat],
    ) -> NDArrayFloat:
        """Return the mean estimate."""
        return solve_regularized_weighted_lstsq(
            coefs=_sum_mahalanobis(
                model.basis_evaluations,
                values_covariances_list,
                model.basis_evaluations,
            ),
            result=_sum_mahalanobis(
                model.basis_evaluations,
                values_covariances_list,
                model.values,
            ),
            lstsq_method="cholesky",
        )

    def _next_params(
        self,
        model: _MixedEffectsModel,
        curr_params: EMMixedEffectsConverter.Params,
        partial_residuals: List[NDArrayFloat],
        values_cov: List[NDArrayFloat],
        random_effects: NDArrayFloat,
    ) -> EMMixedEffectsConverter.Params:
        """Return the next parameters of the EM algorithm."""
        residuals = [
            r - basis_eval @ random_effect
            for r, basis_eval, random_effect in zip(
                partial_residuals, model.basis_evaluations, random_effects,
            )
        ]
        values_cov_inv = [
            np.linalg.pinv(cov, hermitian=True) for cov in values_cov
        ]
        sum_squared_residuals = sum(np.inner(r, r) for r in residuals)
        sum_traces = curr_params.sigmasq * sum(
            # np.trace(np.eye(cov_inv.shape[0]) - params.sigmasq * cov_inv)
            cov_inv.shape[0] - curr_params.sigmasq * np.trace(cov_inv)
            for cov_inv in values_cov_inv
        )
        next_sigmasq = (
            (sum_squared_residuals + sum_traces) / model.n_measurements
        )
        next_covariance = sum(
            np.outer(random_effect, random_effect)
            + curr_params.covariance @ (
                np.eye(curr_params.covariance.shape[1])
                - basis_eval.T @ solve_regularized_weighted_lstsq(
                    Sigma,
                    basis_eval @ curr_params.covariance,
                    lstsq_method="cholesky",
                )
            )
            for basis_eval, Sigma, random_effect in zip(
                model.basis_evaluations,
                values_cov,
                random_effects,
            )
        ) / model.n_samples

        return EMMixedEffectsConverter.Params(
            sigmasq=next_sigmasq,
            covariance=next_covariance,
        )

    def fit(
        self,
        X: FDataIrregular,
        y: object = None,
        *,
        initial_params: (
            EMMixedEffectsConverter.Params | NDArrayFloat | None
        ) = None,
        maxiter: int = 700,
        convergence_criterion: (
            Literal["params", "squared-error", "loglikelihood"] | None
        ) = None,
        rtol: float = 1e-3,
    ) -> Self:
        """Fit the model using the EM algorithm.

        Args:
            X: irregular data to fit.
            y: ignored.
            initial_params: initial params of the model.
            maxiter: maximum number of iterations.
            convergence_criterion: convergence criterion to use when fitting.

                - "params":
                    to use relative differences between parameters
                    (the default).
                - "squared-error":
                    to use relative changes in the squared error
                    of the estimated values with respect to the original data.
                - "loglikelihood":
                    to use relative changes in the loglikelihood.
            rtol: relative tolerance for convergence.

        Returns:
            The converter after fitting.
        """
        model = _MixedEffectsModel(X, self.basis)

        if initial_params is None:
            initial_params_generic = _initial_params(self.basis.n_basis)
            next_params = EMMixedEffectsConverter.Params(
                sigmasq=initial_params_generic.sigmasq,
                covariance=initial_params_generic.covariance,
            )
        elif isinstance(initial_params, np.ndarray):
            next_params = EMMixedEffectsConverter.Params.from_vec(
                initial_params, dim_effects=self.basis.n_basis,
            )
        else:
            next_params = initial_params

        if convergence_criterion is None:
            convergence_criterion = "params"

        if convergence_criterion not in _EM_MINIMIZATION_METHODS:
            raise ValueError(
                "Invalid convergence criterion for the EM algorithm: "
                f"\"{convergence_criterion}\"."
            )

        use_error = convergence_criterion in ("squared-error",)

        if use_error:
            big_values = np.concatenate(model.values)

        converged = False
        convergence_val: NDArrayFloat | float | None = None
        prev_convergence_val: NDArrayFloat | float | None = None
        for iter_number in range(maxiter):
            curr_params = next_params
            values_cov = model.values_covariances(
                curr_params.sigmasq, curr_params.covariance,
            )
            mean = self._mean(model, values_cov)
            partial_residuals = model.partial_residuals(mean)
            random_effects = model._random_effects_estimate(
                curr_params.covariance, values_cov, partial_residuals,
            )
            next_params = self._next_params(
                model=model,
                curr_params=curr_params,
                partial_residuals=partial_residuals,
                values_cov=values_cov,
                random_effects=random_effects,
            )

            if convergence_criterion == "params":
                convergence_val = next_params.to_vec()
            elif convergence_criterion == "squared-error":
                estimates = np.concatenate([  # estimated values
                    basis_eval @ (mean + random_effect)
                    for basis_eval, random_effect in zip(
                        model.basis_evaluations, random_effects,
                    )
                ])
                error = big_values - estimates
                convergence_val = np.inner(error, error)  # sum of squares
            elif convergence_criterion == "loglikelihood":
                convergence_val = model.profile_loglikelihood(
                    _MixedEffectsParamsResult(
                        mean=mean,
                        covariance=next_params.covariance,
                        sigmasq=next_params.sigmasq,
                    )
                )

            if prev_convergence_val is not None:
                converged = np.allclose(
                    convergence_val, prev_convergence_val, rtol=rtol,
                )
                if converged:
                    break

            prev_convergence_val = convergence_val

        if not converged:
            message = f"EM algorithm did not converge ({maxiter=})."
        else:
            message = (
                "EM algorithm converged after "
                f"{iter_number}/{maxiter} iterations."
            )

        final_params = next_params
        values_cov = model.values_covariances(
            curr_params.sigmasq, curr_params.covariance,
        )
        final_mean = self._mean(model, values_cov)
        fitted_params = _MixedEffectsParamsResult(
            mean=final_mean,
            covariance=final_params.covariance,
            sigmasq=final_params.sigmasq,
        )

        self.result_ = Bunch(
            model=model,
            fitted_params=fitted_params,
            success=converged,
            message=message,
            nit=iter_number + 1,
        )
        return self
