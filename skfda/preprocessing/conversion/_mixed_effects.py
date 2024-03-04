# -*- coding: utf-8 -*-
"""Mixed effects converters.

This module contains the class for converting irregular data to basis
representation using the mixed effects model.

#TODO: Add references ? (laird & ware)

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Optional,
    List,
)

import numpy as np
import scipy
from typing_extensions import Final, Self

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
    "trust-ncg",
    "trust-exact",
    "trust-krylov",
    "TNC",
    "dogleg",
    "Newton-CG",  # requires jacobian
]

_EM_MINIMIZATION_METHODS = [
    "params",
    "square-error",
    "square-error-big",
    "prop-offset",
    "loglikelihood"
]


def _get_values_list(
    fdatairregular: FDataIrregular,
) -> List[NDArrayFloat]:
    assert fdatairregular.dim_domain == 1
    assert fdatairregular.dim_codomain == 1
    return np.split(
        fdatairregular.values.reshape(-1),
        fdatairregular.start_indices[1:],
    )


def _get_basis_evaluations_list(
    fdatairregular: FDataIrregular,
    basis: Basis,
) -> List[NDArrayFloat]:
    assert fdatairregular.dim_domain == 1
    assert fdatairregular.dim_codomain == 1
    return np.split(
        basis(fdatairregular.points)[:, :, 0].T,
        fdatairregular.start_indices[1:],
    )


def _minimize(
    fun: Callable[[NDArrayFloat], float],
    x0: NDArrayFloat,
    minimization_methods: str | List[str] | None = None,
) -> scipy.optimize.OptimizeResult:
    """Minimize a scalar function of one or more variables."""
    if isinstance(minimization_methods, str):
        minimization_methods = [minimization_methods]

    if minimization_methods is None:
        minimization_methods = _SCIPY_MINIMIZATION_METHODS
    else:
        for method in minimization_methods:
            if method not in _SCIPY_MINIMIZATION_METHODS:
                raise ValueError(f"Invalid method: \"{method}\".")

    for method in minimization_methods:
        result = scipy.optimize.minimize(
            fun=fun,
            x0=x0,
            method=method,
            options={
                # "disp": True,
                # "maxiter": 1000,
            },
        )
        if result.success is True:
            # print(
            #   f"[MEEstimator info]: Minimization method {method} succeeded.",
            # )
            return result
        # else:
        #     print(f"[MEEstimator info]: Minimization method {method} failed.")
    return result  # even if it failed


def _linalg_solve(
    a: NDArrayFloat, b: NDArrayFloat, *, assume_a: str = 'gen'
) -> NDArrayFloat:
    """Solve a linear system of equations: a @ x = b"""
    try:
        return scipy.linalg.solve(a=a, b=b, assume_a=assume_a)  # type: ignore
    except scipy.linalg.LinAlgError:
        # TODO: is the best way to handle this ?
        # print("Warning: scipy.linalg.solve failed, using scipy.linalg.lstsq")
        return scipy.linalg.lstsq(a=a, b=b)[0]  # type: ignore


def sum_mahalanobis(
    r_list: List[NDArrayFloat],
    cov_mat_list: List[NDArrayFloat],
    r_list2: Optional[List[NDArrayFloat]] = None,
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
        r1.T @ _linalg_solve(cov_mat, r2, assume_a="pos")
        for r1, cov_mat, r2 in zip(r_list, cov_mat_list, r_list2)
    )  # type: ignore


class _MixedEffectsCovParams(ABC):
    """Covariance params of the mixed effects model for irregular data."""

    @abstractmethod
    def covariance(self) -> NDArrayFloat:
        """Covariance of the mixed effects."""
        pass

    @abstractmethod
    def covariance_div_sigmasq(self) -> NDArrayFloat:
        """Covariance of the mixed effects."""
        pass

    @abstractmethod
    def sigmasq(self) -> float:
        """Variance of the residuals."""
        pass


class _MixedEffectsParams(_MixedEffectsCovParams):
    """Params of the mixed effects model for irregular data."""

    @abstractmethod
    def mean(self) -> NDArrayFloat:
        """Fixed effects."""
        pass


@dataclass
class _MixedEffectsParamsResult(_MixedEffectsParams):
    """Basic mixed effects params implementation."""
    _mean: NDArrayFloat
    _covariance: NDArrayFloat
    _sigmasq: float

    def covariance(self) -> NDArrayFloat:
        return self._covariance

    def mean(self) -> NDArrayFloat:
        return self._mean

    def sigmasq(self) -> float:
        return self._sigmasq

    def covariance_div_sigmasq(self) -> NDArrayFloat:
        return self._covariance / self._sigmasq


class _MinimizeMixedEffectsParams(_MixedEffectsParams):
    """Default class to represent the mixed effects parameters.

    Used to implement the optimization of loglikelihood as suggested in
    Mary J. Lindstrom & Douglas M. Bates (1988).

    Args:
        _L: (_L @ _L.T) is the Cholesky decomposition of covariance/sigmasq.
        _has_mean: Whether the mean is fixed or estimated with ML estimator.
        _mean: Fixed effects (will be none iff _has_mean=False).
        _model: Mixed effects model to use for the estimation of the mean in
            case _has_mean=False (will be None otherwise).
    """

    _L: NDArrayFloat
    _mean: Optional[NDArrayFloat]
    _has_mean: bool
    _model: Optional[_MixedEffectsModel]

    def __init__(
        self,
        L: NDArrayFloat,
        mean: Optional[NDArrayFloat],
        has_mean: bool = True,
        model: Optional[_MixedEffectsModel] = None,
    ) -> None:
        self._L = L
        self._mean = mean
        self._has_mean = has_mean
        self._model = model
        if has_mean:
            assert mean is not None
        else:
            assert mean is None
            assert model is not None

    def mean(self) -> NDArrayFloat:
        if self._has_mean:
            assert self._mean is not None  # TODO: remove
            return self._mean
        assert self._model is not None, "model is required"
        values_covariances = self._model._values_covariances(
            self, div_sigmasq=True,
        )
        return _linalg_solve(
            a=sum_mahalanobis(
                self._model.basis_evaluations,
                values_covariances,
                self._model.basis_evaluations,
            ),
            b=sum_mahalanobis(
                self._model.basis_evaluations,
                values_covariances,
                self._model.values,
            ),
            assume_a="pos",
        )

    def covariance_div_sigmasq(self) -> NDArrayFloat:
        return self._L @ self._L.T

    def covariance(self) -> NDArrayFloat:
        return self.covariance_div_sigmasq() * self.sigmasq()

    def sigmasq(self) -> float:
        assert self._model is not None, "Model is required"
        return sum_mahalanobis(
            self._model._partial_residuals_list(self),
            self._model._values_covariances(self, div_sigmasq=True),
        ) / self._model._n_measurements  # type: ignore

    @classmethod
    def from_vec(
        cls,
        vec: NDArrayFloat,
        dim_effects: int,
        model: Optional[_MixedEffectsModel] = None,
        has_mean: bool = True,
    ) -> Self:
        """Create Params from vectorized parameters."""
        mean = vec[:dim_effects] if has_mean else None
        L_vec_len = dim_effects * (dim_effects + 1) // 2
        L = np.zeros((dim_effects, dim_effects))
        L[np.tril_indices(dim_effects)] = vec[-L_vec_len:]
        return cls(mean=mean, L=L, model=model, has_mean=has_mean)

    def to_vec(self) -> NDArrayFloat:
        """Vectorize parameters."""
        return np.concatenate([
            self._mean if self._has_mean else np.array([]),
            self._L[np.tril_indices(self._L.shape[0])]
        ])

    @classmethod
    def initial_params(
        cls,
        dim_effects: int,
        has_mean: bool,
        model: _MixedEffectsModel,
    ) -> Self:
        """Generic initial parameters ."""
        return cls(
            mean=np.zeros(dim_effects) if has_mean else None,
            L=np.eye(dim_effects),
            has_mean=has_mean,
            model=model,
        )


class _EMMixedEffectsParams(_MixedEffectsCovParams):
    """Mixed effects parameters for the EM algorithm."""
    _sigmasq: float
    _covariance: NDArrayFloat
    # _model: _MixedEffectsModel

    def __init__(
        self,
        sigmasq: float,
        covariance: NDArrayFloat,
        # model: _MixedEffectsModel,
    ) -> None:
        self._sigmasq = sigmasq
        self._covariance = covariance
        # self._model = model

    def covariance(self) -> NDArrayFloat:
        """Covariance of the mixed effects."""
        return self._covariance

    def covariance_div_sigmasq(self) -> NDArrayFloat:
        """Covariance of the mixed effects."""
        return self._covariance / self._sigmasq

    def sigmasq(self) -> float:
        """Variance of the residuals."""
        return self._sigmasq

    def mean(self) -> NDArrayFloat:
        raise NotImplementedError()

    def to_vec(self) -> NDArrayFloat:
        return np.concatenate([
            np.array([self._sigmasq]),
            self._covariance[np.tril_indices(self._covariance.shape[0])],
        ])

    def len_vec(self) -> int:
        dim_effects = self._covariance.shape[0]
        return 1 + dim_effects * (dim_effects + 1) // 2


class _MixedEffectsModel:
    """Mixed effects model.

    Class representing the mixed effects model for irregular data.

    Model:


    values[k] = basis_evaluations[k] @ (mean + mixed_effects[k]) + error[k]

    Args:
        values: List of the values of each curve.
        basis_evaluations: List of the basis evaluations corresponding to the
            points where the curves are evaluated.
    """

    values: List[NDArrayFloat]
    basis_evaluations: List[NDArrayFloat]
    _n_measurements: int
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
        self._n_measurements = len(fdatairregular.points)
        self._profile_loglikelihood_additive_constants = (
            + self._n_measurements / 2 * np.log(self._n_measurements)
            - self._n_measurements / 2 * np.log(2 * np.pi)
            - self._n_measurements / 2
        )

    def _dim_effects(self) -> int:
        """Dimension of the mixed and of the fixed effects."""
        return self.basis_evaluations[0].shape[1]

    def _partial_residuals_list(
        self,
        params: _MixedEffectsParams,
    ) -> List[NDArrayFloat]:
        """Residuals of the mixed effects model.

        r[k] = value[k] - basis_evaluations[k] @ mean
        """
        mean = params.mean()
        return [
            value - basis_evaluation @ mean
            for value, basis_evaluation in zip(
                self.values, self.basis_evaluations,
            )
        ]

    def _values_covariances(
        self,
        params: _MixedEffectsParams,
        div_sigmasq: bool,
    ) -> List[NDArrayFloat]:
        """Covariance of the values.

        values_covariances[k] = (
            sigmasq * I
            + basis_evaluations[k] @ covariance @ basis_evaluations[k].T
        )

        If div_sigmasq is True, then the results will be divided by sigmasq.
        div_sigmasq = True for the model from Lindstrom & Bates (1988).
        """
        if div_sigmasq:
            cov_div_sigmasq = params.covariance_div_sigmasq()
            return [
                np.eye(basis_evaluation.shape[0])
                + basis_evaluation @ cov_div_sigmasq @ basis_evaluation.T
                for basis_evaluation in self.basis_evaluations
            ]

        sigmasq = params.sigmasq()
        params_covariance = params.covariance()

        return [
            sigmasq * np.eye(basis_evaluation.shape[0])
            + basis_evaluation @ params_covariance @ basis_evaluation.T
            for basis_evaluation in self.basis_evaluations
        ]

    def _mixed_effects_estimate(
        self,
        params: _MixedEffectsParams,
    ) -> NDArrayFloat:
        """Estimates of the mixed effects (generalized least squares)

        mixed_effects_estimate[k] = (
            covariance @ basis_evaluations[k].T
            @ values_covariances[k]^{-1} @ partial_residuals[k]
        )
        """
        covariance = params.covariance()
        partial_residuals_list = self._partial_residuals_list(params)
        values_cov_list = self._values_covariances(params, div_sigmasq=False)

        return np.array([
            covariance @ basis_eval.T @ _linalg_solve(
                value_cov, r, assume_a="pos",
            )
            for basis_eval, value_cov, r in zip(
                self.basis_evaluations,
                values_cov_list,
                partial_residuals_list,
            )
        ])

    def profile_loglikelihood(
        self,
        params: _MinimizeMixedEffectsParams | NDArrayFloat,
        has_mean: bool = True,
    ) -> float:
        """Profile loglikelihood."""
        if isinstance(params, np.ndarray):
            params = _MinimizeMixedEffectsParams.from_vec(
                params, self._dim_effects(), model=self, has_mean=has_mean,
            )

        r_list = self._partial_residuals_list(params)
        V_list = self._values_covariances(params, div_sigmasq=True)

        # slogdet_V_list = [np.linalg.slogdet(V) for V in V_list]
        # if any(slogdet_V[0] <= 0 for slogdet_V in slogdet_V_list):
        #     return -np.inf
        # TODO remove check sign?

        # sum_logdet_V: float = sum(
        #     slogdet_V[1] for slogdet_V in slogdet_V_list
        # )
        sum_logdet_V: float = sum(np.linalg.slogdet(V)[1] for V in V_list)
        sum_mahalanobis_ = sum_mahalanobis(r_list, V_list)
        log_sum_mahalanobis: float = np.log(sum_mahalanobis_)  # type: ignore

        return (
            - sum_logdet_V / 2
            - self._n_measurements / 2 * log_sum_mahalanobis
            + self._profile_loglikelihood_additive_constants
        )


class MixedEffectsConverter(_ToBasisConverter[FDataIrregular]):
    """Mixed effects to-basis-converter."""

    basis: Basis

    # after fitting:
    fitted_model: Optional[_MixedEffectsModel]
    fitted_params: Optional[_MixedEffectsParams]
    result: Any

    def __init__(
        self,
        basis: Basis,
    ) -> None:
        self.basis = basis
        self.fitted_model = None
        self.fitted_params = None

    def transform(
        self,
        X: FDataIrregular,
    ) -> FDataBasis:
        if self.fitted_params is None:  # or self.model is None:
            raise ValueError("The converter has not been fitted.")

        X_model = _MixedEffectsModel(X, self.basis)
        mean = self.fitted_params.mean()
        gamma_estimates = X_model._mixed_effects_estimate(self.fitted_params)

        coefficients = mean[np.newaxis, :] + gamma_estimates

        return FDataBasis(
            basis=self.basis,
            coefficients=coefficients,
        )


class MinimizeMixedEffectsConverter(MixedEffectsConverter):
    """Mixed effects to-basis-converter using scipy.optimize.

    Minimizes the profile loglikelihood of the mixed effects model as proposed
    by Lindstrom & Bates (1988).
    """

    def fit(
        self,
        X: FDataIrregular,
        y: object = None,
        *,
        initial_params: Optional[
            _MinimizeMixedEffectsParams | NDArrayFloat
        ] = None,
        minimization_method: Optional[str] = None,
        has_mean: bool = True,
    ) -> Self:
        """Fit the model.

        Args:
            X: irregular data to fit.
            y: ignored.
            initial_params: initial params of the model.
            minimization_methods: scipy.optimize.minimize method to be used for
                the minimization of the loglikelihood of the model.

        Returns:
            self after fit
        """
        dim_effects = self.basis.n_basis
        if isinstance(initial_params, _MinimizeMixedEffectsParams):
            # assert has_beta == initial_params.has_beta
            initial_params_vec = initial_params.to_vec()
        elif initial_params is not None:
            initial_params_vec = initial_params
        else:
            initial_params_vec = _MinimizeMixedEffectsParams.initial_params(
                dim_effects=dim_effects, has_mean=has_mean, model=self,
            ).to_vec()

        if minimization_method is None:
            minimization_method = _SCIPY_MINIMIZATION_METHODS[0]

        model = _MixedEffectsModel(X, self.basis)
        n_samples = X.n_samples

        def objective_function(params: NDArrayFloat) -> float:
            return - model.profile_loglikelihood(
                params, has_mean=has_mean,
            ) / n_samples

        self.result = _minimize(
            fun=objective_function,
            x0=initial_params_vec,
            minimization_methods=minimization_method,
        )
        self.fitted_model = model
        params = _MinimizeMixedEffectsParams.from_vec(
            self.result.x,
            dim_effects=dim_effects,
            model=model,
            has_mean=has_mean,
        )
        self.fitted_params = _MixedEffectsParamsResult(
            _mean=params.mean(),
            _covariance=params.covariance(),
            _sigmasq=params.sigmasq(),
        )

        return self
