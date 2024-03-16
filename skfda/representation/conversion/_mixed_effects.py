# -*- coding: utf-8 -*-
"""Mixed effects converters.

This module contains the class for converting irregular data to basis
representation using the mixed effects model.

#TODO: Add references ? (laird & ware)

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
)

import numpy as np
import scipy
from typing_extensions import Self

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
    "squared-error",
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
                raise ValueError(f"Invalid minimize method: \"{method}\".")

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
            break
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


def _sum_mahalanobis(
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


class _MixedEffectsParams(Protocol):
    """Params of the mixed effects model for irregular data."""

    @property
    def covariance(self) -> NDArrayFloat:
        """Covariance of the mixed effects."""
        ...

    @property
    def sigmasq(self) -> float:
        """Variance of the residuals."""
        ...

    @property
    def covariance_div_sigmasq(self) -> NDArrayFloat:
        """Covariance of the mixed effects."""
        ...

    @property
    def mean(self) -> NDArrayFloat:
        """Fixed effects."""
        ...


@dataclass(frozen=True)
class _MixedEffectsParamsResult:
    """Result of the fitting of a mixed effects model for irregular data."""
    mean: NDArrayFloat
    covariance: NDArrayFloat
    sigmasq: float

    @property
    def covariance_div_sigmasq(self) -> NDArrayFloat:
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
            sigmasq: Variance of the residuals.
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
        """Estimates of the random effects (generalized least squares)

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
            random_effects_covariance @ basis_eval.T @ _linalg_solve(
                value_cov, r, assume_a="pos",
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


class _MixedEffectsConverter(_ToBasisConverter[FDataIrregular]):
    """Mixed effects to-basis-converter."""

    # after fitting:
    fitted_model: Optional[_MixedEffectsModel]
    fitted_params: Optional[_MixedEffectsParams]
    result: Optional[Dict[str, Any] | scipy.optimize.OptimizeResult]

    def __init__(
        self,
        basis: Basis,
    ) -> None:
        self.fitted_model = None
        self.fitted_params = None
        self.result = None
        super().__init__(basis)

    def transform(
        self,
        X: FDataIrregular,
    ) -> FDataBasis:
        """Transform to FDataBasis using the fitted converter."""
        if self.fitted_params is None:  # or self.model is None:
            raise ValueError("The converter has not been fitted.")

        model = _MixedEffectsModel(X, self.basis)
        mean = self.fitted_params.mean
        gamma_estimates = model.random_effects_estimate(self.fitted_params)

        coefficients = mean[np.newaxis, :] + gamma_estimates

        return FDataBasis(
            basis=self.basis,
            coefficients=coefficients,
        )


class MinimizeMixedEffectsConverter(_MixedEffectsConverter):
    """Mixed effects to-basis-converter using scipy.optimize.

    Minimizes the profile loglikelihood of the mixed effects model as proposed
    by Mary J. Lindstrom & Douglas M. Bates (1988).
    """

    @dataclass(frozen=True)
    class _Params:
        """Private class for the parameters of the minimization.
        Args:
            L: (_L @ _L.T) is the Cholesky decomposition of covariance/sigmasq.
            has_mean: Whether the mean is fixed or estimated with ML estimator.
            mean: Fixed effects (can be none).
            model: Mixed effects model to use for the estimation of the mean in
                case mean=None (will be None otherwise).
        """

        sqrt_cov_div_sigmasq: NDArrayFloat
        _mean: Optional[NDArrayFloat]
        _model: Optional[_MixedEffectsModel]

        def __init__(
            self,
            sqrt_cov_div_sigmasq: NDArrayFloat,
            mean: Optional[NDArrayFloat],
            model: Optional[_MixedEffectsModel] = None,
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
            if self._mean is not None:
                return self._mean
            assert self._model is not None, "Model is required"
            values_covariances = self._model.values_covariances_div_sigmasq(
                self.covariance_div_sigmasq,
            )
            return _linalg_solve(
                a=_sum_mahalanobis(
                    self._model.basis_evaluations,
                    values_covariances,
                    self._model.basis_evaluations,
                ),
                b=_sum_mahalanobis(
                    self._model.basis_evaluations,
                    values_covariances,
                    self._model.values,
                ),
                assume_a="pos",
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
            """Variance of the residuals."""
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
            model: Optional[_MixedEffectsModel] = None,
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
        initial_params: Optional[
            MinimizeMixedEffectsConverter._Params | NDArrayFloat
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
            has_mean: Whether the mean is a fixed parameter to be optimized or
                estimated with ML estimator from the covariance parameters.

        Returns:
            self after fit
        """
        dim_effects = self.basis.n_basis
        model = _MixedEffectsModel(X, self.basis)
        n_samples = X.n_samples
        if isinstance(initial_params, MinimizeMixedEffectsConverter._Params):
            initial_params_vec = initial_params.to_vec()
        elif initial_params is not None:
            initial_params_vec = initial_params
        else:
            initial_params_generic = _initial_params(dim_effects)
            initial_params_vec = MinimizeMixedEffectsConverter._Params(
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
                params=MinimizeMixedEffectsConverter._Params.from_vec(
                    params_vec, dim_effects, model=self, has_mean=has_mean,
                )
            ) / n_samples

        self.result = _minimize(
            fun=objective_function,
            x0=initial_params_vec,
            minimization_methods=minimization_method,
        )
        self.fitted_model = model
        params = MinimizeMixedEffectsConverter._Params.from_vec(
            self.result.x,
            dim_effects=dim_effects,
            model=model,
            has_mean=has_mean,
        )
        self.fitted_params = _MixedEffectsParamsResult(
            mean=params.mean,
            covariance=params.covariance,
            sigmasq=params.sigmasq,
        )

        return self


class EMMixedEffectsConverter(_MixedEffectsConverter):
    """Mixed effects to-basis-converter using the EM algorithm."""
    @dataclass(frozen=True)
    class _Params:
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
        ) -> EMMixedEffectsConverter._Params:
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
        """Return the beta estimate."""
        return _linalg_solve(
            a=_sum_mahalanobis(
                model.basis_evaluations,
                values_covariances_list,
                model.basis_evaluations,
            ),
            b=_sum_mahalanobis(
                model.basis_evaluations,
                values_covariances_list,
                model.values,
            ),
            assume_a="pos",
        )

    def _next_params(
        self,
        model: _MixedEffectsModel,
        curr_params: EMMixedEffectsConverter._Params,
        partial_residuals: List[NDArrayFloat],
        values_cov: List[NDArrayFloat],
        random_effects: NDArrayFloat,
    ) -> EMMixedEffectsConverter._Params:
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
                - basis_eval.T @ _linalg_solve(
                    Sigma, basis_eval @ curr_params.covariance, assume_a="pos",
                )
            )
            for basis_eval, Sigma, random_effect in zip(
                model.basis_evaluations, values_cov, random_effects,
            )
        ) / model.n_samples

        return EMMixedEffectsConverter._Params(
            sigmasq=next_sigmasq,
            covariance=next_covariance,
        )

    def fit(
        self,
        X: FDataIrregular,
        y: object = None,
        *,
        initial_params: Optional[
            EMMixedEffectsConverter._Params | NDArrayFloat
        ] = None,
        maxiter: int = 700,
        convergence_criterion: Optional[
            Literal["params", "squared-error", "loglikelihood"]
        ] = None,
        rtol: float = 1e-3,
    ) -> Self:
        """Fit the model using the EM algorithm.

        Args:
            X: irregular data to fit.
            y: ignored.
            initial_params: initial params of the model.
            niter: maximum number of iterations.
            convergence_criterion: convergence criterion to use when fitting.
                - "params" to use relative differences between parameters.
                - "squared-error" to userelative changes in the squared error
                    of the estimated values with respect to the original data.
                - "loglikelihood" to use relative changes in the loglikelihood.
                # - "prop-offset" to use the criteria proposed by Bates &
                #     Watts 1981 (A Relative Offset Convergence Criterion for
                #     Nonlinear Least Squares).
            rtol: relative tolerance for convergence.

        Returns:
            The converter after fitting.
        """
        model = _MixedEffectsModel(X, self.basis)

        if initial_params is None:
            initial_params_generic = _initial_params(self.basis.n_basis)
            next_params = EMMixedEffectsConverter._Params(
                sigmasq=initial_params_generic.sigmasq,
                covariance=initial_params_generic.covariance,
            )
        elif isinstance(initial_params, np.ndarray):
            next_params = EMMixedEffectsConverter._Params.from_vec(
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
        convergence_val: Optional[NDArrayFloat | float] = None
        prev_convergence_val: Optional[NDArrayFloat | float] = None
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

        self.result = {
            "success": converged,
            "message": message,
            "nit": iter_number,
        }
        self.fitted_model = model

        final_params = next_params
        values_cov = model.values_covariances(
            curr_params.sigmasq, curr_params.covariance,
        )
        final_mean = self._mean(model, values_cov)
        self.fitted_params = _MixedEffectsParamsResult(
            mean=final_mean,
            covariance=final_params.covariance,
            sigmasq=final_params.sigmasq,
        )
        return self
