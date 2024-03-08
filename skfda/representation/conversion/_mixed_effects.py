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
    List,
    Literal,
    Optional,
    Protocol,
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
    def mean(self) -> NDArrayFloat:
        """Fixed effects."""
        ...

    @property
    def covariance(self) -> NDArrayFloat:
        """Covariance of the mixed effects."""
        ...

    @property
    def covariance_div_sigmasq(self) -> NDArrayFloat:
        """Covariance of the mixed effects."""
        ...

    @property
    def sigmasq(self) -> float:
        """Variance of the residuals."""
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

    def partial_residuals_list(
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

    def values_covariances(
        self,
        params: _MixedEffectsParams,
        div_sigmasq: bool,
    ) -> List[NDArrayFloat]:
        """Covariance of the values.

        If div_sigmasq is False, then the results will be:

        values_covariances[k] = (
            sigmasq * I
            + basis_evaluations[k] @ covariance @ basis_evaluations[k].T
        )

        If div_sigmasq is True, then the results will be (divided by sigmasq):

        values_covariances[k] = (
            I + basis_evaluations[k] @ cov_div_sigmasq @ basis_evaluations[k].T
        )

        div_sigmasq=True for the model from Lindstrom & Bates (1988).
        """
        if div_sigmasq:
            cov_div_sigmasq = params.covariance_div_sigmasq
            return [
                np.eye(basis_evaluation.shape[0])
                + basis_evaluation @ cov_div_sigmasq @ basis_evaluation.T
                for basis_evaluation in self.basis_evaluations
            ]

        sigmasq = params.sigmasq
        params_covariance = params.covariance

        return [
            sigmasq * np.eye(basis_evaluation.shape[0])
            + basis_evaluation @ params_covariance @ basis_evaluation.T
            for basis_evaluation in self.basis_evaluations
        ]

    def mixed_effects_estimate(
        self,
        params: _MixedEffectsParams,
    ) -> NDArrayFloat:
        """Estimates of the mixed effects (generalized least squares)

        mixed_effects_estimate[k] = (
            covariance @ basis_evaluations[k].T
            @ values_covariances[k]^{-1} @ partial_residuals[k]
        )
        """
        covariance = params.covariance
        partial_residuals_list = self.partial_residuals_list(params.mean)
        values_cov_list = self.values_covariances(params, div_sigmasq=False)

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
        params: _MixedEffectsParams,
    ) -> float:
        """Profile loglikelihood."""
        r_list = self.partial_residuals_list(params.mean)
        V_list = self.values_covariances(params, div_sigmasq=True)

        # slogdet_V_list = [np.linalg.slogdet(V) for V in V_list]
        # if any(slogdet_V[0] <= 0 for slogdet_V in slogdet_V_list):
        #     return -np.inf
        # TODO remove check sign?

        # sum_logdet_V: float = sum(
        #     slogdet_V[1] for slogdet_V in slogdet_V_list
        # )
        sum_logdet_V: float = sum(np.linalg.slogdet(V)[1] for V in V_list)
        sum_mahalanobis = _sum_mahalanobis(r_list, V_list)
        log_sum_mahalanobis: float = np.log(sum_mahalanobis)  # type: ignore

        return (
            - sum_logdet_V / 2
            - self._n_measurements / 2 * log_sum_mahalanobis
            + self._profile_loglikelihood_additive_constants
        )


class MixedEffectsConverter(_ToBasisConverter[FDataIrregular]):
    """Mixed effects to-basis-converter."""

    # after fitting:
    fitted_model: Optional[_MixedEffectsModel]
    fitted_params: Optional[_MixedEffectsParamsResult]
    result: Optional[Any]

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
        if self.fitted_params is None:  # or self.model is None:
            raise ValueError("The converter has not been fitted.")

        X_model = _MixedEffectsModel(X, self.basis)
        mean = self.fitted_params.mean
        gamma_estimates = X_model.mixed_effects_estimate(self.fitted_params)

        coefficients = mean[np.newaxis, :] + gamma_estimates

        return FDataBasis(
            basis=self.basis,
            coefficients=coefficients,
        )


class MinimizeMixedEffectsConverter(MixedEffectsConverter):
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

        L: NDArrayFloat
        _mean: Optional[NDArrayFloat]
        _model: Optional[_MixedEffectsModel]

        def __init__(
            self,
            L: NDArrayFloat,
            mean: Optional[NDArrayFloat],
            model: Optional[_MixedEffectsModel] = None,
        ) -> None:
            if mean is None:
                assert model is not None

            # must use object.__setattr__ due to frozen=True
            object.__setattr__(self, "L", L)
            object.__setattr__(self, "_mean", mean)
            object.__setattr__(self, "_model", model)

        @property
        def mean(self) -> NDArrayFloat:
            if self._mean is not None:
                return self._mean
            assert self._model is not None, "model is required"
            values_covariances = self._model.values_covariances(
                self, div_sigmasq=True,
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
            return self.L @ self.L.T

        @property
        def covariance(self) -> NDArrayFloat:
            return self.covariance_div_sigmasq * self.sigmasq

        @property
        def sigmasq(self) -> float:
            assert self._model is not None, "Model is required"
            return _sum_mahalanobis(
                self._model.partial_residuals_list(self.mean),
                self._model.values_covariances(self, div_sigmasq=True),
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
            return cls(mean=mean, L=L, model=model)

        def to_vec(self) -> NDArrayFloat:
            """Vectorize parameters."""
            return np.concatenate([
                self._mean if self._mean is not None else np.array([]),
                self.L[np.tril_indices(self.L.shape[0])]
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
                L=np.linalg.cholesky(initial_params_generic.covariance),
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


class EMMixedEffectsConverter(MixedEffectsConverter):
    """Mixed effects to-basis-converter using the EM algorithm."""
    @dataclass(frozen=True)
    class _Params:
        """Mixed effects parameters for the EM algorithm."""
        sigmasq: float
        covariance: NDArrayFloat

        def covariance_div_sigmasq(self) -> NDArrayFloat:
            """Covariance of the mixed effects."""
            return self.covariance / self.sigmasq

        def to_vec(self) -> NDArrayFloat:
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
            dim_effects = self.covariance.shape[0]
            return 1 + dim_effects * (dim_effects + 1) // 2

    def fit(
        self,
        X: FDataIrregular,
        y: object = None,
        *,
        initial_params: Optional[
            EMMixedEffectsConverter._Params | NDArrayFloat
        ] = None,
        niter: int = 700,
        convergence_criterion: Optional[Literal["params"]] = None,
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
                # - "square-error" to use the square error of the estimates wrt
                #     the original data.
                # - "prop-offset" to use the criteria proposed by Bates &
                #     Watts 1981 (A Relative Offset Convergence Criterion for
                #     Nonlinear Least Squares).
                # - "loglikelihood" to use the loglikelihood.
            rtol: relative tolerance for convergence.

        Returns:
            self after fit
        """
        model = self.model = _MixedEffectsModel(X, self.basis)

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

        assert convergence_criterion in _EM_MINIMIZATION_METHODS

        use_error = convergence_criterion in [
            "square-error",  "square-error-big", "prop-offset",
        ]
        use_big_model = convergence_criterion[-3:] == "big"

        conv_estimate = prev_conv_estimate = None
        converged = False

        for iter_number in range(niter):
            curr_params = next_params
            Sigma_list = self.Sigma_list(model, curr_params)
            beta = self.beta(model, Sigma_list)
            r_list = self.r_list(model, beta)
            random_effects = self._gamma_estimates(
                model, curr_params, r_list, Sigma_list,
            )
            Sigma_inv_list = [
                # _linalg_solve(Sigma, np.eye(Sigma.shape[0]), assume_a="pos")
                np.linalg.pinv(Sigma, hermitian=True)
                for Sigma in Sigma_list
            ]
            next_params = self.next_params(
                model, curr_params, r_list, Sigma_inv_list, Sigma_list, random_effects,
            )

            if use_error:
                me_params = self.meparams_from_emparams(curr_params, beta)
                # error = values - estimates
                error = model.error(me_params, use_big_model)
                if convergence_criterion == "prop-offset":
                    conv_estimate = em_prop_offset_conv_estimate(
                        curr_params, error, model,
                    )
                elif convergence_criterion in [
                    "square-error", "square-error-big",
                ]:
                    conv_estimate = em_square_error_conv_estimate(error)
                else:
                    raise ValueError("Invalid minimization method.")
            elif convergence_criterion == "params":
                conv_estimate = next_params.to_vec()
            elif convergence_criterion == "loglikelihood":
                me_params = self.meparams_from_emparams(curr_params, beta)
                conv_estimate = model.profile_loglikelihood(
                    me_params, has_beta=True,
                )
            else:
                raise ValueError("Invalid minimization method.")
            
            if iter_number > 0:
                if convergence_criterion != "prop-offset":
                    converged = np.allclose(
                        conv_estimate, prev_conv_estimate, rtol=rtol,
                    )
                else:
                    converged = conv_estimate < rtol
                if converged:
                    break

            prev_conv_estimate = conv_estimate

        
        if not converged:
            message = f"EM algorithm did not converge ({niter=})."
            # raise RuntimeError(f"EM algorithm did not converge ({niter=}).")
        else:
            message = (
                "EM algorithm converged after "
                f"{iter_number}/{niter} iterations."
            )

        curr_params = next_params
        Sigma_list = self.Sigma_list(model, curr_params)
        beta = self.beta(model, Sigma_list)
        self.result = {"success": converged, "message": message}
        self.params = MEParams(
            beta=beta,
            model=model,
            L=np.linalg.cholesky(curr_params.Gamma/curr_params.sigmasq),
        )
        self.params_result = MEParamsResult(
            beta=beta,
            Gamma=curr_params.Gamma,
            sigmasq=curr_params.sigmasq,
        )
        return self
