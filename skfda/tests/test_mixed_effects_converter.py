"""Tests for the mixed effects to-basis-converter."""
from typing import Any, Literal, Optional, Tuple, Type

import numpy as np
import pytest

from skfda import FDataBasis
from skfda.datasets import irregular_sample
from skfda.misc.scoring import r2_score
from skfda.representation import FDataBasis, FDataIrregular
from skfda.representation.basis import (
    Basis,
    BSplineBasis,
    FourierBasis,
    MonomialBasis,
    TensorBasis,
    VectorValuedBasis,
)
from skfda.representation.conversion._mixed_effects import (
    EMMixedEffectsConverter,
    MinimizeMixedEffectsConverter,
    MixedEffectsConverter,
    _MixedEffectsModel,
)


def test_loglikelihood() -> None:
    """Test loglikelihood function comparing it with Statsmodels' MixedLM.

    Below is the code used to get the values with statsmodels:

    ... from statsmodels.regression.mixed_linear_model import (
    ...     MixedLM, MixedLMParams
    ... )
    ... def _mixedlm_from_me_model(
    ...     me_model: _MixedEffectsModel,
    ...     fdatairregular: FDataIrregular = None,
    ... ) -> MixedLM:
    ...     # Convert MEModel to statsmodels MixedLM.
    ...     # endog = exog @ beta + exog_re @ gamma + epsilon
    ...     endog = np.concatenate(me_model.values)
    ...     exog = exog_re = np.vstack(me_model.basis_evaluations)
    ...     groups = np.repeat(
    ...         np.arange(me_model.n_samples),
    ...         np.ediff1d(
    ...             np.append(
    ...                 fdatairregular.start_indices,
    ...                 len(fdatairregular.points),
    ...             ),
    ...         ),
    ...     )
    ...     mixedlm = MixedLM(
    ...         endog=endog,
    ...         exog=exog,
    ...         exog_re=exog_re,
    ...         groups=groups,
    ...         use_sqrt=True,  # use cholesky decomposition
    ...     )
    ...     # to avoid calling mixedlm.fit:
    ...     mixedlm.cov_pen = None
    ...     mixedlm.reml = False
    ...     return mixedlm
    ... if __name__ == "__main__":
    ...     n_measurements = 200
    ...     n_measurements_per_function = 5
    ...     fdatairregular = FDataIrregular(
    ...         start_indices=list(
    ...             range(0, n_measurements, n_measurements_per_function)
    ...         ),
    ...         values=list(range(n_measurements)),
    ...         points=list(range(n_measurements)),
    ...     )
    ...     basis = FourierBasis(n_basis=5, domain_range=(0, 10))
    ...     model = _MixedEffectsModel(fdatairregular, basis)
    ...     mixedlm = _mixedlm_from_me_model(model, fdatairregular)
    ...     params_len = (
    ...         basis.n_basis + basis.n_basis * (basis.n_basis + 1) // 2
    ...     )
    ...     n_tests = 6
    ...     np.random.seed(100)
    ...     params_list = np.random.rand(n_tests, params_len) * 400
    ...     params_loglike_list = []
    ...
    ...     # assert the loglikelihood is the same for mixedlm and model for
    ...     # n_tests random params
    ...     for params_vec in params_list:
    ...         params = MinimizeMixedEffectsConverter.Params.from_vec(
    ...             params_vec, basis.n_basis, model,
    ...         )
    ...         mixedlmparams = MixedLMParams.from_components(
    ...             fe_params=params.mean,
    ...             cov_re_sqrt=params.sqrt_cov_div_sigmasq,
    ...         )
    ...         mixedlm_loglikelihood = mixedlm.loglike(
    ...             mixedlmparams,
    ...             profile_fe=False,
    ...         )
    ...         params_loglike_list.append((params_vec, mixedlm_loglikelihood))
    ...     print(repr(params_loglike_list))
    """
    n_measurements = 200
    n_measurements_per_function = 5
    fdatairregular = FDataIrregular(
        start_indices=list(
            range(0, n_measurements, n_measurements_per_function)
        ),
        values=list(range(n_measurements)),
        points=list(range(n_measurements)),
    )

    basis = FourierBasis(n_basis=5, domain_range=(0, 10))
    model = _MixedEffectsModel(fdatairregular, basis)

    # These values have been obtained with Statsmodels' MixedLM
    # for the same model
    params_loglike_list = [
        (np.array([
            217.36197672, 111.34775404, 169.8070363, 337.91045293,
            1.88754248, 48.62764831, 268.29963389, 330.34110204,
            54.68263587, 230.03733177, 356.52878172, 83.68084885,
            74.13128782, 43.35075619, 87.87899705, 391.44951388,
            324.67325964, 68.77640509, 326.48989949, 109.62949882,
        ]), -1412.9937447885836),
        (np.array([
            172.68167347, 376.01192785, 327.05975151, 134.44478005,
            70.1641815, 149.13281852, 2.27540294, 100.97054138,
            318.26500339, 6.1019885, 239.53735077, 241.52181562,
            42.05907416, 152.77737798, 14.59042264, 356.16462538,
            392.3683428, 23.97679553, 356.21837789, 230.76059976,
        ]), -1333.6585307493442),
        (np.array([
            296.99187564, 252.07357459, 232.73687696, 8.17565281,
            84.01063107, 217.87395127, 307.64606844, 100.27809166,
            114.35827616, 340.95803514, 390.00259744, 353.9413174,
            143.80313757, 239.54357835, 141.91824466, 136.07608615,
            71.2323958, 95.07768345, 17.94491298, 202.17257185,
        ]), -1270.1651275382442),
        (np.array([
            150.50098172, 237.12216039, 251.97675023, 57.04012578,
            373.53651979, 378.55195232, 240.91866309, 155.10651213,
            145.27520164, 81.73811075, 110.70602456, 98.61435248,
            69.4432007, 386.64387779, 382.80504014, 239.18947373,
            292.52030122, 136.15408913, 36.82224135, 185.39920757,
        ]), -1218.0955679886356),
        (np.array([
            203.4795573, 35.3840692, 211.21408933, 396.8632146,
            158.0143727, 134.23857669, 322.18021493, 301.73959783,
            125.22657664, 253.61467318, 216.16183012, 118.71750035,
            44.31516047, 125.05611915, 182.79165202, 263.57602809,
            101.70300713, 256.44050348, 80.04944289, 263.04992221
        ]), -1231.9562787796967),
        (np.array([
            311.31568618, 311.83935944, 244.13126128, 123.60013941,
            279.09396301, 343.84731829, 250.1295031, 392.96313184,
            390.60005081, 66.67765248, 9.27125459, 64.2978194,
            369.3987301, 381.41993995, 84.39136749, 144.21010033,
            219.75010465, 108.73233967, 184.24064843, 278.46462593
        ]), -1437.3441872940807),
    ]

    for params_vec, mixedlm_loglikelihood in params_loglike_list:
        params = MinimizeMixedEffectsConverter.Params.from_vec(
            params_vec, basis.n_basis, model,
        )
        model_loglikelihood = model.profile_loglikelihood(params)

        assert np.allclose(mixedlm_loglikelihood, model_loglikelihood)


def _create_irregular_samples_with_noise(
    fdatabasis_original: FDataBasis,
    *,
    noise_generate_std: float,
    n_points_range: Tuple[int],
    random_state: np.random.RandomState,
) -> FDataIrregular:
    """Generate samples of functions at random points with Gaussian noise.

    Args:
        fdatabasis_original: Functions to sample.
        noise_generate_std: Standard deviation of the gaussian noise.
        n_points_range: Range of the number of points of each sample.
    """
    n_points_per_sample = random_state.randint(
        *n_points_range, fdatabasis_original.n_samples,
    )
    fdatairregular_no_noise = irregular_sample(
        fdatabasis_original,
        n_points_per_curve=n_points_per_sample,
        random_state=random_state,
    )
    noise_values = np.random.normal(
        0, noise_generate_std, fdatairregular_no_noise.values.shape,
    )
    return FDataIrregular(
        start_indices=fdatairregular_no_noise.start_indices,
        points=fdatairregular_no_noise.points,
        values=fdatairregular_no_noise.values + noise_values,
    )


def _cmp_estimation_with_original(
    fdatabasis_original: FDataBasis,
    noise_generate_std: float,  # to generate the noise
    converter: MixedEffectsConverter,
    fit_kwargs: dict[str, Any],
    check: Literal["r2_score", "coefficients", "both"],
    random_state: np.random.RandomState,
) -> None:
    fdatairregular = _create_irregular_samples_with_noise(
        fdatabasis_original=fdatabasis_original,
        noise_generate_std=noise_generate_std,
        n_points_range=(5, 9),
        random_state=random_state,
    )
    fdatabasis_estimated = converter.fit_transform(
        fdatairregular, **fit_kwargs,
    )

    assert converter.result_.success, "Optimization failed"
    if check in ("r2_score", "both"):
        assert r2_score(fdatabasis_estimated, fdatabasis_original) > 0.9
    if check in ("r2_score", "both"):
        np.allclose(
            fdatabasis_estimated.coefficients,
            fdatabasis_original.coefficients,
        )


def _get_fdatabasis_original(
    basis: Basis,
    n_samples: int,
    random_state: np.random.RandomState,
) -> FDataBasis:
    # These scales are arbitrary
    _scale_cov = 5
    _scale_mean = 10

    n_basis = basis.n_basis
    fe_cov_sqrt = np.zeros((n_basis, n_basis))
    fe_cov_sqrt[np.tril_indices(n_basis)] = random_state.randn(
        n_basis * (n_basis + 1) // 2,
    ) * _scale_cov
    fe_cov = fe_cov_sqrt @ fe_cov_sqrt.T
    mean = random_state.randn(n_basis) * _scale_mean
    return FDataBasis(
        basis=basis,
        coefficients=random_state.multivariate_normal(
            mean=mean, cov=fe_cov, size=n_samples,
        ),
    )


def _test_cmp_with_original_bsplines(
    converter_cls: Type[MixedEffectsConverter],
    fit_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    random_state = np.random.RandomState(238953274)
    if fit_kwargs is None:
        fit_kwargs = {}

    fdatabasis_original = _get_fdatabasis_original(
        basis=BSplineBasis(
            n_basis=3, domain_range=(0, 100), order=2,
        ),
        n_samples=20,
        random_state=random_state,
    )

    _cmp_estimation_with_original(
        fdatabasis_original=fdatabasis_original,
        noise_generate_std=0.1,
        converter=converter_cls(basis=fdatabasis_original.basis),
        fit_kwargs=fit_kwargs,
        check="both",
        random_state=random_state,
    )


def test_cmp_minimize_with_original() -> None:
    """Compare the MinimizeMixedEffects conversion with the original data."""
    _test_cmp_with_original_bsplines(
        converter_cls=MinimizeMixedEffectsConverter,
        fit_kwargs={
            "minimization_method": "Powell",
        }
    )


# This test for EM with simple splines as we have the multidimensional one,
# so as to reduce execution time.
# def test_compare_em_with_original_bsplines() -> None:
#     """Compare the EM conversion with the original data."""
#     _test_cmp_with_original_bsplines(
#         converter_cls=EMMixedEffectsConverter,
#         fit_kwargs={
#             "maxiter": 500,
#             "convergence_criterion": "params",
#             "rtol": 1e-3,
#         }
#     )


def _test_cmp_with_original_multidimensional_data(
    converter_cls: Type[MixedEffectsConverter],
    fit_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    """Compare the conversion with the original data.

    The dimension of the domain and the dimension of the codomain are both 2.
    """
    random_state = np.random.RandomState(238953274)
    if fit_kwargs is None:
        fit_kwargs = {}

    basis_momonial1 = MonomialBasis(n_basis=3, domain_range=(-3, 3))
    basis_fourier1 = FourierBasis(n_basis=1, domain_range=(-3, 3))
    basis_monomial2 = MonomialBasis(n_basis=1, domain_range=(0, 1))
    basis_fourier2 = FourierBasis(n_basis=3, domain_range=(0, 1))

    tensor_basis1 = TensorBasis([basis_momonial1, basis_monomial2])
    tensor_basis2 = TensorBasis([basis_fourier1, basis_fourier2])

    basis = VectorValuedBasis([tensor_basis1, tensor_basis2, tensor_basis1])
    fdatabasis_original = _get_fdatabasis_original(
        basis=basis,
        n_samples=40,
        random_state=random_state,
    )

    _cmp_estimation_with_original(
        fdatabasis_original=fdatabasis_original,
        noise_generate_std=0.1,
        converter=converter_cls(basis=fdatabasis_original.basis),
        fit_kwargs=fit_kwargs,
        check="coefficients",
        random_state=random_state,
    )


def test_cmp_em_with_original_multidimensional_data() -> None:
    """Compare the EM conversion with the original data.

    The dimension of the domain and the dimension of the codomain are both 2.
    """
    _test_cmp_with_original_multidimensional_data(
        converter_cls=EMMixedEffectsConverter,
        fit_kwargs={
            "maxiter": 300,
            "convergence_criterion": "params",
            "rtol": 1e-1,
        }
    )
