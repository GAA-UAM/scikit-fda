"""Tests for the mixed effects to-basis-converter."""
import pytest
import numpy as np
import numpy.typing as npt
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)
import matplotlib.pyplot as plt

from skfda import FDataBasis
from skfda.misc.scoring import r2_score
from skfda.representation import (
    FDataBasis,
    FDataIrregular,
)
from skfda.typing._numpy import (NDArrayFloat, NDArrayInt)
from skfda.representation.basis import (
    BSplineBasis,
    FourierBasis,
)
from skfda.representation.conversion._mixed_effects import (
    MinimizeMixedEffectsConverter,
    MixedEffectsConverter,
    EMMixedEffectsConverter,
    _get_values_list,
    _get_basis_evaluations_list,
    _MixedEffectsModel,
)

_fdatairregular = FDataIrregular(
    start_indices=[0, 1, 5],
    values=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
    points=list(range(9)),
)


def test_loglikelihood() -> None:
    """Test loglikelihood function comparing it with Statsmodels' MixedLM."""
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


def test_values_list() -> None:
    """Test conversion from FDataIrregular to ME model: values."""
    fdatairregular = _fdatairregular
    x_list = _get_values_list(fdatairregular)
    expected_x_list = [
        np.array([1]),
        np.array([2, 3, 4, 5]),
        np.array([6, 7, 8, 9]),
    ]
    for x, expected_x in zip(x_list, expected_x_list):
        assert np.all(x == expected_x)


def test_basis_evaluations_list() -> None:
    """Test conversion from FDataIrregular to ME model: basis evaluations."""
    fdatairregular = _fdatairregular
    basis = FourierBasis(n_basis=3, domain_range=(0, 10))
    phi_list = _get_basis_evaluations_list(fdatairregular, basis)

    def eval_basis(x: float) -> npt.NDArray[np.float_]:
        return basis(x).reshape(-1)

    expected_phi = [
        np.array([eval_basis(0)]),
        np.array([eval_basis(j) for j in [1, 2, 3, 4]]),
        np.array([eval_basis(j) for j in [5, 6, 7, 8]]),
    ]

    for phi, expected_phi in zip(phi_list, expected_phi):
        np.testing.assert_allclose(phi, expected_phi)


def _create_irregular_samples(
    funcs: Iterable[
        Callable[[npt.NDArray[np.float_]], npt.NDArray[np.float_]]
    ],
    points: npt.NDArray[np.float_],
    noise_generate_std: float,
    *,
    start_indices: NDArrayInt | None = None,
    n_points: int | None = None,
) -> FDataIrregular:
    """Generate samples of functions at points with gaussian noise.

    Args:
        funcs: Functions to sample.
        points: Points where to sample.
        noise_generate_std: Standard deviation of the gaussian noise.
        start_indices: Start indices of each sample.
        n_points: Number of points of each sample. If not None, start_indices
            is ignored.
    """
    if n_points is not None:
        start_indices = np.arange(0, len(points), n_points)
    elif start_indices is None:
        raise ValueError("Either n_points or start_indices must be provided")
    fun_points = np.split(points, start_indices[1:])
    fun_values = np.concatenate([
        func(point) for func, point in zip(funcs, fun_points)
    ]).reshape((-1, 1))
    noise_values = np.random.normal(
        0, noise_generate_std, len(fun_values),
    ).reshape((-1, 1))
    return FDataIrregular(
        start_indices=start_indices,
        points=points,
        values=fun_values + noise_values,
    )


def _get_points(
    domain_range: Tuple[float, float],
    n_points: int,
    n_samples: int,
    type_gen_points: int,
) -> npt.NDArray[np.float_]:
    n = type_gen_points
    tot_n_points = n_points * n_samples
    domain_split = np.linspace(*domain_range, n + 1)
    domains = list(zip(domain_split[:-1], domain_split[1:]))
    points_list = [
        np.random.uniform(
            domain[0] - 0.6 * (domain[1] - domain[0]),
            domain[1] + 0.6 * (domain[1] - domain[0]),
            size=tot_n_points // n)
        for domain in domains
    ]
    ret_value = np.concatenate(points_list).reshape((-1, 1))[:tot_n_points]

    return (
        ret_value
        * (ret_value >= domain_range[0])
        * (ret_value <= domain_range[1])
        + domain_range[0] * (ret_value < domain_range[0])
        + domain_range[1] * (ret_value > domain_range[1])
    )


# def __test_simple_conversion() -> None:
#     """Visual test."""
#     _max_val = 10
#     _domain_range = (0, 10)
#     n_points = 6
#     n_basis = 5
#     n_samples = 50
#     points = _get_points(_domain_range, n_points, n_samples, 9)

#     basis = FourierBasis(n_basis=n_basis, domain_range=_domain_range)
#     # BSplineBasis(
#     #     n_basis=n_basis, domain_range=_domain_range, order=n_basis - 1,
#     # )

#     sigma = 0.3
#     Gamma_sqrt = np.zeros((n_basis, n_basis))
#     Gamma_sqrt[np.tril_indices(n_basis)] = np.random.rand(
#         n_basis * (n_basis + 1) // 2,
#     ) * _max_val
#     Gamma = Gamma_sqrt @ Gamma_sqrt.T
#     beta = np.random.rand(n_basis) * _max_val
#     fdatabasis_original = FDataBasis(
#         basis=basis,
#         coefficients=np.random.multivariate_normal(
#             mean=beta, cov=Gamma, size=n_samples,
#         ),
#     )

#     def fun(i: int) -> Callable[[NDArrayFloat], NDArrayFloat]:
#         def fi(x: NDArrayFloat) -> NDArrayFloat:
#             return fdatabasis_original[i](x).reshape(x.shape)
#         return fi

#     funcs = [fun(i) for i in range(n_samples)]

#     fdatairregular = _create_irregular_samples(
#         funcs=funcs,
#         n_points=n_points,
#         points=points,
#         noise_generate_std=sigma,
#     )
#     converter = MinimizeMixedEffectsConverter(basis)
#     fdatabasis_estimated = converter.fit_transform(fdatairregular)
#     fdatabasis_basic = fdatairregular.to_basis(basis)
#     if True:
#         _ = plt.figure(figsize=(15, 6))

#         axes = plt.subplot(2, 2, 1)
#         plt.title("Original data")
#         fdatairregular[:5].plot(axes=axes)
#         left, right = plt.ylim()
#         plt.ylim((min(0, left), max(1.4, right)))

#         axes = plt.subplot(2, 2, 2)
#         plt.title("Estimated basis representation.\n")
#         fdatairregular.scatter(axes=axes)
#         fdatabasis_estimated[:5].plot(axes=axes)
#         left, right = plt.ylim()
#         plt.ylim((min(0, left), max(1.4, right)))

#         axes = plt.subplot(2, 2, 4)
#         plt.title("Original basis representation")
#         fdatairregular.scatter(axes=axes)
#         fdatabasis_original[:5].plot(axes=axes)
#         left, right = plt.ylim()
#         plt.ylim((min(0, left), max(1.4, right)))

#         axes = plt.subplot(2, 2, 3)
#         plt.title(f"{basis}")
#         basis.plot(axes=axes)

#         plt.show()


def _cmp_estimation_with_original(
    n_points: int,
    sigma: float,  # to generate the noise
    domain_range: Tuple[float, float],
    funcs: List[Callable[[NDArrayFloat], NDArrayFloat]],
    type_gen_points: int,
    estimator: MixedEffectsConverter,
    fit_kwargs: dict[str, Any],
    fdatabasis_original: FDataBasis,
) -> None:
    n_samples = len(funcs)
    points = _get_points(domain_range, n_points, n_samples, type_gen_points)
    fdatairregular = _create_irregular_samples(
        funcs=funcs,
        points=points,
        noise_generate_std=sigma,
        n_points=n_points,
    )

    fdatabasis_estimated = estimator.fit_transform(
        fdatairregular, **fit_kwargs,
    )

    assert estimator.result.success, "Optimization failed"
    assert r2_score(fdatabasis_estimated, fdatabasis_original) > 0.9


def _test_compare_with_original(
    estimator_cls: Type[MixedEffectsConverter],
    fit_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    np.random.seed(34285676)
    if fit_kwargs is None:
        fit_kwargs = {}

    domain_range = (0, 100)
    _max_val = 5
    n_points = 7
    n_basis = 3
    n_samples = 40

    basis = BSplineBasis(
        n_basis=n_basis, domain_range=domain_range, order=2,
    )
    sigma = 0.1
    fe_cov_sqrt = np.zeros((n_basis, n_basis))
    fe_cov_sqrt[np.tril_indices(n_basis)] = np.random.rand(
        n_basis * (n_basis + 1) // 2,
    ) * _max_val
    fe_cov = fe_cov_sqrt @ fe_cov_sqrt.T
    mean = np.array([-15, 20, 6])
    fdatabasis_original = FDataBasis(
        basis=basis,
        coefficients=np.random.multivariate_normal(
            mean=mean, cov=fe_cov, size=n_samples,
        ),
    )

    def fun(i: int):
        return lambda x: fdatabasis_original[i](x).reshape(x.shape)

    funcs = [fun(i) for i in range(n_samples)]

    _cmp_estimation_with_original(
        n_points=n_points,
        sigma=sigma,
        funcs=funcs,
        type_gen_points=5,
        estimator=estimator_cls(basis=basis),
        domain_range=domain_range,
        fit_kwargs=fit_kwargs,
        fdatabasis_original=fdatabasis_original,
    )


# def test_compare_with_statsmodels_minimize() -> None:
#     _test_general_compare_with_original(
#         MinimizeMixedEffectsConverter,
#     )


def test_compare_minimize_with_original() -> None:
    """Compare the EM conversion with the original data."""
    _test_compare_with_original(
        estimator_cls=MinimizeMixedEffectsConverter,
        fit_kwargs={
            "minimization_method": "Powell",
        }
    )


def test_compare_em_with_original() -> None:
    """Compare the EM conversion with the original data."""
    _test_compare_with_original(
        estimator_cls=EMMixedEffectsConverter,
        fit_kwargs={
            "maxiter": 500,
            "convergence_criterion": "params",
            "rtol": 1e-3,
        }
    )
