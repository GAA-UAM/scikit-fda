"""Tests of milstein."""

import numpy as np
from scipy.stats import norm

from skfda.datasets import milstein
from skfda.typing._numpy import NDArrayFloat


def gbm_drift(
    t: float,
    x: NDArrayFloat,
    mu: float = 1,
) -> NDArrayFloat:
    """Drift term of a Geometric Brownian motion."""
    return mu * x


def gbm_diffusion(
    t: float,
    x: NDArrayFloat,
    sigma: float = 1,
) -> NDArrayFloat:
    """Diffusion term of a Geometric Brownian motion."""
    return sigma * x


def gbm_diffusion_derivative(
    t: float,
    x: NDArrayFloat,
    sigma: float = 1,
) -> NDArrayFloat:
    """Spacial derivative of the diffusion term of a GBM."""
    dim_codomain = x.shape[1]
    gbm_diffusion_derivative = np.zeros(
        x.shape + (dim_codomain,),
    )
    for i in np.arange(dim_codomain):
        gbm_diffusion_derivative[:, i, i] = sigma

    return gbm_diffusion_derivative


def test_one_initial_point() -> None:
    """Case 1 -> One initial point + n_samples > 0.

    1.1: initial_condition = 1, n_samples = 2
    mu = 1, sigma = 1

    """
    n_samples = 2
    n_grid_points = 5
    random_state = np.random.RandomState(1)  # noqa: WPS204 -- reproducibility
    initial_float = 1
    n_L0_discretization_points = 1

    expected_result = np.array([[
        [1],
        [2.26698491],
        [1.96298798],
        [1.75841479],
        [1.28790413],
    ],

        [[1],
         [1.65132011],
         [1.05084347],
         [2.49885523],
         [2.04113003],
         ],
    ])

    fd = milstein(
        initial_float,
        gbm_drift,
        gbm_diffusion,
        gbm_diffusion_derivative,
        n_L0_discretization_points,
        n_samples=n_samples,
        n_grid_points=n_grid_points,
        random_state=random_state,
    )

    np.testing.assert_almost_equal(
        fd.data_matrix,
        expected_result,
    )


def test_one_initial_point_monodimensional() -> None:
    """Case 1.2 Starting point = float in an array format + n_samples.

    initial_condition = [0], n_samples = 2.
        The result should be the same as 1.1
    """
    n_samples = 2
    n_grid_points = 5
    random_state = np.random.RandomState(1)
    initial_float_in_list = [1]
    n_L0_discretization_points = 1

    expected_result = np.array([[
        [1],
        [2.26698491],
        [1.96298798],
        [1.75841479],
        [1.28790413],
    ],

        [[1],
         [1.65132011],
         [1.05084347],
         [2.49885523],
         [2.04113003],
         ],
    ])

    fd = milstein(
        initial_float_in_list,
        gbm_drift,
        gbm_diffusion,
        gbm_diffusion_derivative,
        n_L0_discretization_points,
        n_samples=n_samples,
        n_grid_points=n_grid_points,
        random_state=random_state,
    )

    np.testing.assert_almost_equal(
        fd.data_matrix,
        expected_result,
    )


def test_one_initial_point_multidimensional() -> None:
    """Case 1.3 Starting point = array + n_samples.

    initial_condition = np.array([1, 2]), n_samples = 2.
    """
    n_samples = 2
    n_grid_points = 5
    random_state = np.random.RandomState(1)
    initial_array = np.array([1, 2])
    n_L0_discretization_points = 2

    expected_result = np.array([
        [[1, 2],
         [1.54708778, 1.4382791],
         [1.15436809, 2.20520438],
         [1.32744823, 2.06388647],
         [1.20322359, 2.34674103],
         ],

        [[1, 2],
         [0.82261148, 2.74079487],
         [0.93836515, 4.78168655],
         [1.13046062, 3.92459221],
         [1.38153796, 3.19551206],
         ],
    ])

    fd = milstein(
        initial_array,
        gbm_drift,
        gbm_diffusion,
        gbm_diffusion_derivative,
        n_L0_discretization_points,
        n_samples=n_samples,
        n_grid_points=n_grid_points,
        diffusion_matricial_term=False,
        random_state=random_state,
    )

    np.testing.assert_almost_equal(
        fd.data_matrix,
        expected_result,
    )


def test_matrix_diffusion_multidimensional() -> None:
    """Case 2 Starting point = array + n_samples.

    initial_condition = np.array([1, 2]), n_samples = 2.
    """
    n_samples = 2
    n_grid_points = 5
    dim_codomain = 2
    random_state = np.random.RandomState(1)
    initial_array = np.array([1, 2])
    n_L0_discretization_points = 2

    def gbm_matrix_diffusion(  # noqa: WPS430
        t: float,
        x: NDArrayFloat,
        sigma: float = 1,
    ) -> NDArrayFloat:
        """Diffusion term of a Geometric Brownian motion."""
        diffusion = np.zeros((n_samples, dim_codomain, dim_codomain))
        for i in np.arange(dim_codomain):
            diffusion[:, i, i] = sigma * x[:, i]
        return diffusion

    def gbm_matrix_diffusion_derivative(  # noqa: WPS430
        t: float,
        x: NDArrayFloat,
        sigma: float = 1,
    ) -> NDArrayFloat:
        """Diffusion term of a Geometric Brownian motion."""
        gbm_diffusion_derivative = np.zeros(
            (n_samples, dim_codomain, dim_codomain, dim_codomain),
        )
        for i in np.arange(dim_codomain):
            gbm_diffusion_derivative[:, i, i, i] = sigma

        return gbm_diffusion_derivative

    expected_result = np.array([
        [[1, 2],
         [1.54708778, 1.4382791],
         [1.15436809, 2.20520438],
         [1.32744823, 2.06388647],
         [1.20322359, 2.34674103],
         ],

        [[1, 2],
         [0.82261148, 2.74079487],
         [0.93836515, 4.78168655],
         [1.13046062, 3.92459221],
         [1.38153796, 3.19551206],
         ],
    ])

    fd = milstein(
        initial_array,
        gbm_drift,
        gbm_matrix_diffusion,
        gbm_matrix_diffusion_derivative,
        n_L0_discretization_points,
        n_samples=n_samples,
        n_grid_points=n_grid_points,
        random_state=random_state,
    )

    np.testing.assert_almost_equal(
        fd.data_matrix,
        expected_result,
    )


def test_grid_points() -> None:
    """Test for checking that the grid_points are correct."""
    random_state = np.random.RandomState(1)
    initial_condition = np.array([0, 1])
    start = 0
    stop = 10
    n_grid_points = 105
    n_L0_discretization_points = 1
    expected_grid_points = np.atleast_2d(
        np.linspace(start, stop, n_grid_points),
    )

    fd = milstein(
        initial_condition,
        gbm_drift,
        gbm_diffusion,
        gbm_diffusion_derivative,
        n_L0_discretization_points,
        n_grid_points=n_grid_points,
        start=start,
        stop=stop,
        random_state=random_state,
    )

    np.testing.assert_array_equal(
        fd.grid_points,
        expected_grid_points,
    )


def test_random_generator() -> None:
    """Test using Generator instead of RandomState."""
    n_samples = 2
    n_grid_points = 5
    n_L0_discretization_points = 1
    random_state = np.random.default_rng(seed=1)
    normal_distribution = norm().rvs

    expected_result = np.array([
        [[0.34558419],
         [0.45059587],
         [0.30897301],
         [0.51911686],
         [0.71279603],
         ],

        [[0.82161814],
         [0.73334614],
         [1.06905099],
         [1.41531695],
         [1.81568250],
         ],
    ])

    fd = milstein(
        normal_distribution,
        gbm_drift,
        gbm_diffusion,
        gbm_diffusion_derivative,
        n_L0_discretization_points,
        n_samples=n_samples,
        n_grid_points=n_grid_points,
        random_state=random_state,
    )

    np.testing.assert_almost_equal(
        fd.data_matrix,
        expected_result,
    )


def test_n_l0_discretization_negative_cases() -> None:
    """Test for checking related errors with n_L0_discretization_points."""
    initial_condition = np.array([1, 2])
    n_samples = 2
    random_state = np.random.RandomState(1)

    initial_condition = np.array([0, 0, 0])

    with np.testing.assert_raises(ValueError):
        milstein(
            initial_condition,
            gbm_drift,
            gbm_diffusion,
            gbm_diffusion_derivative,
            n_samples=n_samples,
            random_state=random_state,
        )
