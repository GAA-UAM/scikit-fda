"""Tests of Euler Maruyama."""

import numpy as np
from scipy.stats import multivariate_normal, norm

from skfda.datasets import euler_maruyama
from skfda.typing._numpy import NDArrayFloat


def test_one_initial_point() -> None:
    """Case 1 -> One initial point + n_samples > 0.

    Tests:
        - initial_condition = 0, n_samples = 2
        - initial_condition = [0], n_samples = 2.
            # The result should be the same as the first one

    """
    n_samples = 2
    n_grid_points = 5
    random_state = np.random.RandomState(1)
    # Case 1 Starting point = float + n_samples
    initial_float = 0

    expected_result = np.array([[
        [0],
        [0.81217268],
        [0.50629448],
        [0.2422086],
        [-0.29427571],
    ],
        [[0],
         [0.43270381],
         [-0.71806553],
         [0.15434035],
         [-0.2262631],
         ],
    ])

    fd = euler_maruyama(
        initial_float,
        n_samples=n_samples,
        n_grid_points=n_grid_points,
        random_state=random_state,
    )

    np.testing.assert_almost_equal(
        fd.data_matrix,
        expected_result,
    )

    # Case 2 Starting point = float in an array format + n_samples
    # The result must be the same as in Case 1
    n_samples = 2
    n_grid_points = 5
    random_state = np.random.RandomState(1)
    initial_float_in_list = [0]

    fd = euler_maruyama(
        initial_float_in_list,
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

    initial_condition = np.array([1, 0]), n_samples = 2.
    """
    n_samples = 2
    n_grid_points = 5
    random_state = np.random.RandomState(1)
    initial_array = np.array([1, 0])

    expected_result = np.array([
        [[1, 0],
         [1.81217268, -0.30587821],
         [1.54808681, -0.84236252],
         [1.98079062, -1.99313187],
         [2.8531965, -2.37373532],
         ],
        [[1, 0],
         [1.15951955, -0.12468519],
         [1.89057352, -1.15475554],
         [1.72936491, -1.34678272],
         [2.29624964, -1.89672835],
         ],
    ])

    fd = euler_maruyama(
        initial_array,
        n_samples=n_samples,
        n_grid_points=n_grid_points,
        random_state=random_state,
    )

    np.testing.assert_almost_equal(
        fd.data_matrix,
        expected_result,
    )


def test_initial_data_generator() -> None:
    """.

    Case 2 -> A function that generates random variables or vectors +
    n_samples > 0.

    Tests:
        - initial_condition = random_state.standard_normal,
            n_samples = 2.
        - initial_condition = random_state.standard_normal_2d,
            n_samples = 2.
    """
    n_samples = 2
    n_grid_points = 5

    # Case 1 Starting generator of floats + n_samples
    random_state = np.random.RandomState(1)

    def standard_normal_generator(
        size: int,
        random_state: np.random.RandomState,
    ) -> NDArrayFloat:
        return (
            random_state.standard_normal(size)
        )

    expected_result = np.array([
        [[1.62434536],
         [1.36025949],
         [0.82377518],
         [1.25647899],
         [0.10570964],
         ],
        [[-0.61175641],
         [0.26064947],
         [-0.11995398],
         [0.03956557],
         [-0.08511962],
         ],
    ])

    fd = euler_maruyama(
        standard_normal_generator,
        n_samples=n_samples,
        n_grid_points=n_grid_points,
        random_state=random_state,
    )

    np.testing.assert_almost_equal(
        fd.data_matrix,
        expected_result,
    )

    # Case 2 Starting generator of vecotrs + n_samples
    n_samples = 2
    n_grid_points = 5
    random_state = np.random.RandomState(1)

    def standard_normal_generator_2d(
        size: int,
        random_state: np.random.RandomState,
    ) -> NDArrayFloat:
        return random_state.standard_normal((size, 2))

    expected_result = np.array([
        [[1.62434536, -0.61175641],
         [2.05704918, -1.76252576],
         [2.92945506, -2.14312921],
         [3.08897461, -2.2678144],
         [3.82002858, -3.29788476],
         ],
        [[-0.52817175, -1.07296862],
         [-0.68938035, -1.2649958],
         [-0.12249563, -1.81494143],
         [-0.20870974, -2.25387064],
         [-0.18760286, -1.96246304],
         ],
    ])

    fd = euler_maruyama(
        standard_normal_generator_2d,
        n_samples=n_samples,
        n_grid_points=n_grid_points,
        random_state=random_state,
    )

    np.testing.assert_almost_equal(
        fd.data_matrix,
        expected_result,
    )


def test_initial_distribution() -> None:
    """.

    Case 3 -> A scipy.stats distribution that generates data with the
    rvs method.
    Tests:
        - initial_condition = scipy.stats.norm().rvs, n_samples = 2
        - initial_condition = scipy.stats.multivariate_norm().rvs,
            n_samples = 2.
    """
    n_samples = 2
    n_grid_points = 5

    # Case 1 Starting 1-d distribution + n_samples
    # The result should be the same as with the generator function
    random_state = np.random.RandomState(1)
    normal_distribution = norm().rvs

    expected_result = np.array([
        [[1.62434536],
         [1.36025949],
         [0.82377518],
         [1.25647899],
         [0.10570964],
         ],
        [[-0.61175641],
         [0.26064947],
         [-0.11995398],
         [0.03956557],
         [-0.08511962],
         ],
    ])

    fd = euler_maruyama(
        normal_distribution,
        n_samples=n_samples,
        n_grid_points=n_grid_points,
        random_state=random_state,
    )

    np.testing.assert_almost_equal(
        fd.data_matrix,
        expected_result,
    )

    # Case 2 Starting n-d distribution + n_samples
    n_samples = 2
    n_grid_points = 5
    random_state = np.random.RandomState(1)
    mean = np.array([0, 1])
    covarianze_matrix = np.array([[1, 0], [0, 1]])

    multivariate_normal_distribution = (
        multivariate_normal(mean, covarianze_matrix).rvs
    )

    expected_result = np.array([
        [[1.62434536, 0.38824359],
         [2.05704918, -0.76252576],
         [2.92945506, -1.14312921],
         [3.08897461, -1.2678144],
         [3.82002858, -2.29788476],
         ],
        [[-0.52817175, -0.07296862],
         [-0.68938035, -0.2649958],
         [-0.12249563, -0.81494143],
         [-0.20870974, -1.25387064],
         [-0.18760286, -0.96246304],
         ],
    ])

    fd = euler_maruyama(
        multivariate_normal_distribution,
        n_samples=n_samples,
        n_grid_points=n_grid_points,
        random_state=random_state,
    )

    np.testing.assert_almost_equal(
        fd.data_matrix,
        expected_result,
    )


def test_vector_initial_points() -> None:
    """Caso 4 -> A vector of initial points.

    Tests:
        - initial_condition = 0
        - initial_condition = np.array([0, 1])
        - initial_condition = np.array([[0, 1],
                                        [2, 0]])
    """
    n_grid_points = 5

    # Case 1. One Starting 1d point = float
    initial_float = 0
    random_state = np.random.RandomState(1)
    expected_result = np.array([
        [[0],
         [0.81217268],
         [0.50629448],
         [0.2422086],
         [-0.29427571],
         ],
    ])

    fd = euler_maruyama(
        initial_float,
        n_grid_points=n_grid_points,
        random_state=random_state,
    )

    np.testing.assert_almost_equal(
        fd.data_matrix,
        expected_result,
    )

    # Case 2. Two Starting 1d points
    n_grid_points = 5
    initial_array = np.array([0, 1])
    random_state = np.random.RandomState(1)
    expected_result = np.array([
        [[0],
         [0.81217268],
         [0.50629448],
         [0.2422086],
         [-0.29427571],
         ],
        [[1],
         [1.43270381],
         [0.28193447],
         [1.15434035],
         [0.7737369],
         ],
    ])

    fd = euler_maruyama(
        initial_array,
        n_grid_points=n_grid_points,
        random_state=random_state,
    )

    np.testing.assert_almost_equal(
        fd.data_matrix,
        expected_result,
    )

    # Case 3. Starting vector
    n_grid_points = 5
    initial_array = np.array([[0, 1], [2, 0]])
    random_state = np.random.RandomState(1)
    expected_result = np.array([
        [[0, 1],
         [0.81217268, 0.69412179],
         [0.54808681, 0.15763748],
         [0.98079062, -0.99313187],
         [1.8531965, -1.37373532],
         ],
        [[2, 0],
         [2.15951955, -0.12468519],
         [2.89057352, -1.15475554],
         [2.72936491, -1.34678272],
         [3.29624964, -1.89672835],
         ],
    ])

    fd = euler_maruyama(
        initial_array,
        n_grid_points=n_grid_points,
        random_state=random_state,
    )

    np.testing.assert_almost_equal(
        fd.data_matrix,
        expected_result,
    )


def test_drift_cases() -> None:
    """Test for different drift inputs."""
    initial_condition = np.array([0, 0])
    n_samples = 2
    n_grid_points = 5
    random_state = np.random.RandomState(1)

    def base_drift(
        t: float,
        x: NDArrayFloat,
    ) -> float:
        return 0

    expected_result = np.array([
        [[0, 0],
         [0.81217268, -0.30587821],
         [0.54808681, -0.84236252],
         [0.98079062, -1.99313187],
         [1.8531965, -2.37373532],
         ],
        [[0, 0],
         [0.15951955, -0.12468519],
         [0.89057352, -1.15475554],
         [0.72936491, -1.34678272],
         [1.29624964, -1.89672835],
         ],
    ])

    fd = euler_maruyama(
        initial_condition,
        n_grid_points=n_grid_points,
        n_samples=n_samples,
        drift=base_drift,
        diffusion=1,
        random_state=random_state,
    )

    np.testing.assert_almost_equal(
        fd.data_matrix,
        expected_result,
    )

    initial_condition = np.array([0, 0])
    n_samples = 2
    n_grid_points = 5
    random_state = np.random.RandomState(1)

    def vector_drift(
        t: float,
        x: NDArrayFloat,
    ) -> NDArrayFloat:
        return np.array([0, 0])

    expected_result = np.array([
        [[0, 0],
         [0.81217268, -0.30587821],
         [0.54808681, -0.84236252],
         [0.98079062, -1.99313187],
         [1.8531965, -2.37373532],
         ],
        [[0, 0],
         [0.15951955, -0.12468519],
         [0.89057352, -1.15475554],
         [0.72936491, -1.34678272],
         [1.29624964, -1.89672835],
         ],
    ])

    fd = euler_maruyama(
        initial_condition,
        n_grid_points=n_grid_points,
        n_samples=n_samples,
        drift=vector_drift,
        diffusion=1,
        random_state=random_state,
    )

    np.testing.assert_almost_equal(
        fd.data_matrix,
        expected_result,
    )


def test_diffusion_cases() -> None:
    """Test for different diffusion inputs."""
    initial_condition = np.array([0, 0])
    n_samples = 2
    n_grid_points = 5
    random_state = np.random.RandomState(1)

    vector_diffusion = np.array([1, 2])

    # The result should be the same as the previous test, but with
    # the second column doubled
    expected_result = np.array([
        [[0, 0],
         [0.81217268, -0.30587821],
         [0.54808681, -0.84236252],
         [0.98079062, -1.99313187],
         [1.8531965, -2.37373532],
         ],
        [[0, 0],
         [0.15951955, -0.12468519],
         [0.89057352, -1.15475554],
         [0.72936491, -1.34678272],
         [1.29624964, -1.89672835],
         ],
    ])
    expected_result[:, :, 1] *= 2

    fd = euler_maruyama(
        initial_condition,
        n_grid_points=n_grid_points,
        n_samples=n_samples,
        drift=0,
        diffusion=vector_diffusion,
        random_state=random_state,
    )

    np.testing.assert_almost_equal(
        fd.data_matrix,
        expected_result,
    )

    # The expected result should be the same than in test 2 of case 1 of
    # initial_conditions tests but adding a second column that is the
    # first doubled

    initial_condition = np.array([0, 0])
    n_samples = 2
    n_grid_points = 5
    random_state = np.random.RandomState(1)

    matrix_diffusion = np.array([[1], [2]])

    expected_result = np.array([
        [[0],
         [0.81217268],
         [0.50629448],
         [0.2422086],
         [-0.29427571],
         ],
        [[0],
         [0.43270381],
         [-0.71806553],
         [0.15434035],
         [-0.2262631],
         ],
    ])
    expected_result = np.concatenate(
        (
            expected_result,
            2 * expected_result,
        ),
        axis=2,
    )

    fd = euler_maruyama(
        initial_condition,
        n_grid_points=n_grid_points,
        n_samples=n_samples,
        drift=0,
        diffusion=matrix_diffusion,
        random_state=random_state,
        is_diffusion_matrix=True,
        dim_noise=1,
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
    expected_grid_points = np.atleast_2d(
        np.linspace(start, stop, n_grid_points),
    )

    fd = euler_maruyama(
        initial_condition,
        n_grid_points=n_grid_points,
        n_samples=10,
        start=start,
        stop=stop,
        random_state=random_state,
    )

    np.testing.assert_array_equal(
        fd.grid_points,
        expected_grid_points,
    )


def test_initial_condition_negative_cases() -> None:
    """Test for checking initial_condition related error cases."""
    random_state = np.random.RandomState(1)

    # Case 1 Starting vector of points and n_samples
    initial_condition = np.zeros((2, 2))
    n_samples = 3

    with np.testing.assert_raises(ValueError):
        euler_maruyama(
            initial_condition,
            n_samples=n_samples,
            random_state=random_state,
        )

    # Case 2: Inital condition is an array of more than 2d
    initial_condition = np.zeros((1, 1, 1))

    with np.testing.assert_raises(ValueError):
        euler_maruyama(
            initial_condition,
            random_state=random_state,
        )

    # Case 3: generator function without n_samples
    initial_generator = norm().rvs

    with np.testing.assert_raises(ValueError):
        euler_maruyama(
            initial_generator,
            random_state=random_state,
        )

    # Case 4: n_samples not greater than 0

    with np.testing.assert_raises(ValueError):
        euler_maruyama(
            0,
            n_samples=-1,
            random_state=random_state,
        )


def test_diffusion_negative_cases() -> None:
    """Test for checking diffusion related error cases."""
    initial_condition = np.array([0, 0])
    n_samples = 2
    random_state = np.random.RandomState(1)

    bad_diffusion = np.array([[1, 2, 3]])

    with np.testing.assert_raises(ValueError):
        euler_maruyama(
            initial_condition,
            n_samples=n_samples,
            diffusion=bad_diffusion,
            random_state=random_state,
        )

    vector_diffusion = np.array([1, 2])

    initial_condition = np.array([0, 0, 0])

    with np.testing.assert_raises(ValueError):
        euler_maruyama(
            initial_condition,
            n_samples=n_samples,
            diffusion=vector_diffusion,
            random_state=random_state,
        )


def test_random_generator() -> None:
    """Test using Generator instead of RandomState."""
    n_samples = 2
    n_grid_points = 5
    random_state = np.random.default_rng(seed=1)
    normal_distribution = norm().rvs

    expected_result = np.array([
        [[0.34558419],
         [0.51080273],
         [-0.14077589],
         [0.31190205],
         [0.53508933],
         ],

        [[0.82161814],
         [0.55314153],
         [0.84370058],
         [1.02598678],
         [1.17305302],
         ],
    ])

    fd = euler_maruyama(
        normal_distribution,
        n_samples=n_samples,
        n_grid_points=n_grid_points,
        random_state=random_state,
    )

    np.testing.assert_almost_equal(
        fd.data_matrix,
        expected_result,
    )
