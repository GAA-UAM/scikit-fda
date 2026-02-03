"""Tests for PACE module."""

from typing import Any

import numpy as np
import pytest

from skfda.datasets import fetch_cd4
from skfda.preprocessing.dim_reduction import PACE
from skfda.representation import FDataGrid
from skfda.representation.irregular import FDataIrregular

##############################################################################
# Matlab reference code
##############################################################################
#
# The expected values for the Matlab comparison tests were generated using
# the PACE package for Matlab. The following script was used:
#
#     % Load CD4 data from CSV
#     fid = fopen('cd4_counts.csv');
#     header_line = fgetl(fid);
#     raw = textscan(fid, repmat('%s', 1, 61), 'Delimiter', ',');
#     fclose(fid);
#
#     % Convert to numeric matrix
#     nRows = length(raw{1});
#     nCols = length(raw);
#     raw_matrix = strings(nRows, nCols);
#     for i = 1:nCols
#         raw_matrix(:, i) = string(raw{i});
#     end
#     raw_matrix(raw_matrix == "NA") = "NaN";
#     data = double(raw_matrix);
#
#     % Parse time points from header
#     time_points = str2double(strrep(strsplit(header_line, ','), '"', ''));
#
#     % Build cell arrays for irregular data
#     nSubjects = size(data, 1);
#     y = cell(1, nSubjects);
#     t = cell(1, nSubjects);
#     for i = 1:nSubjects
#         valid_idx = ~isnan(data(i, :));
#         t{i} = time_points(valid_idx);
#         y{i} = data(i, valid_idx);
#     end
#
#     % Configure PACE options
#     options = setOptions(...
#         'yname', 'CD4', ...
#         'regular', 0, ...           % Irregular data
#         'kernel', 'gauss', ...      % Gaussian kernel
#         'bwmu', 0, ...              % Estimate bandwidth for mean
#         'bwmu_gcv', 2, ...          % Use GCV for mean bandwidth
#         'bwxcov', [0,0], ...        % Estimate bandwidth for covariance
#         'bwxcov_gcv', 1, ...        % Use GCV for covariance bandwidth
#         'selection_k', 'FVE', ...   % Fraction of Variance Explained
#         'FVE_threshold', 0.99, ...  % 99% variance threshold
#         'method', 'CE', ...         % Conditional Expectation
#         'verbose', 'on', ...
#         'error', 1 ...              % Assume noisy data
#     );
#
#     % Run FPCA
#     result = FPCA(y, t, options);
#
#     % Extract results
#     out1 = getVal(result, 'out1');      % Grid points
#     mu = getVal(result, 'mu');          % Mean function
#     xcov = getVal(result, 'xcov');      % Covariance surface
#     phi = getVal(result, 'phi');        % Eigenfunctions
#
# The bandwidth values used in tests (8.25 for mean, 6.269 for covariance)
# were obtained from the GCV-selected bandwidths in the Matlab output.
#
##############################################################################

##############################################################################
# Sample objects to check parameters
##############################################################################

sample_fd = FDataIrregular(
    start_indices=[0, 2, 4, 6],
    points=np.linspace(1, 10, 10),
    values=np.linspace(1, 10, 10),
)

##############################################################################
# Fixtures
##############################################################################


@pytest.fixture
def fetch_cd4_fixture() -> FDataIrregular:
    """Fixture for loading the CD4 cell counts dataset."""
    fd, _ = fetch_cd4(return_X_y=True)
    return fd


@pytest.fixture(
    params=[
        [
            PACE(
                n_components=5,
                bandwidth_mean=np.array([0.1, 10]),
                bandwidth_cov=np.array([0.1, 10]),
            ),
            sample_fd,
            ValueError,
            "The sample size must be bigger",
        ],
        [
            PACE(
                n_components=5,
                bandwidth_mean=np.array([0.1, 10]),
                bandwidth_cov=np.array([0.1, 10]),
            ),
            FDataIrregular(
                start_indices=[0, 1, 2, 3, 4],
                points=[[1.0], [2.0], [3.0], [4.0], [5.0]],
                values=[[1.0], [2.0], [3.0], [4.0], [5.0]],
            ),
            ValueError,
            "Unable to perform computations with one measurement",
        ],
        [
            PACE(
                n_components=2,
                bandwidth_mean=0.1,
                bandwidth_cov=0.1,
                n_grid_points=51,
            ),
            sample_fd,
            ValueError,
            "Covariance matrix has invalid eigenvalues",
        ],
    ],
)
def input_fd_raises_fixture(
    request: Any,
) -> tuple[PACE, FDataIrregular, type[Exception], str]:
    """Fixture for checking incorrect input data forms."""
    return tuple(request.param)


@pytest.fixture(
    params=[
        [
            PACE(
                n_components=2,
                bandwidth_mean=10.0,
                bandwidth_cov=10.0,
            ),
            sample_fd,
            10.0,
            10.0,
        ],
        [
            PACE(
                n_components=2,
                bandwidth_mean=np.array([0.1, 100]),
                bandwidth_cov=np.array([0.1, 100]),
            ),
            sample_fd,
            99.999,  # GCV-selected mean bandwidth
            1.293,  # GCV-selected cov bandwidth
        ],
    ],
)
def test_bandwidth_fixture(
    request: Any,
) -> tuple[PACE, FDataIrregular, float, float]:
    """Fixture for testing the bandwidth for the mean and covariance."""
    return tuple(request.param)


@pytest.fixture(
    params=[
        [
            PACE(
                n_components=2,
                bandwidth_mean=10.0,
                bandwidth_cov=10.0,
                assume_noisy=False,
            ),
            sample_fd,
            0.0,
        ],
        [
            PACE(
                n_components=2,
                bandwidth_mean=10.0,
                bandwidth_cov=10.0,
                assume_noisy=True,
            ),
            sample_fd,
            2.0427258509225148e-16,
        ],
    ],
)
def test_sigma_fixture(
    request: Any,
) -> tuple[PACE, FDataIrregular, float]:
    """Fixture for testing the noise in the data."""
    return tuple(request.param)


@pytest.fixture(
    params=[
        [
            PACE(
                n_components=2,
                bandwidth_mean=10.0,
                bandwidth_cov=10.0,
                boundary_effect_interval=(0.25, 0.75),
            ),
            FDataIrregular(
                start_indices=[0, 2, 4, 6],
                points=np.linspace(1, 10, 10),
                values=np.array([0, 0, 5, 5, 5, 5, 5, 5, 0, 0]),
            ),
            FDataGrid(
                data_matrix=np.array([
                    4.99999941,
                    4.99999964,
                    4.99999981,
                    4.99999991,
                    4.99999996,
                    4.99999996,
                    4.99999991,
                    4.99999981,
                    4.99999964,
                    4.99999941,
                ]),
                grid_points=np.linspace(1, 10, 10),
            ),
        ],
        [
            PACE(
                n_components=2,
                bandwidth_mean=10.0,
                bandwidth_cov=10.0,
                boundary_effect_interval=(0.0, 1.0),
            ),
            FDataIrregular(
                start_indices=[0, 2, 4, 6],
                points=np.linspace(1, 10, 10),
                values=np.array([0, 0, 5, 5, 5, 5, 5, 5, 0, 0]),
            ),
            FDataGrid(
                data_matrix=np.array([
                    2.69681501,
                    2.84793566,
                    2.9612617,
                    3.03680541,
                    3.07457514,
                    3.07457514,
                    3.03680541,
                    2.9612617,
                    2.84793566,
                    2.69681501,
                ]),
                grid_points=np.linspace(1, 10, 10),
            ),
        ],
    ],
)
def test_boundary_effect_fixture(
    request: Any,
) -> tuple[PACE, FDataIrregular, FDataGrid]:
    """Fixture for testing the boundary effect in the data."""
    return tuple(request.param)


##############################################################################
# Tests
##############################################################################


class TestPACEValidation:
    """Tests for PACE parameter validation."""

    @pytest.mark.parametrize(
        ("kwargs", "expected_msg"),
        [
            # n_components invalid
            (
                {
                    "n_components": 0,
                    "bandwidth_mean": 0.1,
                    "bandwidth_cov": 0.1,
                },
                "n_components must be an integer or a float in",
            ),
            (
                {
                    "n_components": 0.0,
                    "bandwidth_mean": 0.1,
                    "bandwidth_cov": 0.1,
                },
                "n_components must be an integer or a float in",
            ),
            (
                {
                    "n_components": 1.0,
                    "bandwidth_mean": 0.1,
                    "bandwidth_cov": 0.1,
                },
                "n_components must be an integer or a float in",
            ),
            # grid points invalid
            (
                {
                    "n_components": 0.9,
                    "bandwidth_mean": 0.1,
                    "bandwidth_cov": 0.1,
                    "n_grid_points": 0,
                },
                "Grid points must be positive",
            ),
            (
                {
                    "n_components": 0.9,
                    "bandwidth_mean": 0.1,
                    "bandwidth_cov": 0.1,
                    "bw_cov_n_grid_points": 0,
                },
                "Grid points must be positive",
            ),
            # boundary_effect_interval invalid
            (
                {
                    "n_components": 0.9,
                    "bandwidth_mean": 0.1,
                    "bandwidth_cov": 0.1,
                    "boundary_effect_interval": (0.0,),
                },
                "boundary_effect_interval must have exactly 2 elements",
            ),
            (
                {
                    "n_components": 0.9,
                    "bandwidth_mean": 0.1,
                    "bandwidth_cov": 0.1,
                    "boundary_effect_interval": (0.8, 0.2),
                },
                "boundary_effect_interval must satisfy 0 <= a < b <= 1",
            ),
            (
                {
                    "n_components": 0.9,
                    "bandwidth_mean": 0.1,
                    "bandwidth_cov": 0.1,
                    "boundary_effect_interval": (-0.1, 0.5),
                },
                "boundary_effect_interval must satisfy 0 <= a < b <= 1",
            ),
            (
                {
                    "n_components": 0.9,
                    "bandwidth_mean": 0.1,
                    "bandwidth_cov": 0.1,
                    "boundary_effect_interval": (0.0, 1.5),
                },
                "boundary_effect_interval must satisfy 0 <= a < b <= 1",
            ),
            # variance_error_interval invalid
            (
                {
                    "n_components": 0.9,
                    "bandwidth_mean": 0.1,
                    "bandwidth_cov": 0.1,
                    "variance_error_interval": (0.3,),
                },
                "variance_error_interval must have exactly 2 elements",
            ),
            (
                {
                    "n_components": 0.9,
                    "bandwidth_mean": 0.1,
                    "bandwidth_cov": 0.1,
                    "variance_error_interval": (0.7, 0.2),
                },
                "variance_error_interval must satisfy 0 <= a < b <= 1",
            ),
            (
                {
                    "n_components": 0.9,
                    "bandwidth_mean": 0.1,
                    "bandwidth_cov": 0.1,
                    "variance_error_interval": (-0.1, 0.8),
                },
                "variance_error_interval must satisfy 0 <= a < b <= 1",
            ),
            (
                {
                    "n_components": 0.9,
                    "bandwidth_mean": 0.1,
                    "bandwidth_cov": 0.1,
                    "variance_error_interval": (0.0, 1.5),
                },
                "variance_error_interval must satisfy 0 <= a < b <= 1",
            ),
            # bandwidth_mean invalid
            (
                {
                    "n_components": 0.9,
                    "bandwidth_mean": -0.5,
                    "bandwidth_cov": 0.1,
                },
                "Given bandwidth values must be positive",
            ),
            (
                {
                    "n_components": 0.9,
                    "bandwidth_mean": [0.1],
                    "bandwidth_cov": 0.1,
                },
                "Bandwidth search ranges must be",
            ),
            (
                {
                    "n_components": 0.9,
                    "bandwidth_mean": [0.0, 1.0],
                    "bandwidth_cov": 0.1,
                },
                "Bandwidth search ranges must be",
            ),
            (
                {
                    "n_components": 0.9,
                    "bandwidth_mean": [1.0, 0.5],
                    "bandwidth_cov": 0.1,
                },
                "Bandwidth search ranges must be",
            ),
            # bandwidth_cov invalid
            (
                {
                    "n_components": 0.9,
                    "bandwidth_mean": 0.1,
                    "bandwidth_cov": -0.1,
                },
                "Given bandwidth values must be positive",
            ),
            (
                {
                    "n_components": 0.9,
                    "bandwidth_mean": 0.1,
                    "bandwidth_cov": [0.1],
                },
                "Bandwidth search ranges must be",
            ),
            (
                {
                    "n_components": 0.9,
                    "bandwidth_mean": 0.1,
                    "bandwidth_cov": [0.0, 1.0],
                },
                "Bandwidth search ranges must be",
            ),
            (
                {
                    "n_components": 0.9,
                    "bandwidth_mean": 0.1,
                    "bandwidth_cov": [1.0, 0.5],
                },
                "Bandwidth search ranges must be",
            ),
        ],
    )
    def test_raises(
        self,
        kwargs: dict[str, Any],
        expected_msg: str,
    ) -> None:
        """Check that PACE.fit() raises ValueError for invalid params."""
        pace = PACE(**kwargs)
        with pytest.raises(ValueError, match=expected_msg):
            pace.fit(sample_fd)


def test_input_fd_adequate(
    input_fd_raises_fixture: tuple[PACE, FDataIrregular, type[Exception], str],
) -> None:
    """Check that the number of components is valid."""
    pace, fd, excep, error_msg = input_fd_raises_fixture
    with pytest.raises(excep, match=error_msg):
        pace.fit(fd)


def test_mean_from_matlab(fetch_cd4_fixture: FDataIrregular) -> None:
    """
    Check that the mean is calculated correctly against Matlab PACE.

    See the 'Matlab reference code' section at the top of this file for
    the script used to generate the expected values.
    """
    # Expected mean from Matlab PACE (rounded to 4 decimals for clarity)
    matlab_mean = np.array([
        986.1189, 989.1560, 991.6712, 993.5323, 994.6030, 994.7466, 993.8285,
        991.7216, 988.3114, 983.5016, 977.2211, 969.4295, 960.1220, 949.3330,
        937.1374, 923.6488, 909.0164, 893.4185, 860.1407, 842.8924, 825.5250,
        808.2426, 791.2328, 774.6622, 758.6738, 743.3849, 728.8873, 715.2470,
        702.5060, 690.6839, 679.7799, 669.7756, 660.6366, 652.3158, 644.7553,
        637.8886, 631.6432, 625.9426, 620.7085, 615.8628, 611.3299, 607.0380,
        602.9214, 598.9214, 594.9876, 591.0794, 587.1662, 583.2283, 579.2575,
        575.2572, 571.2423, 567.2387, 563.2831, 559.4211, 555.7061, 552.1970,
        548.9562, 546.0469, 543.5310, 541.4660,
    ])

    pace = PACE(
        n_components=0.99,
        n_grid_points=51,
        bandwidth_mean=8.25,
        bandwidth_cov=6.269,
        _apply_gaussian_bandwidth_correction=True,
    )
    pace.fit(fetch_cd4_fixture)

    np.testing.assert_allclose(
        matlab_mean,
        pace.mean_.data_matrix[0, :, 0],
        atol=1e-3,
    )


def test_cov_from_matlab(fetch_cd4_fixture: FDataIrregular) -> None:
    """
    Check that the covariance is calculated correctly against Matlab PACE.

    See the 'Matlab reference code' section at the top of this file for
    the script used to generate the expected values. Covariance values
    are rounded since relative tolerance is used for comparison.
    """
    # Expected covariance from Matlab PACE (first row of 51x51 matrix)
    # Full matrix is 51x51=2601 values; storing first row for brevity
    matlab_cov_row0 = np.array([
        58000, 62753, 66823, 70091, 72456, 73841, 74198, 73516, 71837, 69264,
        65976, 62219, 58282, 54450, 50973, 48028, 45724, 44102, 43153, 42826,
        43035, 43666, 44581, 45632, 46670, 47570, 48249, 48682, 48908, 49026,
        49178, 49524, 50221, 51398, 53141, 55490, 58433, 61923, 65888, 70248,
        74930, 79880, 85073, 90515, 96245, 102333, 108875, 115990, 123815,
        132491, 142161,
    ])

    pace = PACE(
        n_components=0.99,
        n_grid_points=51,
        bandwidth_mean=8.25,
        bandwidth_cov=6.269,
        _apply_gaussian_bandwidth_correction=True,
    )
    pace.fit(fetch_cd4_fixture)

    # Compare first row of covariance matrix
    np.testing.assert_allclose(
        matlab_cov_row0,
        pace.covariance_[0, :, 0],
        rtol=0.01,  # 1% relative tolerance
    )


def test_select_bandwidth(
    test_bandwidth_fixture: tuple[PACE, FDataIrregular, float, float],
) -> None:
    """Check the bandwidth usage is correct."""
    pace, fd, bw_mean, bw_cov = test_bandwidth_fixture
    pace.fit(fd)

    np.testing.assert_allclose(
        bw_mean,
        pace.bandwidth_mean_ if pace.bandwidth_mean_ is not None else 0.0,
        atol=1e-1,
    )

    np.testing.assert_allclose(
        bw_cov,
        pace.bandwidth_cov_ if pace.bandwidth_cov_ is not None else 0.0,
        atol=1e-1,
    )


def test_noise(
    test_sigma_fixture: tuple[PACE, FDataIrregular, float],
) -> None:
    """Check the noise is accounted for when relevant."""
    pace, fd, sigma = test_sigma_fixture
    pace.fit(fd)

    np.testing.assert_allclose(
        sigma,
        pace.sigma2_,
        atol=2e-15,
    )


def test_boundary_effect(
    test_boundary_effect_fixture: tuple[PACE, FDataIrregular, FDataGrid],
) -> None:
    """Check the boundary effect is correctly evaluated."""
    pace, fd, expected_fd = test_boundary_effect_fixture
    pace.fit(fd)

    np.testing.assert_allclose(
        pace.mean_.data_matrix,
        expected_fd.data_matrix,
        atol=1e-3,
    )


##############################################################################
# Edge case tests
##############################################################################


class TestPACEEdgeCases:
    """Tests for edge cases and special scenarios."""

    @pytest.fixture
    def dense_irregular_fd(self) -> FDataIrregular:
        """Create irregular data with enough points for interpolation."""
        n_samples = 6
        n_points_per_sample = 8
        t = np.linspace(0, 10, n_points_per_sample)

        # Create data with some variation
        all_points = []
        all_values = []
        start_indices = []
        idx = 0

        for i in range(n_samples):
            start_indices.append(idx)
            for tj in t:
                all_points.append([tj])
                all_values.append([10 + i * 2 + np.sin(tj)])
                idx += 1

        return FDataIrregular(
            start_indices=np.array(start_indices),
            points=np.array(all_points),
            values=np.array(all_values),
        )

    def test_single_component(
        self,
        dense_irregular_fd: FDataIrregular,
    ) -> None:
        """Test PACE with n_components=1."""
        pace = PACE(
            n_components=1,
            bandwidth_mean=2.0,
            bandwidth_cov=2.0,
        )
        pace.fit(dense_irregular_fd)

        assert pace.n_components == 1
        assert pace.components_.n_samples == 1
        assert pace.explained_variance_.shape == (1,)

    def test_reconstruction_grid_int(
        self,
        dense_irregular_fd: FDataIrregular,
    ) -> None:
        """Test PACE with integer reconstruction_grid."""
        pace = PACE(
            n_components=2,
            bandwidth_mean=2.0,
            bandwidth_cov=2.0,
            reconstruction_grid=15,
        )
        pace.fit(dense_irregular_fd)

        # Mean should have 15 grid points
        assert len(pace.mean_.grid_points[0]) == 15

    def test_reconstruction_grid_array(
        self,
        dense_irregular_fd: FDataIrregular,
    ) -> None:
        """Test PACE with custom array reconstruction_grid."""
        custom_grid = np.linspace(0, 10, 20)
        pace = PACE(
            n_components=2,
            bandwidth_mean=2.0,
            bandwidth_cov=2.0,
            reconstruction_grid=custom_grid,
        )
        pace.fit(dense_irregular_fd)

        np.testing.assert_array_equal(
            pace.mean_.grid_points[0],
            custom_grid,
        )

    def test_fve_threshold(self, dense_irregular_fd: FDataIrregular) -> None:
        """Test PACE with FVE threshold for component selection."""
        pace = PACE(
            n_components=0.9,  # 90% variance explained
            bandwidth_mean=2.0,
            bandwidth_cov=2.0,
        )
        pace.fit(dense_irregular_fd)

        # Should select enough components to explain 90% variance
        assert np.sum(pace.explained_variance_ratio_) >= 0.9

    def test_transform_returns_correct_shape(
        self,
        dense_irregular_fd: FDataIrregular,
    ) -> None:
        """Test that transform returns correct shape."""
        pace = PACE(
            n_components=2,
            bandwidth_mean=2.0,
            bandwidth_cov=2.0,
        )
        scores = pace.fit_transform(dense_irregular_fd)

        assert scores.shape == (6, 2)  # 6 samples, 2 components

    def test_inverse_transform(
        self,
        dense_irregular_fd: FDataIrregular,
    ) -> None:
        """Test that inverse_transform returns FDataGrid."""
        pace = PACE(
            n_components=2,
            bandwidth_mean=2.0,
            bandwidth_cov=2.0,
        )
        scores = pace.fit_transform(dense_irregular_fd)
        reconstructed = pace.inverse_transform(scores)

        assert isinstance(reconstructed, FDataGrid)
        assert reconstructed.n_samples == 6

    def test_assume_noisy_false(
        self,
        dense_irregular_fd: FDataIrregular,
    ) -> None:
        """Test PACE with assume_noisy=False."""
        pace = PACE(
            n_components=2,
            bandwidth_mean=2.0,
            bandwidth_cov=2.0,
            assume_noisy=False,
        )
        pace.fit(dense_irregular_fd)

        assert pace.sigma2_ == 0.0
