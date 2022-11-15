"""Tests for registration (alignment)."""
import unittest

import numpy as np
from sklearn.exceptions import NotFittedError

from skfda import FDataGrid
from skfda._utils import invert_warping, normalize_warping
from skfda.datasets import (
    make_multimodal_landmarks,
    make_multimodal_samples,
    make_sinusoidal_process,
)
from skfda.exploratory.stats import mean
from skfda.preprocessing.registration import (
    LeastSquaresShiftRegistration,
    landmark_elastic_registration,
    landmark_elastic_registration_warping,
    landmark_shift_deltas,
    landmark_shift_registration,
)
from skfda.preprocessing.registration.validation import (
    AmplitudePhaseDecomposition,
    LeastSquares,
    PairwiseCorrelation,
    SobolevLeastSquares,
)
from skfda.representation.basis import FourierBasis
from skfda.representation.interpolation import SplineInterpolation


class TestWarping(unittest.TestCase):
    """Test warpings functions."""

    def setUp(self) -> None:
        """Initialize samples."""
        self.time = np.linspace(-1, 1, 50)
        interpolation = SplineInterpolation(3, monotone=True)
        self.polynomial = FDataGrid(
            [self.time**3, self.time**5],
            self.time,
            interpolation=interpolation,
        )

    def test_invert_warping(self) -> None:
        """Test that the composition with invert warping is the identity."""
        inverse = invert_warping(self.polynomial)

        # Check if identity
        identity = self.polynomial.compose(inverse)

        np.testing.assert_array_almost_equal(
            [self.time, self.time],
            identity.data_matrix[..., 0],
            decimal=3,
        )

    def test_standard_normalize_warping(self) -> None:
        """Test normalization to (0, 1)."""
        normalized = normalize_warping(self.polynomial, (0, 1))

        # Test new domain range (0, 1)
        np.testing.assert_array_equal(normalized.domain_range, [(0, 1)])

        np.testing.assert_array_almost_equal(
            normalized.grid_points[0],
            np.linspace(0, 1, 50),
        )

        np.testing.assert_array_almost_equal(
            normalized(0)[..., 0],
            [[0], [0]],
        )

        np.testing.assert_array_almost_equal(
            normalized(1)[..., 0],
            [[1.0], [1.0]],
        )

    def test_standard_normalize_warping_default_value(self) -> None:
        """Test normalization."""
        normalized = normalize_warping(self.polynomial)

        # Test new domain range (0, 1)
        np.testing.assert_array_equal(normalized.domain_range, [(-1, 1)])

        np.testing.assert_array_almost_equal(
            normalized.grid_points[0],
            np.linspace(-1, 1, 50),
        )

        np.testing.assert_array_almost_equal(
            normalized(-1)[..., 0],
            [[-1], [-1]],
        )

        np.testing.assert_array_almost_equal(
            normalized(1)[..., 0],
            [[1.0], [1.0]],
        )

    def test_normalize_warping(self) -> None:
        """Test normalization to (a, b)."""
        a = -4
        b = 3
        domain = (a, b)
        normalized = normalize_warping(self.polynomial, domain)

        # Test new domain range (0, 1)
        np.testing.assert_array_equal(normalized.domain_range, [domain])

        np.testing.assert_array_almost_equal(
            normalized.grid_points[0],
            np.linspace(*domain, 50),
        )

        np.testing.assert_array_equal(normalized(a)[..., 0], [[a], [a]])

        np.testing.assert_array_equal(normalized(b)[..., 0], [[b], [b]])

    def test_landmark_shift_deltas(self) -> None:
        """Test landmark shift deltas."""
        fd = make_multimodal_samples(n_samples=3, random_state=1)
        landmarks = make_multimodal_landmarks(n_samples=3, random_state=1)
        landmarks = landmarks.squeeze()

        shifts = landmark_shift_deltas(fd, landmarks).round(3)
        np.testing.assert_almost_equal(shifts, [0.327, -0.173, -0.154])

    def test_landmark_shift_registration(self) -> None:
        """Test landmark shift registration."""
        fd = make_multimodal_samples(n_samples=3, random_state=1)
        landmarks = make_multimodal_landmarks(n_samples=3, random_state=1)
        landmarks = landmarks.squeeze()

        original_modes = fd(
            landmarks.reshape((3, 1, 1)),
            aligned=False,
        )
        # Test default location
        fd_registered = landmark_shift_registration(fd, landmarks)
        center = np.mean(landmarks)
        reg_modes = fd_registered(center)

        # Test callable location
        np.testing.assert_almost_equal(reg_modes, original_modes, decimal=2)

        fd_registered = landmark_shift_registration(
            fd,
            landmarks,
            location=np.mean,
        )
        center = np.mean(landmarks)
        reg_modes = fd_registered(center)

        np.testing.assert_almost_equal(reg_modes, original_modes, decimal=2)

        # Test integer location
        fd_registered = landmark_shift_registration(
            fd,
            landmarks,
            location=0,
        )
        center = np.mean(landmarks)
        reg_modes = fd_registered(0)

        np.testing.assert_almost_equal(reg_modes, original_modes, decimal=2)

        # Test array location
        fd_registered = landmark_shift_registration(
            fd,
            landmarks,
            location=[0, 0.1, 0.2],
        )
        reg_modes = fd_registered([[0], [0.1], [0.2]], aligned=False)

        np.testing.assert_almost_equal(reg_modes, original_modes, decimal=2)

    def test_landmark_elastic_registration_warping(self) -> None:
        """Test the warpings in landmark elastic registration."""
        fd = make_multimodal_samples(n_samples=3, n_modes=2, random_state=9)
        landmarks = make_multimodal_landmarks(
            n_samples=3,
            n_modes=2,
            random_state=9,
        )
        landmarks = landmarks.squeeze()

        # Default location
        warping = landmark_elastic_registration_warping(fd, landmarks)
        center = (landmarks.max(axis=0) + landmarks.min(axis=0)) / 2
        np.testing.assert_almost_equal(
            warping(center)[..., 0],
            landmarks,
            decimal=1,
        )

        # Fixed location
        center = [0.3, 0.6]
        warping = landmark_elastic_registration_warping(
            fd,
            landmarks,
            location=center,
        )
        np.testing.assert_almost_equal(
            warping(center)[..., 0],
            landmarks,
            decimal=3,
        )

    def test_landmark_elastic_registration(self) -> None:
        """Test landmark elastic registration."""
        fd = make_multimodal_samples(n_samples=3, n_modes=2, random_state=9)
        landmarks = make_multimodal_landmarks(
            n_samples=3,
            n_modes=2,
            random_state=9,
        )
        landmarks = landmarks.squeeze()

        original_values = fd(landmarks.reshape(3, 2), aligned=False)

        # Default location
        fd_reg = landmark_elastic_registration(fd, landmarks)
        center = (landmarks.max(axis=0) + landmarks.min(axis=0)) / 2
        np.testing.assert_almost_equal(
            fd_reg(center),
            original_values,
            decimal=2,
        )

        # Fixed location
        center = [0.3, 0.6]
        fd_reg = landmark_elastic_registration(fd, landmarks, location=center)
        np.testing.assert_array_almost_equal(
            fd_reg(center),
            original_values,
            decimal=2,
        )


class TestLeastSquaresShiftRegistration(unittest.TestCase):
    """Test shift registration."""

    def setUp(self) -> None:
        """Initialize samples."""
        self.fd = make_sinusoidal_process(
            n_samples=2,
            error_std=0,
            random_state=1,
        )
        self.fd.extrapolation = "periodic"  # type: ignore[assignment]

    def test_fit_transform(self) -> None:

        reg = LeastSquaresShiftRegistration[FDataGrid]()

        # Test fit transform with FDataGrid
        fd_reg = reg.fit_transform(self.fd)

        # Check attributes fitted
        self.assertTrue(hasattr(reg, 'deltas_'))
        self.assertTrue(hasattr(reg, 'template_'))
        self.assertTrue(hasattr(reg, 'n_iter_'))
        self.assertTrue(isinstance(fd_reg, FDataGrid))

        deltas = reg.deltas_.round(3)
        np.testing.assert_array_almost_equal(deltas, [-0.022, 0.03])

        # Test with Basis
        fd = self.fd.to_basis(FourierBasis())
        reg.fit_transform(fd)
        deltas = reg.deltas_.round(3)
        np.testing.assert_array_almost_equal(deltas, [-0.022, 0.03])

    def test_fit_and_transform(self) -> None:
        """Test wrapper of shift_registration_deltas."""
        fd = make_sinusoidal_process(
            n_samples=2,
            error_std=0,
            random_state=10,
        )

        reg = LeastSquaresShiftRegistration[FDataGrid]()
        response = reg.fit(self.fd)

        # Check attributes and returned value
        self.assertTrue(hasattr(reg, 'template_'))
        self.assertTrue(response is reg)

        reg.transform(fd)
        deltas = reg.deltas_.round(3)
        np.testing.assert_allclose(deltas, [0.071, -0.072])

    def test_inverse_transform(self) -> None:

        reg = LeastSquaresShiftRegistration[FDataGrid]()
        fd = reg.fit_transform(self.fd)
        fd = reg.inverse_transform(fd)

        np.testing.assert_array_almost_equal(
            fd.data_matrix,
            self.fd.data_matrix,
            decimal=3,
        )

    def test_raises(self) -> None:

        reg = LeastSquaresShiftRegistration[FDataGrid]()

        # Test not fitted
        with np.testing.assert_raises(NotFittedError):
            reg.transform(self.fd)

        reg.fit(self.fd)
        reg.set_params(restrict_domain=True)

        # Test use transform with restrict_domain=True
        with np.testing.assert_raises(AttributeError):
            reg.transform(self.fd)

        # Test inverse_transform without previous transformation
        with np.testing.assert_raises(AttributeError):
            reg.inverse_transform(self.fd)

        reg.fit_transform(self.fd)

        # Test inverse transform with different number of sample
        with np.testing.assert_raises(ValueError):
            reg.inverse_transform(self.fd[:1])

        fd = make_multimodal_samples(dim_domain=2, random_state=0)

        with np.testing.assert_raises(ValueError):
            reg.fit_transform(fd)

        reg.set_params(initial=[0.])

        # Wrong initial estimation
        with np.testing.assert_raises(ValueError):
            reg.fit_transform(self.fd)

    def test_template(self) -> None:

        reg = LeastSquaresShiftRegistration[FDataGrid]()
        fd_registered_1 = reg.fit_transform(self.fd)

        reg_2 = LeastSquaresShiftRegistration[FDataGrid](
            template=reg.template_,
        )
        fd_registered_2 = reg_2.fit_transform(self.fd)

        reg_3 = LeastSquaresShiftRegistration[FDataGrid](template=mean)
        fd_registered_3 = reg_3.fit_transform(self.fd)

        reg_4 = LeastSquaresShiftRegistration[FDataGrid](
            template=reg.template_,
        )
        fd_registered_4 = reg_4.fit(self.fd).transform(self.fd)

        np.testing.assert_array_almost_equal(
            fd_registered_1.data_matrix,
            fd_registered_3.data_matrix,
        )

        # With the template fixed could vary the convergence
        np.testing.assert_array_almost_equal(
            fd_registered_1.data_matrix,
            fd_registered_2.data_matrix,
            decimal=3,
        )

        np.testing.assert_array_almost_equal(
            fd_registered_2.data_matrix,
            fd_registered_4.data_matrix,
        )

    def test_restrict_domain(self) -> None:
        reg = LeastSquaresShiftRegistration[FDataGrid](restrict_domain=True)
        fd_registered_1 = reg.fit_transform(self.fd)

        np.testing.assert_array_almost_equal(
            np.array(fd_registered_1.domain_range).round(3),
            [[0.022, 0.969]],
        )

        reg2 = LeastSquaresShiftRegistration[FDataGrid](
            restrict_domain=True,
            template=reg.template_.copy(domain_range=self.fd.domain_range),
        )
        fd_registered_2 = reg2.fit_transform(self.fd)

        np.testing.assert_array_almost_equal(
            fd_registered_2.data_matrix,
            fd_registered_1.data_matrix,
            decimal=3,
        )

        reg3 = LeastSquaresShiftRegistration[FDataGrid](
            restrict_domain=True,
            template=mean,
        )
        fd_registered_3 = reg3.fit_transform(self.fd)

        np.testing.assert_array_almost_equal(
            fd_registered_3.data_matrix,
            fd_registered_1.data_matrix,
        )

    def test_initial_estimation(self) -> None:
        reg = LeastSquaresShiftRegistration[FDataGrid](
            initial=[-0.02161235, 0.03032652],
        )
        reg.fit_transform(self.fd)

        # Only needed 1 iteration until convergence
        self.assertEqual(reg.n_iter_, 1)

    def test_custom_grid_points(self) -> None:
        reg = LeastSquaresShiftRegistration[FDataGrid](
            grid_points=np.linspace(0, 1, 50),
        )
        reg.fit_transform(self.fd)


class TestRegistrationValidation(unittest.TestCase):
    """Test validation functions."""

    def setUp(self) -> None:
        """Initialize the samples."""
        self.X = make_sinusoidal_process(error_std=0, random_state=0)
        self.shift_registration = LeastSquaresShiftRegistration[FDataGrid]()
        self.shift_registration.fit(self.X)

    def test_amplitude_phase_score(self) -> None:
        """Test basic usage of AmplitudePhaseDecomposition."""
        scorer = AmplitudePhaseDecomposition()
        score = scorer(self.shift_registration, self.X)
        np.testing.assert_allclose(score, 0.971144, rtol=1e-6)

    def test_amplitude_phase_score_with_basis(self) -> None:
        """Test the AmplitudePhaseDecomposition with FDataBasis."""
        scorer = AmplitudePhaseDecomposition()
        X = self.X.to_basis(FourierBasis())
        score = scorer(self.shift_registration, X)
        np.testing.assert_allclose(score, 0.995086, rtol=1e-6)

    def test_default_score(self) -> None:
        """Test default score of a registration transformer."""
        score = self.shift_registration.score(self.X)
        np.testing.assert_allclose(score, 0.971144, rtol=1e-6)

    def test_least_squares_score(self) -> None:
        """Test LeastSquares."""
        scorer = LeastSquares()
        score = scorer(self.shift_registration, self.X)
        np.testing.assert_allclose(score, 0.953355, rtol=1e-6)

    def test_sobolev_least_squares_score(self) -> None:
        """Test SobolevLeastSquares."""
        scorer = SobolevLeastSquares()
        score = scorer(self.shift_registration, self.X)
        np.testing.assert_allclose(score, 0.923962, rtol=1e-6)

    def test_pairwise_correlation(self) -> None:
        """Test PairwiseCorrelation."""
        scorer = PairwiseCorrelation()
        score = scorer(self.shift_registration, self.X)
        np.testing.assert_allclose(score, 1.816228, rtol=1e-6)

    def test_mse_decomposition(self) -> None:
        """Test obtaining all stats from AmplitudePhaseDecomposition."""
        fd = make_multimodal_samples(n_samples=3, random_state=1)
        landmarks = make_multimodal_landmarks(n_samples=3, random_state=1)
        landmarks = landmarks.squeeze()
        warping = landmark_elastic_registration_warping(fd, landmarks)
        fd_registered = fd.compose(warping)
        scorer = AmplitudePhaseDecomposition()
        ret = scorer.stats(fd, fd_registered)
        np.testing.assert_allclose(ret.mse_amplitude, 0.0009465483)
        np.testing.assert_allclose(ret.mse_phase, 0.1051769136)
        np.testing.assert_allclose(ret.r_squared, 0.9910806875)
        np.testing.assert_allclose(ret.c_r, 0.9593073773)

    def test_raises_amplitude_phase(self) -> None:
        scorer = AmplitudePhaseDecomposition()

        # Inconsistent number of functions registered
        with np.testing.assert_raises(ValueError):
            scorer.score_function(self.X, self.X[:2])

        # Inconsistent number of functions registered
        with np.testing.assert_raises(ValueError):
            scorer.score_function(self.X, self.X[:-1])


if __name__ == '__main__':
    unittest.main()
