from skfda import FDataGrid
from skfda._utils import _check_estimator
from skfda.datasets import (make_multimodal_samples, make_multimodal_landmarks,
                            make_sinusoidal_process)
from skfda.exploratory.stats import mean
from skfda.preprocessing.registration import (
    normalize_warping, invert_warping, landmark_shift_deltas, landmark_shift,
    landmark_registration_warping, landmark_registration, ShiftRegistration)
from skfda.preprocessing.registration.validation import (
    AmplitudePhaseDecomposition, LeastSquares,
    SobolevLeastSquares, PairwiseCorrelation)
from skfda.representation.basis import Fourier
from skfda.representation.interpolation import SplineInterpolation
import unittest

from sklearn.exceptions import NotFittedError

import numpy as np


class TestWarping(unittest.TestCase):
    """Test warpings functions"""

    def setUp(self):
        """Initialization of samples"""

        self.time = np.linspace(-1, 1, 50)
        interpolation = SplineInterpolation(3, monotone=True)
        self.polynomial = FDataGrid([self.time**3, self.time**5],
                                    self.time, interpolation=interpolation)

    def test_invert_warping(self):

        inverse = invert_warping(self.polynomial)

        # Check if identity
        id = self.polynomial.compose(inverse)

        np.testing.assert_array_almost_equal([self.time, self.time],
                                             id.data_matrix[..., 0],
                                             decimal=3)

    def test_standard_normalize_warping(self):
        """Test normalization to (0, 1)"""

        normalized = normalize_warping(self.polynomial, (0, 1))

        # Test new domain range (0, 1)
        np.testing.assert_array_equal(normalized.domain_range, [(0, 1)])

        np.testing.assert_array_almost_equal(normalized.grid_points[0],
                                             np.linspace(0, 1, 50))

        np.testing.assert_array_almost_equal(
            normalized(0)[..., 0], [[0.], [0.]])

        np.testing.assert_array_almost_equal(
            normalized(1)[..., 0], [[1.], [1.]])

    def test_standard_normalize_warping_default_value(self):
        """Test normalization """

        normalized = normalize_warping(self.polynomial)

        # Test new domain range (0, 1)
        np.testing.assert_array_equal(normalized.domain_range, [(-1, 1)])

        np.testing.assert_array_almost_equal(normalized.grid_points[0],
                                             np.linspace(-1, 1, 50))

        np.testing.assert_array_almost_equal(
            normalized(-1)[..., 0], [[-1], [-1]])

        np.testing.assert_array_almost_equal(
            normalized(1)[..., 0], [[1.], [1.]])

    def test_normalize_warping(self):
        """Test normalization to (a, b)"""
        a = -4
        b = 3
        domain = (a, b)
        normalized = normalize_warping(self.polynomial, domain)

        # Test new domain range (0, 1)
        np.testing.assert_array_equal(normalized.domain_range, [domain])

        np.testing.assert_array_almost_equal(normalized.grid_points[0],
                                             np.linspace(*domain, 50))

        np.testing.assert_array_equal(normalized(a)[..., 0], [[a], [a]])

        np.testing.assert_array_equal(normalized(b)[..., 0], [[b], [b]])

    def test_landmark_shift_deltas(self):

        fd = make_multimodal_samples(n_samples=3, random_state=1)
        landmarks = make_multimodal_landmarks(n_samples=3, random_state=1)
        landmarks = landmarks.squeeze()

        shifts = landmark_shift_deltas(fd, landmarks).round(3)
        np.testing.assert_almost_equal(shifts, [0.25, -0.25, -0.231])

    def test_landmark_shift(self):

        fd = make_multimodal_samples(n_samples=3, random_state=1)
        landmarks = make_multimodal_landmarks(n_samples=3, random_state=1)
        landmarks = landmarks.squeeze()

        original_modes = fd(landmarks.reshape((3, 1, 1)),
                            aligned=False)
        # Test default location
        fd_registered = landmark_shift(fd, landmarks)
        center = (landmarks.max() + landmarks.min()) / 2
        reg_modes = fd_registered(center)

        # Test callable location
        np.testing.assert_almost_equal(reg_modes, original_modes, decimal=2)

        fd_registered = landmark_shift(fd, landmarks, location=np.mean)
        center = np.mean(landmarks)
        reg_modes = fd_registered(center)

        np.testing.assert_almost_equal(reg_modes, original_modes, decimal=2)

        # Test integer location
        fd_registered = landmark_shift(fd, landmarks, location=0)
        center = np.mean(landmarks)
        reg_modes = fd_registered(0)

        np.testing.assert_almost_equal(reg_modes, original_modes, decimal=2)

        # Test array location
        fd_registered = landmark_shift(fd, landmarks, location=[0, 0.1, 0.2])
        reg_modes = fd_registered([[0], [.1], [.2]], aligned=False)

        np.testing.assert_almost_equal(reg_modes, original_modes, decimal=2)

    def test_landmark_registration_warping(self):
        fd = make_multimodal_samples(n_samples=3, n_modes=2, random_state=9)
        landmarks = make_multimodal_landmarks(n_samples=3, n_modes=2,
                                              random_state=9)
        landmarks = landmarks.squeeze()

        # Default location
        warping = landmark_registration_warping(fd, landmarks)
        center = (landmarks.max(axis=0) + landmarks.min(axis=0)) / 2
        np.testing.assert_almost_equal(
            warping(center)[..., 0], landmarks, decimal=1)

        # Fixed location
        center = [.3, .6]
        warping = landmark_registration_warping(fd, landmarks, location=center)
        np.testing.assert_almost_equal(
            warping(center)[..., 0], landmarks, decimal=3)

    def test_landmark_registration(self):
        fd = make_multimodal_samples(n_samples=3, n_modes=2, random_state=9)
        landmarks = make_multimodal_landmarks(n_samples=3, n_modes=2,
                                              random_state=9)
        landmarks = landmarks.squeeze()

        original_values = fd(landmarks.reshape(3, 2), aligned=False)

        # Default location
        fd_reg = landmark_registration(fd, landmarks)
        center = (landmarks.max(axis=0) + landmarks.min(axis=0)) / 2
        np.testing.assert_almost_equal(fd_reg(center), original_values,
                                       decimal=2)

        # Fixed location
        center = [.3, .6]
        fd_reg = landmark_registration(fd, landmarks, location=center)
        np.testing.assert_array_almost_equal(fd_reg(center), original_values,
                                             decimal=2)


class TestShiftRegistration(unittest.TestCase):
    """Test shift registration"""

    def setUp(self):
        """Initialization of samples"""
        self.fd = make_sinusoidal_process(n_samples=2, error_std=0,
                                          random_state=1)
        self.fd.extrapolation = "periodic"

    def test_fit_transform(self):

        reg = ShiftRegistration()

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
        fd = self.fd.to_basis(Fourier())
        reg.fit_transform(fd)
        deltas = reg.deltas_.round(3)
        np.testing.assert_array_almost_equal(deltas, [-0.022, 0.03])

    def test_fit_and_transform(self):
        """Test wrapper of shift_registration_deltas"""

        fd = make_sinusoidal_process(n_samples=2, error_std=0, random_state=10)

        reg = ShiftRegistration()
        response = reg.fit(self.fd)

        # Check attributes and returned value
        self.assertTrue(hasattr(reg, 'template_'))
        self.assertTrue(response is reg)

        fd_registered = reg.transform(fd)
        deltas = reg.deltas_.round(3)
        np.testing.assert_allclose(deltas, [0.071, -0.072])

    def test_inverse_transform(self):

        reg = ShiftRegistration()
        fd = reg.fit_transform(self.fd)
        fd = reg.inverse_transform(fd)

        np.testing.assert_array_almost_equal(fd.data_matrix,
                                             self.fd.data_matrix, decimal=3)

    def test_raises(self):

        reg = ShiftRegistration()

        # Test not fitted
        with np.testing.assert_raises(NotFittedError):
            reg.transform(self.fd)

        reg.fit(self.fd)
        reg.set_params(restrict_domain=True)

        # Test use fit or transform with restrict_domain=True
        with np.testing.assert_raises(AttributeError):
            reg.transform(self.fd)

        with np.testing.assert_raises(AttributeError):
            reg.fit(self.fd)

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

    def test_template(self):

        reg = ShiftRegistration()
        fd_registered_1 = reg.fit_transform(self.fd)

        reg_2 = ShiftRegistration(template=reg.template_)
        fd_registered_2 = reg_2.fit_transform(self.fd)

        reg_3 = ShiftRegistration(template=mean)
        fd_registered_3 = reg_3.fit_transform(self.fd)

        reg_4 = ShiftRegistration(template=reg.template_)
        fd_registered_4 = reg_4.fit(self.fd).transform(self.fd)

        np.testing.assert_array_almost_equal(fd_registered_1.data_matrix,
                                             fd_registered_3.data_matrix)

        # With the template fixed could vary the convergence
        np.testing.assert_array_almost_equal(fd_registered_1.data_matrix,
                                             fd_registered_2.data_matrix,
                                             decimal=3)

        np.testing.assert_array_almost_equal(fd_registered_2.data_matrix,
                                             fd_registered_4.data_matrix)

    def test_restrict_domain(self):
        reg = ShiftRegistration(restrict_domain=True)
        fd_registered_1 = reg.fit_transform(self.fd)

        np.testing.assert_array_almost_equal(
            np.array(fd_registered_1.domain_range).round(3), [[0.022, 0.969]])

        reg2 = ShiftRegistration(restrict_domain=True, template=reg.template_)
        fd_registered_2 = reg2.fit_transform(self.fd)

        np.testing.assert_array_almost_equal(
            fd_registered_2.data_matrix, fd_registered_1.data_matrix,
            decimal=3)

        reg3 = ShiftRegistration(restrict_domain=True, template=mean)
        fd_registered_3 = reg3.fit_transform(self.fd)

        np.testing.assert_array_almost_equal(
            fd_registered_3.data_matrix, fd_registered_1.data_matrix)

    def test_initial_estimation(self):
        reg = ShiftRegistration(initial=[-0.02161235, 0.03032652])
        reg.fit_transform(self.fd)

        # Only needed 1 iteration until convergence
        self.assertEqual(reg.n_iter_, 1)

    def test_custom_output_points(self):
        reg = ShiftRegistration(output_points=np.linspace(0, 1, 50))
        reg.fit_transform(self.fd)


class TestRegistrationValidation(unittest.TestCase):
    """Test shift registration"""

    def setUp(self):
        """Initialization of samples"""
        self.X = make_sinusoidal_process(error_std=0, random_state=0)
        self.shift_registration = ShiftRegistration().fit(self.X)

    def test_amplitude_phase_score(self):
        scorer = AmplitudePhaseDecomposition()
        score = scorer(self.shift_registration, self.X)
        np.testing.assert_allclose(score, 0.972095, rtol=1e-6)

    def test_amplitude_phase_score_with_output_points(self):
        eval_points = self.X.grid_points[0]
        scorer = AmplitudePhaseDecomposition(eval_points=eval_points)
        score = scorer(self.shift_registration, self.X)
        np.testing.assert_allclose(score, 0.972095, rtol=1e-6)

    def test_amplitude_phase_score_with_basis(self):
        scorer = AmplitudePhaseDecomposition()
        X = self.X.to_basis(Fourier())
        score = scorer(self.shift_registration, X)
        np.testing.assert_allclose(score, 0.995087, rtol=1e-6)

    def test_default_score(self):

        score = self.shift_registration.score(self.X)
        np.testing.assert_allclose(score, 0.972095, rtol=1e-6)

    def test_least_squares_score(self):
        scorer = LeastSquares()
        score = scorer(self.shift_registration, self.X)
        np.testing.assert_allclose(score, 0.795933, rtol=1e-6)

    def test_sobolev_least_squares_score(self):
        scorer = SobolevLeastSquares()
        score = scorer(self.shift_registration, self.X)
        np.testing.assert_allclose(score, 0.76124, rtol=1e-6)

    def test_pairwise_correlation(self):
        scorer = PairwiseCorrelation()
        score = scorer(self.shift_registration, self.X)
        np.testing.assert_allclose(score, 1.816228, rtol=1e-6)

    def test_mse_decomposition(self):

        fd = make_multimodal_samples(n_samples=3, random_state=1)
        landmarks = make_multimodal_landmarks(n_samples=3, random_state=1)
        landmarks = landmarks.squeeze()
        warping = landmark_registration_warping(fd, landmarks)
        fd_registered = fd.compose(warping)
        scorer = AmplitudePhaseDecomposition(return_stats=True)
        ret = scorer.score_function(fd, fd_registered, warping=warping)
        np.testing.assert_allclose(ret.mse_amp, 0.0009866997121476962)
        np.testing.assert_allclose(ret.mse_pha, 0.11576935495450151)
        np.testing.assert_allclose(ret.r_squared, 0.9915489952877273)
        np.testing.assert_allclose(ret.c_r, 0.999999, rtol=1e-6)

    def test_raises_amplitude_phase(self):
        scorer = AmplitudePhaseDecomposition()

        # Inconsistent number of functions registered
        with np.testing.assert_raises(ValueError):
            scorer.score_function(self.X, self.X[:2])

        # Inconsistent number of functions registered
        with np.testing.assert_raises(ValueError):
            scorer.score_function(self.X, self.X, warping=self.X[:2])


if __name__ == '__main__':
    print()
    unittest.main()
