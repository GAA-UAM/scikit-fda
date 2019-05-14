import unittest

import numpy as np

import matplotlib.pyplot as plt

from skfda import FDataGrid
from skfda.representation.interpolation import SplineInterpolator
from skfda.representation.basis import Fourier
from skfda.datasets import (make_multimodal_samples, make_multimodal_landmarks,
                            make_sinusoidal_process)
from skfda.preprocessing.registration import (
    normalize_warping, invert_warping, landmark_shift_deltas, landmark_shift,
    landmark_registration_warping, landmark_registration, mse_decomposition,
    shift_registration_deltas, shift_registration)


class TestWarping(unittest.TestCase):
    """Test warpings functions"""

    def setUp(self):
        """Initialization of samples"""

        self.time = np.linspace(-1, 1, 50)
        interpolator = SplineInterpolator(3, monotone=True)
        self.polynomial = FDataGrid([self.time**3, self.time**5],
                                    self.time, interpolator=interpolator)

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

        np.testing.assert_array_almost_equal(normalized.sample_points[0],
                                             np.linspace(0, 1, 50))

        np.testing.assert_array_almost_equal(normalized(0), [[0.], [0.]])

        np.testing.assert_array_almost_equal(normalized(1), [[1.], [1.]])

    def test_normalize_warpig(self):
        """Test normalization to (a, b)"""
        a = -4
        b = 3
        domain = (a, b)
        normalized = normalize_warping(self.polynomial, domain)

        # Test new domain range (0, 1)
        np.testing.assert_array_equal(normalized.domain_range, [domain])

        np.testing.assert_array_almost_equal(normalized.sample_points[0],
                                             np.linspace(*domain, 50))

        np.testing.assert_array_equal(normalized(a), [[a], [a]])

        np.testing.assert_array_equal(normalized(b), [[b], [b]])

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
                            aligned_evaluation=False)
        # Test default location
        fd_registered = landmark_shift(fd, landmarks)
        center = (landmarks.max() + landmarks.min())/2
        reg_modes = fd_registered(center)

        # Test callable location
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

        # Test array location
        fd_registered = landmark_shift(fd, landmarks, location=[0, 0.1, 0.2])
        reg_modes = fd_registered([[0], [.1], [.2]], aligned_evaluation=False)

        np.testing.assert_almost_equal(reg_modes, original_modes, decimal=2)

    def test_landmark_registration_warping(self):
        fd = make_multimodal_samples(n_samples=3, n_modes=2, random_state=9)
        landmarks = make_multimodal_landmarks(n_samples=3, n_modes=2,
                                              random_state=9)
        landmarks = landmarks.squeeze()

        # Default location
        warping = landmark_registration_warping(fd, landmarks)
        center = (landmarks.max(axis=0) + landmarks.min(axis=0)) / 2
        np.testing.assert_almost_equal(warping(center), landmarks, decimal=1)

        # Fixed location
        center = [.3, .6]
        warping = landmark_registration_warping(fd, landmarks, location=center)
        np.testing.assert_almost_equal(warping(center), landmarks, decimal=3)

    def test_landmark_registration(self):
        fd = make_multimodal_samples(n_samples=3, n_modes=2, random_state=9)
        landmarks = make_multimodal_landmarks(n_samples=3, n_modes=2,
                                              random_state=9)
        landmarks = landmarks.squeeze()

        original_values = fd(landmarks.reshape(3, 2), aligned_evaluation=False)

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

    def test_mse_decomposition(self):
        fd = make_multimodal_samples(n_samples=3, random_state=1)
        landmarks = make_multimodal_landmarks(n_samples=3, random_state=1)
        landmarks = landmarks.squeeze()
        warping = landmark_registration_warping(fd, landmarks)
        fd_registered = fd.compose(warping)
        ret = mse_decomposition(fd, fd_registered, warping)

        np.testing.assert_almost_equal(ret.mse_amp, 0.0009866997121476962)
        np.testing.assert_almost_equal(ret.mse_pha, 0.11576861468435257)
        np.testing.assert_almost_equal(ret.rsq, 0.9915489952877273)
        np.testing.assert_almost_equal(ret.cr, 0.9999963424653829)

    def test_shift_registration_deltas(self):

        fd = make_sinusoidal_process(n_samples=2, error_std=0, random_state=1)

        deltas = shift_registration_deltas(fd).round(3)
        np.testing.assert_array_almost_equal(deltas, [-0.022,  0.03])

        fd = fd.to_basis(Fourier())
        deltas = shift_registration_deltas(fd).round(3)
        np.testing.assert_array_almost_equal(deltas, [-0.022,  0.03])

    def test_shift_registration(self):
        """Test wrapper of shift_registration_deltas"""

        fd = make_sinusoidal_process(n_samples=2, error_std=0, random_state=1)

        fd_reg = shift_registration(fd)
        deltas = shift_registration_deltas(fd)
        np.testing.assert_array_almost_equal(fd_reg.data_matrix,
                                             fd.shift(deltas).data_matrix)


if __name__ == '__main__':
    print()
    unittest.main()
