import unittest

import scipy.stats.mstats

import numpy as np
from skfda import FDataGrid, FDataBasis
from skfda.datasets import make_multimodal_samples
from skfda.exploratory import stats
from skfda.misc.metrics import lp_distance, norm_lp, vectorial_norm
from skfda.representation.basis import Monomial


class TestLpMetrics(unittest.TestCase):

    def setUp(self):
        sample_points = [1, 2, 3, 4, 5]
        self.fd = FDataGrid([[2, 3, 4, 5, 6], [1, 4, 9, 16, 25]],
                            sample_points=sample_points)
        basis = Monomial(n_basis=3, domain_range=(1, 5))
        self.fd_basis = FDataBasis(basis, [[1, 1, 0], [0, 0, 1]])
        self.fd_curve = self.fd.concatenate(self.fd, as_coordinates=True)
        self.fd_surface = make_multimodal_samples(n_samples=3, dim_domain=2,
                                                  random_state=0)

    def test_vectorial_norm(self):

        vec = vectorial_norm(self.fd_curve, p=2)
        np.testing.assert_array_almost_equal(vec.data_matrix,
                                             np.sqrt(2) * self.fd.data_matrix)

        vec = vectorial_norm(self.fd_curve, p='inf')
        np.testing.assert_array_almost_equal(vec.data_matrix,
                                             self.fd.data_matrix)

    def test_vectorial_norm_surface(self):

        fd_surface_curve = self.fd_surface.concatenate(self.fd_surface,
                                                       as_coordinates=True)
        vec = vectorial_norm(fd_surface_curve, p=2)
        np.testing.assert_array_almost_equal(
            vec.data_matrix, np.sqrt(2) * self.fd_surface.data_matrix)

        vec = vectorial_norm(fd_surface_curve, p='inf')
        np.testing.assert_array_almost_equal(vec.data_matrix,
                                             self.fd_surface.data_matrix)

    def test_norm_lp(self):

        np.testing.assert_allclose(norm_lp(self.fd, p=1), [16., 41.33333333])
        np.testing.assert_allclose(norm_lp(self.fd, p='inf'), [6, 25])

    def test_norm_lp_curve(self):

        np.testing.assert_allclose(norm_lp(self.fd_curve, p=1, p2=1),
                                   [32., 82.666667])
        np.testing.assert_allclose(norm_lp(self.fd_curve, p='inf', p2='inf'),
                                   [6, 25])

    def test_norm_lp_surface_inf(self):
        np.testing.assert_allclose(norm_lp(self.fd_surface, p='inf').round(5),
                                   [0.99994, 0.99793, 0.99868])

    def test_norm_lp_surface(self):
        # Integration of surfaces not implemented, add test case after
        # implementation
        self.assertEqual(norm_lp(self.fd_surface), NotImplemented)

    def test_lp_error_dimensions(self):
        # Case internal arrays
        with np.testing.assert_raises(ValueError):
            lp_distance(self.fd, self.fd_surface)

        with np.testing.assert_raises(ValueError):
            lp_distance(self.fd, self.fd_curve)

        with np.testing.assert_raises(ValueError):
            lp_distance(self.fd_surface, self.fd_curve)

    def test_lp_error_domain_ranges(self):
        sample_points = [2, 3, 4, 5, 6]
        fd2 = FDataGrid([[2, 3, 4, 5, 6], [1, 4, 9, 16, 25]],
                        sample_points=sample_points)

        with np.testing.assert_raises(ValueError):
            lp_distance(self.fd, fd2)

    def test_lp_error_sample_points(self):
        sample_points = [1, 2, 4, 4.3, 5]
        fd2 = FDataGrid([[2, 3, 4, 5, 6], [1, 4, 9, 16, 25]],
                        sample_points=sample_points)

        with np.testing.assert_raises(ValueError):
            lp_distance(self.fd, fd2)

    def test_lp_grid_basis(self):

        np.testing.assert_allclose(lp_distance(self.fd, self.fd_basis), 0)
        np.testing.assert_allclose(lp_distance(self.fd_basis, self.fd), 0)
        np.testing.assert_allclose(
            lp_distance(self.fd_basis,
                        self.fd_basis, eval_points=[1, 2, 3, 4, 5]), 0)
        np.testing.assert_allclose(lp_distance(self.fd_basis, self.fd_basis),
                                   0)


if __name__ == '__main__':
    print()
    unittest.main()
