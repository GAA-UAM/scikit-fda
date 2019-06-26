import unittest

import numpy as np

import matplotlib.pyplot as plt

from skfda import FDataGrid
from skfda.datasets import make_multimodal_samples
from skfda.misc.metrics import (fisher_rao_distance, amplitude_distance,
                                phase_distance, pairwise_distance, lp_distance,
                                warping_distance)
from skfda.preprocessing.registration import (
    elastic_registration, to_srsf, from_srsf,
    elastic_registration_warping, invert_warping, normalize_warping)

metric = pairwise_distance(lp_distance)
pairwise_fisher_rao = pairwise_distance(fisher_rao_distance)


class TestElasticRegistration(unittest.TestCase):
    """Test elastic registration"""


    def setUp(self):
        """Initialization of samples"""
        template = make_multimodal_samples(n_samples=1, std=0, random_state=1)
        self.template = template
        self.template_rep = template.concatenate(template).concatenate(template)
        self.unimodal_samples = make_multimodal_samples(n_samples=3,
                                                        random_state=1)

        t = np.linspace(-3, 3, 9)
        self.dummy_sample = FDataGrid([np.sin(t)], t)


    def test_to_srsf(self):
        """Test to srsf"""
        # Checks SRSF conversion
        srsf = to_srsf(self.dummy_sample)

        data_matrix = [[[-0.92155896], [-0.75559027], [ 0.25355399],
                        [ 0.81547327], [ 0.95333713], [ 0.81547327],
                        [ 0.25355399], [-0.75559027], [-0.92155896]]]

        np.testing.assert_almost_equal(data_matrix, srsf.data_matrix)


    def test_from_srsf(self):
        """Test from srsf"""

        # Checks SRSF conversion
        srsf = from_srsf(self.dummy_sample)

        data_matrix = [[[ 0.        ], [-0.23449228], [-0.83464009],
                        [-1.38200046], [-1.55623723], [-1.38200046],
                        [-0.83464009], [-0.23449228], [ 0.        ]]]

        np.testing.assert_almost_equal(data_matrix, srsf.data_matrix)


    def test_srsf_conversion(self):
        """Converts to srsf and pull backs"""
        initial = self.unimodal_samples(-1)
        converted = from_srsf(to_srsf(self.unimodal_samples), initial=initial)

        # Distances between original samples and s -> to_srsf -> from_srsf
        distances = np.diag(metric(converted, self.unimodal_samples))

        np.testing.assert_allclose(distances, 0, atol=8e-3)


    def test_template_alignment(self):
        """Test alignment to 1 template"""
        register = elastic_registration(self.unimodal_samples, self.template)
        distances = metric(self.template, register)

        np.testing.assert_allclose(distances, 0, atol=12e-3)

    def test_one_to_one_alignment(self):
        """Test alignment to 1 sample to a template"""
        register = elastic_registration(self.unimodal_samples[0], self.template)
        distances = metric(self.template, register)

        np.testing.assert_allclose(distances, 0, atol=12e-3)


    def test_set_alignment(self):
        """Test alignment 3 curves to set with 3 templates"""
        # Should give same result than test_template_alignment
        register = elastic_registration(self.unimodal_samples,
                                        self.template_rep)
        distances = metric(self.template, register)

        np.testing.assert_allclose(distances, 0, atol=12e-3)


    def test_simetry_of_aligment(self):
        """Check registration using inverse composition"""
        warping = elastic_registration_warping(self.unimodal_samples,
                                               self.template)
        inverse = invert_warping(warping)
        register = self.template_rep.compose(inverse)
        distances = np.diag(metric(self.unimodal_samples, register))

        np.testing.assert_allclose(distances, 0, atol=12e-3)

class TestElasticDistances(unittest.TestCase):
    """Test elastic distances"""

    def test_fisher_rao(self):
        """Test fisher rao distance"""

        t = np.linspace(0, 1, 100)
        sample  = FDataGrid([t, 1-t], t)
        f = np.square(sample)
        g = np.power(sample, 0.5)

        distance = [[0.62825868, 1.98009242], [1.98009242, 0.62825868]]
        res = pairwise_fisher_rao(f, g)

        np.testing.assert_almost_equal(res, distance, decimal=3)

    def test_fisher_rao_invariance(self):
        """Test invariance of fisher rao metric: d(f,g)= d(foh, goh)"""

        t = np.linspace(0, np.pi)
        id = FDataGrid([t], t)
        cos = np.cos(id)
        sin = np.sin(id)
        gamma = normalize_warping(np.sqrt(id), (0, np.pi))
        gamma2 = normalize_warping(np.square(id), (0, np.pi))

        distance_original = fisher_rao_distance(cos, sin)

        # Construction of 2 warpings
        distance_warping = fisher_rao_distance(cos.compose(gamma),
                                               sin.compose(gamma))
        distance_warping2 = fisher_rao_distance(cos.compose(gamma2),
                                               sin.compose(gamma2))

        # The error ~0.001 due to the derivation
        np.testing.assert_almost_equal(distance_original, distance_warping,
                                       decimal=2)

        np.testing.assert_almost_equal(distance_original, distance_warping2,
                                       decimal=2)

    def test_amplitude_distance_limit(self):
        """Test limit of amplitude distance penalty"""

        f = make_multimodal_samples(n_samples=1, random_state=1)
        g = make_multimodal_samples(n_samples=1, random_state=9999)

        amplitude_limit = amplitude_distance(f, g, lam=1000)
        fr_distance = fisher_rao_distance(f, g)

        np.testing.assert_almost_equal(amplitude_limit, fr_distance)


    def test_phase_distance_id(self):
        """Test of phase distance invariance"""
        f = make_multimodal_samples(n_samples=1, random_state=1)

        phase = phase_distance(f, 2*f)

        np.testing.assert_allclose(phase, 0, atol=1e-7)

    def test_warping_distance(self):
        """Test of warping distance"""
        t = np.linspace(0, 1)
        w1 = FDataGrid([t**5], t)
        w2 = FDataGrid([t**3], t)

        d = warping_distance(w1, w2)
        np.testing.assert_allclose(d,  np.arccos(np.sqrt(15)/4), atol=1e-3)

        d = warping_distance(w2, w2)
        np.testing.assert_allclose(d, 0, atol=2e-2)




if __name__ == '__main__':
    print()
    unittest.main()
