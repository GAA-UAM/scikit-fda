import unittest

import numpy as np

import matplotlib.pyplot as plt

from fda import FDataGrid
from fda.datasets import make_multimodal_samples
from fda.metrics import metric
from fda.registration import (elastic_registration, elastic_mean, to_srsf,
                              from_srsf, elastic_registration_warping,
                              invert_warping)


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


if __name__ == '__main__':
    print()
    unittest.main()
