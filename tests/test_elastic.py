from skfda import FDataGrid
from skfda.datasets import make_multimodal_samples, make_random_warping
from skfda.misc.metrics import (fisher_rao_distance, amplitude_distance,
                                phase_distance, pairwise_distance, lp_distance,
                                warping_distance)
from skfda.preprocessing.registration import (ElasticRegistration,
                                              invert_warping,
                                              normalize_warping)
from skfda.preprocessing.registration.elastic import (SRSF, elastic_mean,
                                                      warping_mean)
import unittest

import numpy as np


metric = pairwise_distance(lp_distance)
pairwise_fisher_rao = pairwise_distance(fisher_rao_distance)


class TestElasticRegistration(unittest.TestCase):
    """Test elastic registration"""

    def setUp(self):
        """Initialization of samples"""
        template = make_multimodal_samples(n_samples=1, std=0, random_state=1)
        self.template = template
        self.template_rep = template.concatenate(
            template).concatenate(template)
        self.unimodal_samples = make_multimodal_samples(n_samples=3,
                                                        random_state=1)

        t = np.linspace(-3, 3, 9)
        self.dummy_sample = FDataGrid([np.sin(t)], t)

    def test_to_srsf(self):
        """Test to srsf"""
        # Checks SRSF conversion

        srsf = SRSF().fit_transform(self.dummy_sample)

        data_matrix = [[[-1.061897], [-0.75559027], [0.25355399],
                        [0.81547327], [0.95333713], [0.81547327],
                        [0.25355399], [-0.75559027], [-1.06189697]]]

        np.testing.assert_almost_equal(data_matrix, srsf.data_matrix)

    def test_from_srsf(self):
        """Test from srsf"""

        # Checks SRSF conversion
        srsf = SRSF(initial_value=0).inverse_transform(self.dummy_sample)

        data_matrix = [[[0.], [-0.23449228], [-0.83464009],
                        [-1.38200046], [-1.55623723], [-1.38200046],
                        [-0.83464009], [-0.23449228], [0.]]]

        np.testing.assert_almost_equal(data_matrix, srsf.data_matrix)

    def test_from_srsf_with_output_points(self):
        """Test from srsf"""

        # Checks SRSF conversion
        srsf_transformer = SRSF(
            initial_value=0,
            output_points=self.dummy_sample.grid_points[0])
        srsf = srsf_transformer.inverse_transform(self.dummy_sample)

        data_matrix = [[[0.], [-0.23449228], [-0.83464009],
                        [-1.38200046], [-1.55623723], [-1.38200046],
                        [-0.83464009], [-0.23449228], [0.]]]

        np.testing.assert_almost_equal(data_matrix, srsf.data_matrix)

    def test_srsf_conversion(self):
        """Converts to srsf and pull backs"""

        srsf = SRSF()

        converted = srsf.fit_transform(self.unimodal_samples)
        converted = srsf.inverse_transform(converted)

        # Distances between original samples and s -> to_srsf -> from_srsf
        distances = np.diag(metric(converted, self.unimodal_samples))

        np.testing.assert_allclose(distances, 0, atol=8e-3)

    def test_template_alignment(self):
        """Test alignment to 1 template"""
        reg = ElasticRegistration(template=self.template)
        register = reg.fit_transform(self.unimodal_samples)
        distances = metric(self.template, register)

        np.testing.assert_allclose(distances, 0, atol=12e-3)

    def test_one_to_one_alignment(self):
        """Test alignment to 1 sample to a template"""
        reg = ElasticRegistration(template=self.template)
        register = reg.fit_transform(self.unimodal_samples[0])
        distances = metric(self.template, register)

        np.testing.assert_allclose(distances, 0, atol=12e-3)

    def test_set_alignment(self):
        """Test alignment 3 curves to set with 3 templates"""
        # Should give same result than test_template_alignment
        reg = ElasticRegistration(template=self.template_rep)
        register = reg.fit_transform(self.unimodal_samples)
        distances = metric(self.template, register)

        np.testing.assert_allclose(distances, 0, atol=12e-3)

    def test_default_alignment(self):
        """Test alignment by default"""
        # Should give same result than test_template_alignment
        reg = ElasticRegistration()
        register = reg.fit_transform(self.unimodal_samples)

        values = register([-.25, -.1, 0, .1, .25])

        expected = [[[0.599058],  [0.997427],  [0.772248],
                     [0.412342],  [0.064725]],
                    [[0.626875],  [0.997155],  [0.791649],
                     [0.382181],  [0.050098]],
                    [[0.620992],  [0.997369],  [0.785886],
                     [0.376556],  [0.048804]]]

        np.testing.assert_allclose(values, expected, atol=1e-4)

    def test_callable_alignment(self):
        """Test alignment by default"""
        # Should give same result than test_template_alignment
        reg = ElasticRegistration(template=elastic_mean)
        register = reg.fit_transform(self.unimodal_samples)

        values = register([-.25, -.1, 0, .1, .25])
        expected = [[[0.599058],  [0.997427],  [0.772248],
                     [0.412342],  [0.064725]],
                    [[0.626875],  [0.997155],  [0.791649],
                     [0.382181],  [0.050098]],
                    [[0.620992],  [0.997369],  [0.785886],
                     [0.376556],  [0.048804]]]

        np.testing.assert_allclose(values, expected, atol=1e-4)

    def test_simmetry_of_aligment(self):
        """Check registration using inverse composition"""
        reg = ElasticRegistration(template=self.template)
        reg.fit_transform(self.unimodal_samples)
        warping = reg.warping_
        inverse = invert_warping(warping)
        register = self.template_rep.compose(inverse)
        distances = np.diag(metric(self.unimodal_samples, register))

        np.testing.assert_allclose(distances, 0, atol=12e-3)

    def test_raises(self):
        reg = ElasticRegistration()

        # X not in fit, but template is not an FDataGrid
        with np.testing.assert_raises(ValueError):
            reg.fit()

        # Inverse transform without previous transform
        with np.testing.assert_raises(ValueError):
            reg.inverse_transform(self.unimodal_samples)

        # Inverse transform with different number of samples than transform
        reg.fit_transform(self.unimodal_samples)
        with np.testing.assert_raises(ValueError):
            reg.inverse_transform(self.unimodal_samples[0])

        # FDataGrid as template with n != 1 and n!= n_samples to transform
        reg = ElasticRegistration(template=self.unimodal_samples).fit()
        with np.testing.assert_raises(ValueError):
            reg.transform(self.unimodal_samples[0])

    def test_score(self):
        """Test score method of the transformer"""
        reg = ElasticRegistration()
        reg.fit(self.unimodal_samples)
        score = reg.score(self.unimodal_samples)
        np.testing.assert_almost_equal(score,  0.9994225)

    def test_warping_mean(self):
        warping = make_random_warping(start=-1, random_state=0)
        mean = warping_mean(warping)
        values = mean([-1, -.5, 0, .5, 1])
        expected = [[[-1.], [-0.376241],  [0.136193],  [0.599291],  [1.]]]
        np.testing.assert_array_almost_equal(values, expected)


class TestElasticDistances(unittest.TestCase):
    """Test elastic distances"""

    def test_fisher_rao(self):
        """Test fisher rao distance"""

        t = np.linspace(0, 1, 100)
        sample = FDataGrid([t, 1 - t], t)
        f = np.square(sample)
        g = np.power(sample, 0.5)

        distance = [[0.64, 1.984], [1.984, 0.64]]
        res = pairwise_fisher_rao(f, g)

        np.testing.assert_almost_equal(res, distance, decimal=3)

    def test_fisher_rao_invariance(self):
        """Test invariance of fisher rao metric: d(f,g)= d(foh, goh)"""

        t = np.linspace(0, np.pi, 1000)
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
        np.testing.assert_allclose(distance_original, distance_warping,
                                   atol=0.01)

        np.testing.assert_allclose(distance_original, distance_warping2,
                                   atol=0.01)

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

        phase = phase_distance(f, 2 * f)

        np.testing.assert_allclose(phase, 0, atol=1e-7)

    def test_warping_distance(self):
        """Test of warping distance"""
        t = np.linspace(0, 1, 1000)
        w1 = FDataGrid([t**5], t)
        w2 = FDataGrid([t**3], t)

        d = warping_distance(w1, w2)
        np.testing.assert_allclose(d, np.arccos(np.sqrt(15) / 4), atol=1e-3)

        d = warping_distance(w2, w2)
        np.testing.assert_allclose(d, 0, atol=2e-2)


if __name__ == '__main__':
    print()
    unittest.main()
