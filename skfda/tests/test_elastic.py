"""Tests for elastic registration and functions in the SRVF framework."""

import unittest

import numpy as np

from skfda import FDataGrid
from skfda._utils import invert_warping, normalize_warping
from skfda.datasets import make_multimodal_samples, make_random_warping
from skfda.exploratory.stats import (
    _fisher_rao_warping_mean,
    fisher_rao_karcher_mean,
)
from skfda.misc.metrics import (
    PairwiseMetric,
    _fisher_rao_warping_distance,
    fisher_rao_amplitude_distance,
    fisher_rao_distance,
    fisher_rao_phase_distance,
    l2_distance,
)
from skfda.misc.operators import SRSF
from skfda.preprocessing.registration import FisherRaoElasticRegistration

metric = PairwiseMetric(l2_distance)
pairwise_fisher_rao = PairwiseMetric(fisher_rao_distance)


class TestFisherRaoElasticRegistration(unittest.TestCase):
    """Test elastic registration."""

    def setUp(self) -> None:
        """Initialize the samples."""
        template = make_multimodal_samples(n_samples=1, std=0, random_state=1)
        self.template = template
        self.template_rep = template.concatenate(
            template,
        ).concatenate(template)
        self.unimodal_samples = make_multimodal_samples(
            n_samples=3,
            random_state=1,
        )

        t = np.linspace(-3, 3, 9)
        self.dummy_sample = FDataGrid([np.sin(t)], t)

    def test_fit_wrong_dimension(self) -> None:
        """Checks that template and fit data is compatible."""
        reg = FisherRaoElasticRegistration(template=self.template)

        unimodal_samples = make_multimodal_samples(
            n_samples=3,
            points_per_dim=10,
            random_state=1,
        )

        with self.assertRaises(ValueError):
            reg.fit(unimodal_samples)

    def test_transform_wrong_dimension(self) -> None:
        """Checks that template and transform data is compatible."""
        reg = FisherRaoElasticRegistration(template=self.template)

        unimodal_samples = make_multimodal_samples(
            n_samples=3,
            points_per_dim=10,
            random_state=1,
        )

        reg.fit(self.unimodal_samples)
        with self.assertRaises(ValueError):
            reg.transform(unimodal_samples)

    def test_to_srsf(self) -> None:
        """Test to srsf."""
        # Checks SRSF conversion
        srsf = SRSF().fit_transform(self.dummy_sample)

        data_matrix = [
            [  # noqa: WPS317
                [-1.061897], [-0.75559027], [0.25355399],
                [0.81547327], [0.95333713], [0.81547327],
                [0.25355399], [-0.75559027], [-1.06189697],
            ],
        ]

        np.testing.assert_almost_equal(data_matrix, srsf.data_matrix)

    def test_from_srsf(self) -> None:
        """Test from srsf."""
        # Checks SRSF conversion
        srsf = SRSF(initial_value=0).inverse_transform(self.dummy_sample)

        data_matrix = [
            [  # noqa: WPS317
                [0.0], [-0.23449228], [-0.83464009],
                [-1.38200046], [-1.55623723], [-1.38200046],
                [-0.83464009], [-0.23449228], [0.0],
            ],
        ]

        np.testing.assert_almost_equal(data_matrix, srsf.data_matrix)

    def test_from_srsf_with_output_points(self) -> None:
        """Test from srsf and explicit output points."""
        # Checks SRSF conversion
        srsf_transformer = SRSF(
            initial_value=0,
            output_points=self.dummy_sample.grid_points[0],
        )
        srsf = srsf_transformer.inverse_transform(self.dummy_sample)

        data_matrix = [
            [  # noqa: WPS317
                [0.0], [-0.23449228], [-0.83464009],
                [-1.38200046], [-1.55623723], [-1.38200046],
                [-0.83464009], [-0.23449228], [0.0],
            ],
        ]

        np.testing.assert_almost_equal(data_matrix, srsf.data_matrix)

    def test_srsf_conversion(self) -> None:
        """Convert to srsf and pull back."""
        srsf = SRSF()

        converted = srsf.fit_transform(self.unimodal_samples)
        converted = srsf.inverse_transform(converted)

        # Distances between original samples and s -> to_srsf -> from_srsf
        distances = np.diag(metric(converted, self.unimodal_samples))

        np.testing.assert_allclose(distances, 0, atol=8e-3)

    def test_template_alignment(self) -> None:
        """Test alignment to 1 template."""
        reg = FisherRaoElasticRegistration(template=self.template)
        register = reg.fit_transform(self.unimodal_samples)
        distances = metric(self.template, register)

        np.testing.assert_allclose(distances, 0, atol=12e-3)

    def test_one_to_one_alignment(self) -> None:
        """Test alignment to 1 sample to a template."""
        reg = FisherRaoElasticRegistration(template=self.template)
        register = reg.fit_transform(self.unimodal_samples[0])
        distances = metric(self.template, register)

        np.testing.assert_allclose(distances, 0, atol=12e-3)

    def test_set_alignment(self) -> None:
        """Test alignment 3 curves to set with 3 templates."""
        # Should give same result than test_template_alignment
        reg = FisherRaoElasticRegistration(template=self.template_rep)
        register = reg.fit_transform(self.unimodal_samples)
        distances = metric(self.template, register)

        np.testing.assert_allclose(distances, 0, atol=12e-3)

    def test_default_alignment(self) -> None:
        """Test alignment by default."""
        # Should give same result than test_template_alignment
        reg = FisherRaoElasticRegistration()
        register = reg.fit_transform(self.unimodal_samples)

        values = register([-0.25, -0.1, 0, 0.1, 0.25])

        expected = [
            [
                [0.599058], [0.997427], [0.772248], [0.412342], [0.064725],
            ],
            [
                [0.626875], [0.997155], [0.791649], [0.382181], [0.050098],
            ],
            [
                [0.620992], [0.997369], [0.785886], [0.376556], [0.048804],
            ],
        ]

        np.testing.assert_allclose(values, expected, atol=1e-4)

    def test_callable_alignment(self) -> None:
        """Test callable template."""
        # Should give same result than test_template_alignment
        reg = FisherRaoElasticRegistration(template=fisher_rao_karcher_mean)
        register = reg.fit_transform(self.unimodal_samples)

        values = register([-0.25, -0.1, 0, 0.1, 0.25])
        expected = [
            [
                [0.599058], [0.997427], [0.772248], [0.412342], [0.064725],
            ],
            [
                [0.626875], [0.997155], [0.791649], [0.382181], [0.050098],
            ],
            [
                [0.620992], [0.997369], [0.785886], [0.376556], [0.048804],
            ],
        ]

        np.testing.assert_allclose(values, expected, atol=1e-4)

    def test_simmetry_of_aligment(self) -> None:
        """Check registration using inverse composition."""
        reg = FisherRaoElasticRegistration(template=self.template)
        reg.fit_transform(self.unimodal_samples)
        warping = reg.warping_
        inverse = invert_warping(warping)
        register = self.template_rep.compose(inverse)
        distances = np.diag(metric(self.unimodal_samples, register))

        np.testing.assert_allclose(distances, 0, atol=12e-3)

    def test_raises(self) -> None:
        """Test that the assertions raise when appropriate."""
        reg = FisherRaoElasticRegistration()

        # Inverse transform without previous transform
        with np.testing.assert_raises(ValueError):
            reg.inverse_transform(self.unimodal_samples)

        # Inverse transform with different number of samples than transform
        reg.fit_transform(self.unimodal_samples)
        with np.testing.assert_raises(ValueError):
            reg.inverse_transform(self.unimodal_samples[0])

        # FDataGrid as template with n != 1 and n!= n_samples to transform
        reg = FisherRaoElasticRegistration(template=self.unimodal_samples).fit(
            self.unimodal_samples[0],
        )
        with np.testing.assert_raises(ValueError):
            reg.transform(self.unimodal_samples[0])

    def test_score(self) -> None:
        """Test score method of the transformer."""
        reg = FisherRaoElasticRegistration()
        reg.fit(self.unimodal_samples)
        score = reg.score(self.unimodal_samples)
        np.testing.assert_almost_equal(score, 0.999389)

    def test_warping_mean(self) -> None:
        """Test the warping_mean function."""
        warping = make_random_warping(start=-1, random_state=0)
        mean = _fisher_rao_warping_mean(warping)
        values = mean([-1, -0.5, 0, 0.5, 1])
        expected = [[[-1.0], [-0.376241], [0.136193], [0.599291], [1.0]]]
        np.testing.assert_array_almost_equal(values, expected)

    def test_linear(self) -> None:
        """
        Test alignment of two equal (linear) functions.

        In this case no alignment should take place.

        """
        grid_points = list(range(10))
        data_matrix = np.array([grid_points, grid_points])
        fd = FDataGrid(
            data_matrix=data_matrix,
            grid_points=grid_points,
        )
        elastic_registration = FisherRaoElasticRegistration()
        fd_registered = elastic_registration.fit_transform(fd)
        np.testing.assert_array_almost_equal(
            fd_registered.data_matrix[..., 0],
            data_matrix,
        )


class TestElasticDistances(unittest.TestCase):
    """Test elastic distances."""

    def test_fisher_rao(self) -> None:
        """Test fisher rao distance."""
        t = np.linspace(0, 1, 100)
        sample = FDataGrid([t, 1 - t], t)
        f = np.square(sample)
        g = np.power(sample, 0.5)

        distance = [[0.64, 1.984], [1.984, 0.64]]
        res = pairwise_fisher_rao(f, g)

        np.testing.assert_almost_equal(res, distance, decimal=3)

    def test_fisher_rao_invariance(self) -> None:
        """Test invariance of fisher rao metric: d(f,g)= d(foh, goh)."""
        t = np.linspace(0, np.pi, 1000)
        identity = FDataGrid([t], t)
        cos = np.cos(identity)
        sin = np.sin(identity)
        gamma = normalize_warping(np.sqrt(identity), (0, np.pi))
        gamma2 = normalize_warping(np.square(identity), (0, np.pi))

        distance_original = fisher_rao_distance(cos, sin)

        # Construction of 2 warpings
        distance_warping = fisher_rao_distance(
            cos.compose(gamma),
            sin.compose(gamma),
        )
        distance_warping2 = fisher_rao_distance(
            cos.compose(gamma2),
            sin.compose(gamma2),
        )

        # The error ~0.001 due to the derivation
        np.testing.assert_allclose(
            distance_original,
            distance_warping,
            atol=0.01,
        )

        np.testing.assert_allclose(
            distance_original,
            distance_warping2,
            atol=0.01,
        )

    def test_fisher_rao_amplitude_distance_limit(self) -> None:
        """Test limit of amplitude distance penalty."""
        f = make_multimodal_samples(n_samples=1, random_state=1)
        g = make_multimodal_samples(n_samples=1, random_state=9999)

        amplitude_limit = fisher_rao_amplitude_distance(f, g, lam=1000)
        fr_distance = fisher_rao_distance(f, g)

        np.testing.assert_almost_equal(amplitude_limit, fr_distance)

    def test_fisher_rao_phase_distance_id(self) -> None:
        """Test of phase distance invariance."""
        f = make_multimodal_samples(n_samples=1, random_state=1)

        phase = fisher_rao_phase_distance(f, 2 * f)

        np.testing.assert_allclose(phase, 0, atol=1e-7)

    def test_fisher_rao_warping_distance(self) -> None:
        """Test of warping distance."""
        t = np.linspace(0, 1, 1000)
        w1 = FDataGrid([t**5], t)
        w2 = FDataGrid([t**3], t)

        d = _fisher_rao_warping_distance(w1, w2)
        np.testing.assert_allclose(d, np.arccos(np.sqrt(15) / 4), atol=1e-3)

        d = _fisher_rao_warping_distance(w2, w2)
        np.testing.assert_allclose(d, 0, atol=2e-2)


if __name__ == '__main__':
    unittest.main()
