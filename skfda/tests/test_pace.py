import numpy as np
import pytest
from skfda.representation.irregular import FDataIrregular
from skfda.preprocessing.dim_reduction import PACE


@pytest.fixture
def fdata_2d_points_2d_values() -> FDataIrregular:
    """Generate a 2D irregular functional dataset with 2D values."""
    np.random.seed(42)

    n_samples = 5
    n_points_per_sample = 10
    domain_dim = 2
    codomain_dim = 2

    all_points = []
    all_values = []
    start_indices = [0]

    for _ in range(n_samples):
        t = np.random.rand(n_points_per_sample, domain_dim)
        y1 = np.sin(2 * np.pi * t[:, 0]) + np.cos(2 * np.pi * t[:, 1])
        y2 = np.cos(4 * np.pi * t[:, 0]) * np.sin(4 * np.pi * t[:, 1])
        y = np.stack([y1, y2], axis=1) + 0.05 * np.random.randn(n_points_per_sample, codomain_dim)

        all_points.append(t)
        all_values.append(y)
        start_indices.append(start_indices[-1] + n_points_per_sample)

    return FDataIrregular(
        points=np.vstack(all_points),
        values=np.vstack(all_values),
        start_indices=np.array(start_indices[:-1], dtype=np.uint32),
        domain_range=[(0, 1), (0, 1)],
    )


def test_pace_fit_with_2d_domain_and_codomain(
    fdata_2d_points_2d_values: FDataIrregular,
) -> None:
    """Test PACE with 2D domain and codomain."""
    pace = PACE(n_components=2, bandwidth_mean=(0.1, 1.0))
    pace.fit(fdata_2d_points_2d_values)

    assert hasattr(pace, "mean_")
    assert isinstance(pace.mean_, np.ndarray)
    assert pace.mean_.shape[1] == 2  # Codominio de dimensión 2

    assert hasattr(pace, "bandwidth_mean_")
    assert 0.1 <= pace.bandwidth_mean_ <= 1.1

    # Asegura que el mean no tenga valores NaN o infinitos
    assert np.all(np.isfinite(pace.mean_))
