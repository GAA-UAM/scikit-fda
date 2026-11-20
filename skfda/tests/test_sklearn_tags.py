"""Tests for scikit-learn tag compatibility."""

from sklearn.utils import get_tags

from skfda.representation.basis import FourierBasis
from skfda.representation.conversion._mixed_effects import (
    EMMixedEffectsConverter,
)


def test_mixed_effects_converter_sklearn_tags_are_available() -> None:
    """Test that sklearn tags can be obtained from the converter."""
    converter = EMMixedEffectsConverter(FourierBasis(n_basis=3))

    assert get_tags(converter) is not None
