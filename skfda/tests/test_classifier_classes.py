"""Tests classes attribute of classifiers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from skfda._utils._sklearn_adapter import ClassifierMixin
from skfda.datasets import make_gaussian_process
from skfda.ml.classification import (
    KNeighborsClassifier,
    RadiusNeighborsClassifier,
)
from skfda.representation import FData

from ..typing._numpy import NDArrayAny


@pytest.fixture(
    params=[
        KNeighborsClassifier(),
        RadiusNeighborsClassifier(),
    ],
    ids=lambda clf: type(clf).__name__,
)
def classifier(request: Any) -> Any:
    """Fixture for classifiers to test."""
    return request.param


@pytest.fixture(
    params=[
        np.array([0, 1]),
        np.array(["class_a", "class_b"]),
    ],
    ids=["int", "str"],
)
def classes(request: Any) -> Any:
    """Fixture for classes to test."""
    return request.param


def test_classes(
    classifier: ClassifierMixin[FData, NDArrayAny],
    classes: NDArrayAny,
) -> None:
    """Test classes attribute of classifiers."""
    n_samples = 30
    y = np.resize(classes, n_samples)
    X = make_gaussian_process(n_samples=n_samples)
    classifier.fit(X, y)
    resulting_classes = np.unique(classifier.predict(X))

    np.testing.assert_array_equal(classifier.classes_, classes)
    np.testing.assert_array_equal(classifier.classes_, resulting_classes)
