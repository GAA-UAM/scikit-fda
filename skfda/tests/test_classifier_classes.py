"""Tests classes attribute of classifiers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from skfda._utils._sklearn_adapter import ClassifierMixin
from skfda.datasets import make_gaussian_process
from skfda.exploratory.depth import ModifiedBandDepth
from skfda.exploratory.stats.covariance import ParametricGaussianCovariance
from skfda.misc.covariances import Gaussian
from skfda.ml.classification import (
    DDClassifier,
    DDGClassifier,
    DTMClassifier,
    KNeighborsClassifier,
    LogisticRegression,
    MaximumDepthClassifier,
    NearestCentroid,
    QuadraticDiscriminantAnalysis,
    RadiusNeighborsClassifier,
)
from skfda.representation import FData

from ..typing._numpy import NDArrayAny


@pytest.fixture(
    params=[
        DDClassifier(degree=2),
        DDGClassifier(
            depth_method=[("mbd", ModifiedBandDepth())],
            multivariate_classifier=KNeighborsClassifier(),
        ),
        DTMClassifier(proportiontocut=0.25),
        KNeighborsClassifier(),
        LogisticRegression(),
        MaximumDepthClassifier(),
        NearestCentroid(),
        QuadraticDiscriminantAnalysis(
            cov_estimator=ParametricGaussianCovariance(
                Gaussian(),
            ),
        ),
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
    X = make_gaussian_process(n_samples=n_samples, random_state=0)
    classifier.fit(X, y)
    resulting_classes = np.unique(classifier.predict(X))

    np.testing.assert_array_equal(classifier.classes_, classes)
    np.testing.assert_array_equal(classifier.classes_, resulting_classes)
