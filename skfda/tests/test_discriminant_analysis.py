"""Tests for LDA and QDA classifiers."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest

from skfda._utils._sklearn_adapter import ClassifierMixin
from skfda.datasets import make_gaussian_process
from skfda.exploratory.stats.covariance import (
    CovarianceEstimator,
    EmpiricalCovariance,
    ParametricGaussianCovariance,
)
from skfda.misc.covariances import Gaussian
from skfda.ml.classification import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from skfda.representation import FData, FDataBasis, FDataGrid
from skfda.representation.basis import Basis, BSplineBasis, FourierBasis

from ..typing._numpy import NDArrayAny, NDArrayFloat

############################
# FIXTURES FOR CLASSIFIERS
############################

DOMAIN = (0, 1)


@pytest.fixture(
    params=[
        FourierBasis(
            domain_range=DOMAIN,
            n_basis=10,
        ),
        BSplineBasis(
            domain_range=DOMAIN,
            n_basis=10,
        ),
    ],
    ids=lambda basis: type(basis).__name__,
)
def basis(request: Any) -> Any:
    """Fixture for bases to test."""
    return request.param


@pytest.fixture(
    params=[
        EmpiricalCovariance(
            regularization_parameter=0.5,
        ),
        ParametricGaussianCovariance(
            Gaussian(),
        ),
    ],
    ids=lambda cov: type(cov).__name__,
)
def cov_estimator_grid(request: Any) -> Any:
    """Fixture for covariance estimators to test."""
    return request.param


@pytest.fixture(
    params=[
        EmpiricalCovariance(
            regularization_parameter=0.5,
        ),
    ],
    ids=lambda cov: type(cov).__name__,
)
def cov_estimator_basis(request: Any) -> Any:
    """Fixture for covariance estimators to test."""
    return request.param


@pytest.fixture(
    params=[
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
    ],
)
def classifier_class(request: Any) -> Any:
    """Fixture for classifiers to test."""
    return request.param


@pytest.fixture
def classifier(
    classifier_class: Callable[..., ClassifierMixin[FDataGrid, NDArrayAny]],
    cov_estimator_grid: CovarianceEstimator[FDataGrid],
) -> ClassifierMixin[FDataGrid, NDArrayAny]:
    """Fixture for classifiers to test."""
    return classifier_class(cov_estimator=cov_estimator_grid)


@pytest.fixture
def classifier_basis(
    classifier_class: Callable[..., ClassifierMixin[FDataBasis, NDArrayAny]],
    cov_estimator_basis: CovarianceEstimator[FDataBasis],
) -> ClassifierMixin[FDataBasis, NDArrayAny]:
    """Fixture for classifiers to test."""
    return classifier_class(cov_estimator=cov_estimator_basis)


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


@pytest.fixture(
    params=[
        30,
    ],
)
def n_samples_per_class(request: Any) -> Any:
    """Fixture for number of samples per class to test."""
    return request.param

#####################
# FIXTURES FOR DATA
#####################

# Auxiliary functions for data generation


def make_gaussian_data(
    classes: NDArrayAny,
    means: NDArrayFloat,
    n_samples_per_class: int,
    seed: int,
) -> tuple[FData, NDArrayAny]:
    """Generate gaussian data."""
    if len(classes) != len(means):
        raise ValueError("classes and means must have the same length")
    data_X = [
        make_gaussian_process(
            n_samples=n_samples_per_class,
            mean=mean,
            random_state=seed,
        )
        for mean in means
    ]
    data_Y = [
        np.full(n_samples_per_class, class_)
        for class_ in classes
    ]
    X = data_X[0].concatenate(*data_X[1:])
    y = np.concatenate(data_Y)

    return X, y


@pytest.fixture
def perfect_separable_data(
    classes: NDArrayAny,
    n_samples_per_class: int,
) -> tuple[FData, NDArrayAny]:
    """Fixture for perfect separable data."""
    means = np.linspace(
        start=-(len(classes) - 1) * 2,
        stop=(len(classes) - 1) * 2,
        num=len(classes),
    )
    return make_gaussian_data(
        classes=classes,
        means=means,
        n_samples_per_class=n_samples_per_class,
        seed=0,
    )


@pytest.fixture
def non_separable_data(
    classes: NDArrayAny,
    n_samples_per_class: int,
) -> tuple[FData, NDArrayAny]:
    """Fixture for non separable data."""
    means = np.array([0.0 for _ in classes])
    return make_gaussian_data(
        classes=classes,
        means=means,
        n_samples_per_class=n_samples_per_class,
        seed=0,
    )

#####################
# TESTS
# -------------------
#####################


def test_classes(
    classifier: ClassifierMixin[FData, NDArrayAny],
    perfect_separable_data: tuple[FData, NDArrayAny],
) -> None:
    """Test classes attribute of classifiers.

    This only tests that the classes attribute is correctly set
    after fitting the classifier.
    """
    X, y = perfect_separable_data
    classes = np.unique(y)
    classifier.fit(X, y)
    resulting_classes = np.unique(classifier.predict(X))

    np.testing.assert_array_equal(classifier.classes_, classes)
    np.testing.assert_array_equal(classifier.classes_, resulting_classes)


def test_perfect_predict(
    classifier: ClassifierMixin[FData, NDArrayAny],
    perfect_separable_data: tuple[FData, NDArrayAny],
) -> None:
    """Test fit and predict methods of classifiers.

    Perfect separable data is used to ensure that the classifier
    is able to classify all the samples correctly.
    This is only for edge testing purposes.
    """
    train_X, train_y = perfect_separable_data
    classifier.fit(train_X, train_y)
    test_y = classifier.predict(train_X)

    np.testing.assert_array_equal(train_y, test_y)


def test_error_predict(
    classifier: ClassifierMixin[FData, NDArrayAny],
    non_separable_data: tuple[FData, NDArrayAny],
) -> None:
    """Test error in predict method of classifiers.

    This test ensures that the classifier is not able to classify
    all the samples correctly. To do this, non separable data is used.
    This is only for edge testing purposes.
    """
    train_X, train_y = non_separable_data
    classifier.fit(train_X, train_y)
    test_y = classifier.predict(train_X)

    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        train_y,
        test_y,
    )


def test_predict_probas(
    classifier: ClassifierMixin[FData, NDArrayAny],
    perfect_separable_data: tuple[FData, NDArrayAny],
) -> None:
    """Test predict probas methods of classifiers.

    This test ensures that the classifier is able to predict
    probabilities, that is the output of the predict probas method
    is a matrix of shape (n_samples, n_classes) and that the sum
    of the probabilities of each sample is 1.
    """
    train_X, train_y = perfect_separable_data
    n_samples = train_X.n_samples
    classifier.fit(train_X, train_y)
    test_y = classifier.predict_proba(train_X)

    np.testing.assert_array_almost_equal(
        np.sum(test_y, axis=0),
        np.ones(n_samples),
    )


def test_predict_basis(
    classifier_basis: ClassifierMixin[FData, NDArrayAny],
    perfect_separable_data: tuple[FData, NDArrayAny],
    basis: Basis,
) -> None:
    """Test basis methods for LDA and QDA classifiers.

    This test ensures that the classifier is able to predict
    using FDataBasis objects.
    """
    train_X, train_y = perfect_separable_data
    train_X = train_X.to_basis(basis)
    classifier_basis.fit(train_X, train_y)
    test_y = classifier_basis.predict(train_X)

    np.testing.assert_array_equal(train_y, test_y)
