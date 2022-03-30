"""Classification."""
from ._centroid_classifiers import DTMClassifier, NearestCentroid
from ._depth_classifiers import (
    DDClassifier,
    DDGClassifier,
    MaximumDepthClassifier,
)
from ._parametrized_functional_qda import ParametrizedFunctionalQDA
from ._logistic_regression import LogisticRegression
from ._neighbors_classifiers import (
    KNeighborsClassifier,
    RadiusNeighborsClassifier,
)
