"""Classification."""

from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_centroid_classifiers": [
            "DTMClassifier",
            "NearestCentroid",
        ],
        "_depth_classifiers": [
            "DDClassifier",
            "DDGClassifier",
            "MaximumDepthClassifier",
        ],
        "_logistic_regression": ["LogisticRegression"],
        "_neighbors_classifiers": [
            "KNeighborsClassifier",
            "RadiusNeighborsClassifier",
        ],
        "_qda": ["QuadraticDiscriminantAnalysis"],
    },
)

if TYPE_CHECKING:
    from ._centroid_classifiers import (
        DTMClassifier as DTMClassifier,
        NearestCentroid as NearestCentroid,
    )
    from ._depth_classifiers import (
        DDClassifier as DDClassifier,
        DDGClassifier as DDGClassifier,
        MaximumDepthClassifier as MaximumDepthClassifier,
    )
    from ._logistic_regression import LogisticRegression as LogisticRegression
    from ._neighbors_classifiers import (
        KNeighborsClassifier as KNeighborsClassifier,
        RadiusNeighborsClassifier as RadiusNeighborsClassifier,
    )
    from ._qda import (
        QuadraticDiscriminantAnalysis as QuadraticDiscriminantAnalysis,
    )
