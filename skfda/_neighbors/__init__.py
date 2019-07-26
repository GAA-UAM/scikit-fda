"""Private module with the implementation of the neighbors estimators
Includes the following classes estimators:
 - NearestNeighbors
 - KNeighborsClassifier
 - RadiusNeighborsClassifier
 - NearestCentroids
 - KNeighborsScalarRegressor
 - RadiusNeighborsScalarRegressor
 - KNeighborsFunctionalRegressor
 - RadiusNeighborsFunctionalRegressor'
"""

from .unsupervised import NearestNeighbors

from .classification import (KNeighborsClassifier, RadiusNeighborsClassifier,
                             NearestCentroids)
from .regression import (KNeighborsFunctionalRegressor,
                         KNeighborsScalarRegressor,
                         RadiusNeighborsFunctionalRegressor,
                         RadiusNeighborsScalarRegressor)
