"""Private module with the implementation of the neighbors estimators
Includes the following classes:
 - NearestNeighbors
 - KNeighborsClassifier
 - RadiusNeighborsClassifier
 - NearestCentroids
 - KNeighborsRegressor
 - RadiusNeighborsRegressor

"""
from .unsupervised import NearestNeighbors
from .regression import KNeighborsRegressor, RadiusNeighborsRegressor
from .classification import (KNeighborsClassifier, RadiusNeighborsClassifier,
                             NearestCentroids)
