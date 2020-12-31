from sklearn.base import BaseEstimator, OutlierMixin

from . import _envelopes
from ..depth import ModifiedBandDepth


class IQROutlierDetector(BaseEstimator, OutlierMixin):
    r"""Outlier detector using the interquartile range.

    Detects as outliers functions that have one or more points outside
    ``factor`` times the interquartile range plus or minus the central
    envelope, given a functional depth measure. This corresponds to the
    points selected as outliers by the functional boxplot.

    Parameters:
        depth_method (Callable): The functional depth measure used.
        factor (float): The number of times the IQR is multiplied.

    Example:
        Function :math:`f : \mathbb{R}\longmapsto\mathbb{R}`.

        >>> import skfda
        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2],
        ...                [0.5, 0.5, 1, 2, 1.5, 1],
        ...                [-1, -1, -0.5, 1, 1, 0.5],
        ...                [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> grid_points = [0, 2, 4, 6, 8, 10]
        >>> fd = skfda.FDataGrid(data_matrix, grid_points)
        >>> out_detector = IQROutlierDetector()
        >>> out_detector.fit_predict(fd)
        array([-1, 1, 1, -1])

    """

    def __init__(self, *, depth_method=ModifiedBandDepth(), factor=1.5):
        self.depth_method = depth_method
        self.factor = factor

    def fit(self, X, y=None):
        depth = self.depth_method(X)
        indices_descending_depth = (-depth).argsort(axis=0)

        # Central region and envelope must be computed for outlier detection
        central_region = _envelopes._compute_region(
            X, indices_descending_depth, 0.5)
        self._central_envelope = _envelopes._compute_envelope(central_region)

        # Non-outlying envelope
        self.non_outlying_threshold_ = _envelopes._non_outlying_threshold(
            self._central_envelope, self.factor)

        return self

    def predict(self, X):
        outliers = _envelopes._predict_outliers(
            X, self.non_outlying_threshold_)

        # Predict as scikit-learn outlier detectors
        predicted = ~outliers + outliers * -1

        return predicted
