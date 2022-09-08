from __future__ import annotations

from ..._utils._sklearn_adapter import BaseEstimator, OutlierMixin
from ...representation import FDataGrid
from ...typing._numpy import NDArrayInt
from ..depth import Depth, ModifiedBandDepth
from . import _envelopes


class BoxplotOutlierDetector(
    BaseEstimator,
    OutlierMixin[FDataGrid],
):
    r"""
    Outlier detector using the interquartile range.

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
        >>> out_detector = BoxplotOutlierDetector()
        >>> out_detector.fit_predict(fd)
        array([-1, 1, 1, -1])

    """

    def __init__(
        self,
        *,
        depth_method: Depth[FDataGrid] | None = None,
        factor: float = 1.5,
    ) -> None:
        self.depth_method = depth_method
        self.factor = factor

    def fit(  # noqa: D102
        self,
        X: FDataGrid,
        y: None = None,
    ) -> BoxplotOutlierDetector:

        depth_method = (
            self.depth_method
            if self.depth_method is not None
            else ModifiedBandDepth()
        )
        depth = depth_method(X)
        indices_descending_depth = (-depth).argsort(axis=0)

        # Central region and envelope must be computed for outlier detection
        central_region = _envelopes.compute_region(
            X,
            indices_descending_depth,
            0.5,
        )
        self._central_envelope = _envelopes.compute_envelope(central_region)

        # Non-outlying envelope
        self.non_outlying_threshold_ = _envelopes.non_outlying_threshold(
            self._central_envelope,
            self.factor,
        )

        return self

    def predict(self, X: FDataGrid) -> NDArrayInt:  # noqa: D102
        outliers = _envelopes.predict_outliers(
            X,
            self.non_outlying_threshold_,
        )

        # Predict as scikit-learn outlier detectors
        return ~outliers + outliers * -1
