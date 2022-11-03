"""Function transformers for feature construction techniques."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

from typing_extensions import Literal

from ..._utils._sklearn_adapter import BaseEstimator, TransformerMixin
from ...representation import FData
from ...representation.grid import FDataGrid
from ...typing._base import DomainRangeLike
from ...typing._numpy import ArrayLike, NDArrayFloat, NDArrayInt
from ._functions import local_averages, number_crossings, occupation_measure


class LocalAveragesTransformer(
    BaseEstimator,
    TransformerMixin[FData, NDArrayFloat, object],
):
    r"""
    Transforms functional data to its local averages.

    It takes functional data and performs the following map:

    .. math::
        f_1(X) = \frac{1}{|T_1|} \int_{T_1} X(t) dt,\dots, \\
        f_p(X) = \frac{1}{|T_p|} \int_{T_p} X(t) dt

    where :math:`T_1, \dots, T_p` are subregions of the original
    :term:`domain`.

    Args:
        domains: Domains for each local average. It is possible to
            pass a number or a list of numbers to automatically split
            each dimension in that number of intervals and use them for
            the averages.

    See also:
        :func:`local_averages`

    Examples:
        We import the Berkeley Growth Study dataset.
        We will use only the first 3 samples to make the
        example easy.

        >>> from skfda.datasets import fetch_growth
        >>> dataset = fetch_growth(return_X_y=True)[0]
        >>> X = dataset[:3]

        We can choose the intervals used for the local averages. For example,
        we could in this case use the averages at different stages of
        development of the child: from 1 to 3 years, from 3 to 10 and from
        10 to 18:

        >>> import numpy as np
        >>> from skfda.preprocessing.feature_construction import (
        ...     LocalAveragesTransformer,
        ... )
        >>> local_averages = LocalAveragesTransformer(
        ...     domains=[(1, 3), (3, 10), (10, 18)],
        ... )
        >>> np.round(local_averages.fit_transform(X), decimals=2)
        array([[  91.37,  126.52,  179.02],
               [  87.51,  120.71,  158.81],
               [  86.36,  115.04,  156.37]])

        A different possibility is to decide how many intervals we want to
        consider.  For example, we could want to split the domain in 2
        intervals of the same length.

        >>> local_averages = LocalAveragesTransformer(domains=2)
        >>> np.around(local_averages.fit_transform(X), decimals=2)
        array([[ 116.94,  177.26],
               [ 111.86,  157.62],
               [ 107.29,  154.97]])
    """

    def __init__(
        self,
        *,
        domains: int | Sequence[int] | Sequence[DomainRangeLike],
    ) -> None:
        self.domains = domains

    def transform(self, X: FData, y: object = None) -> NDArrayFloat:
        """
        Transform the provided data to its local averages.

        Args:
            X: FDataGrid with the samples that are going to be transformed.
            y: Unused.

        Returns:
            Array of shape (n_samples, n_intervals) including
            the transformed data.

        """
        return local_averages(
            X,
            domains=self.domains,
        ).reshape(X.data_matrix.shape[0], -1)


class OccupationMeasureTransformer(
    BaseEstimator,
    TransformerMixin[FData, NDArrayFloat, object],
):
    """
    Transformer that works as an adapter for the occupation_measure function.

    Args:
        intervals: ndarray of tuples containing the
            intervals we want to consider. The shape should be
            (n_sequences, 2)
        n_points: Number of points to evaluate in the domain.
            By default will be used the points defined on the FDataGrid.
            On a FDataBasis this value should be specified.

    Example:
        We will create the FDataGrid that we will use to extract
        the occupation measure
        >>> from skfda.representation import FDataGrid
        >>> import numpy as np
        >>> t = np.linspace(0, 10, 100)
        >>> fd_grid = FDataGrid(
        ...     data_matrix=[
        ...         t,
        ...         2 * t,
        ...         np.sin(t),
        ...     ],
        ...     grid_points=t,
        ... )

        Finally we call to the occupation measure function with the
        intervals that we want to consider. In our case (0.0, 1.0)
        and (2.0, 3.0). We need also to specify the number of points
        we want that the function takes into account to interpolate.
        We are going to use 501 points.
        >>> from skfda.preprocessing.feature_construction import (
        ...     OccupationMeasureTransformer,
        ... )
        >>> occupation_measure = OccupationMeasureTransformer(
        ...     intervals=[(0.0, 1.0), (2.0, 3.0)],
        ...     n_points=501,
        ... )

        >>> np.around(occupation_measure.fit_transform(fd_grid), decimals=2)
        array([[ 0.98,  1.  ],
               [ 0.5 ,  0.52],
               [ 6.28,  0.  ]])
    """

    def __init__(
        self,
        intervals: Sequence[Tuple[float, float]],
        *,
        n_points: Optional[int] = None,
    ):
        self.intervals = intervals
        self.n_points = n_points

    def transform(self, X: FData, y: object = None) -> NDArrayFloat:
        """
        Transform the provided data using the occupation_measure function.

        Args:
            X: FDataGrid or FDataBasis with the samples that are going to be
                transformed.

        Returns:
            Array of shape (n_intervals, n_samples) including the transformed
            data.
        """
        return occupation_measure(X, self.intervals, n_points=self.n_points)


class NumberCrossingsTransformer(
    BaseEstimator,
    TransformerMixin[FDataGrid, NDArrayInt, object],
):
    """
    Transformer that works as an adapter for the number_up_crossings function.

    Args:
        levels: Sequence of numbers including the levels
            we want to consider for the crossings. By
            default it calculates zero-crossings.
        direction: Whether to consider only up-crossings,
            down-crossings or both.

    Example:
        For this example we will use a well known function so the correct
        functioning of this method can be checked.
        We will create and use a DataFrame with a sample extracted from
        the Bessel Function of first type and order 0.
        First of all we import the Bessel Function and create the X axis
        data grid. Then we create the FdataGrid.
        >>> from skfda.preprocessing.feature_construction import (
        ...     NumberCrossingsTransformer,
        ... )
        >>> from scipy.special import jv
        >>> from skfda.representation import FDataGrid
        >>> import numpy as np
        >>> x_grid = np.linspace(0, 14, 14)
        >>> fd_grid = FDataGrid(
        ...     data_matrix=[jv([0], x_grid)],
        ...     grid_points=x_grid,
        ... )
        >>> fd_grid.data_matrix
        array([[[ 1.        ],
        [ 0.73041066],
        [ 0.13616752],
        [-0.32803875],
        [-0.35967936],
        [-0.04652559],
        [ 0.25396879],
        [ 0.26095573],
        [ 0.01042895],
        [-0.22089135],
        [-0.2074856 ],
        [ 0.0126612 ],
        [ 0.20089319],
        [ 0.17107348]]])

        Finally we evaluate the number of zero-upcrossings method with the
        FDataGrid created.
        >>> tf = NumberCrossingsTransformer(levels=0, direction="up")
        >>> tf.fit_transform(fd_grid)
        array([[2]])
    """

    def __init__(
        self,
        *,
        levels: ArrayLike = 0,
        direction: Literal["up", "down", "all"] = "all",
    ):
        self.levels = levels
        self.direction = direction

    def transform(self, X: FDataGrid, y: object = None) -> NDArrayInt:
        """
        Transform the provided data using the number_up_crossings function.

        Args:
            X: FDataGrid with the samples that are going to be transformed.

        Returns:
            Array of shape (n_samples, len(levels)) including the transformed
            data.
        """
        return number_crossings(
            X,
            levels=self.levels,
            direction=self.direction,
        )
