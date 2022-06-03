"""Function transformers for feature construction techniques."""
from typing import Optional, Sequence, Tuple

from sklearn.base import BaseEstimator

from ..._utils import TransformerMixin
from ...exploratory.stats._functional_transformers import (
    local_averages,
    number_up_crossings,
    occupation_measure,
)
from ...representation._typing import NDArrayFloat, Union
from ...representation.basis import FDataBasis
from ...representation.grid import FDataGrid


class LocalAveragesTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that works as an adapter for the local_averages function.

    Args:
        n_intervals: number of intervals we want to consider.

    Example:
        We import the Berkeley Growth Study dataset.
        We will use only the first 3 samples to make the
        example easy.
        >>> from skfda.datasets import fetch_growth
        >>> dataset = fetch_growth(return_X_y=True)[0]
        >>> X = dataset[:3]

        Then we decide how many intervals we want to consider (in our case 2)
        and call the function with the dataset.
        >>> import numpy as np
        >>> from skfda.preprocessing.feature_construction import (
        ...     LocalAveragesTransformer,
        ... )
        >>> local_averages = LocalAveragesTransformer(2)
        >>> np.around(local_averages.fit_transform(X), decimals=2)
        array([[ 116.94,  177.26],
               [ 111.86,  157.62],
               [ 107.29,  154.97]])
    """

    def __init__(self, n_intervals: int):
        self.n_intervals = n_intervals

    def transform(self, X: Union[FDataGrid, FDataBasis]) -> NDArrayFloat:
        """
        Transform the provided data using the local_averages function.

        Args:
            X: FDataGrid with the samples that are going to be transformed.

        Returns:
            Array of shape (n_samples, n_intervals) including
            the transformed data.
        """
        return local_averages(
            X,
            self.n_intervals,
        ).reshape(X.data_matrix.shape[0], -1)


class OccupationMeasureTransformer(BaseEstimator, TransformerMixin):
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
        array([[ 0.98,  1.02],
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

    def transform(self, X: Union[FDataGrid, FDataBasis]) -> NDArrayFloat:
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


class NumberUpCrossingsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that works as an adapter for the number_up_crossings function.

    Args:
        levels: sequence of numbers including the levels
            we want to consider for the crossings.
    Example:
    For this example we will use a well known function so the correct
    functioning of this method can be checked.
    We will create and use a DataFrame with a sample extracted from
    the Bessel Function of first type and order 0.
    First of all we import the Bessel Function and create the X axis
    data grid. Then we create the FdataGrid.
    >>> from skfda.preprocessing.feature_construction import (
    ...     NumberUpCrossingsTransformer,
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

    Finally we evaluate the number of up crossings method with the FDataGrid
    created.
    >>> NumberUpCrossingsTransformer(np.asarray([0])).fit_transform(fd_grid)
    array([[2]])
    """

    def __init__(self, levels: NDArrayFloat):
        self.levels = levels

    def transform(self, X: FDataGrid) -> NDArrayFloat:
        """
        Transform the provided data using the number_up_crossings function.

        Args:
            X: FDataGrid with the samples that are going to be transformed.

        Returns:
            Array of shape (n_samples, len(levels)) including the transformed
            data.
        """
        return number_up_crossings(X, self.levels)
