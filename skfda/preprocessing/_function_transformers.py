from typing import Optional, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import FunctionTransformer

from ..exploratory.stats._functional_transformers import (
    local_averages,
    number_up_crossings,
    occupation_measure,
)
from ..representation._typing import NDArrayFloat, NDArrayInt


class LocalAveragesTransformer(FunctionTransformer):
    """
    Transformer that works as an adapter for the local_averages function.

    Args:
        n_intervals: number of intervals we want to consider.

    Example:
        We import the Berkeley Growth Study dataset.
        We will use only the first 30 samples to make the
        example easy.
        >>> from skfda.datasets import fetch_growth
        >>> dataset = fetch_growth(return_X_y=True)[0]
        >>> X = dataset[:30]

        Then we decide how many intervals we want to consider (in our case 2)
        and call the function with the dataset.
        >>> import numpy as np
        >>> from skfda.preprocessing import LocalAveragesTransformer
        >>> local_averages = LocalAveragesTransformer(2)
        >>> np.around(local_averages.fit_transform(X), decimals=2)
        array([[[ 116.94],
                [ 111.86],
                [ 107.29],
                [ 111.35],
                [ 104.39],
                [ 109.43],
                [ 109.16],
                [ 112.91],
                [ 109.19],
                [ 117.95],
                [ 112.14],
                [ 114.3 ],
                [ 111.48],
                [ 114.85],
                [ 116.25],
                [ 114.6 ],
                [ 111.02],
                [ 113.57],
                [ 108.88],
                [ 109.6 ],
                [ 109.7 ],
                [ 108.54],
                [ 109.18],
                [ 106.92],
                [ 109.44],
                [ 109.84],
                [ 115.32],
                [ 108.16],
                [ 119.29],
                [ 110.62]],
               [[ 177.26],
                [ 157.62],
                [ 154.97],
                [ 163.83],
                [ 156.66],
                [ 157.67],
                [ 155.31],
                [ 169.02],
                [ 154.18],
                [ 174.43],
                [ 161.33],
                [ 170.14],
                [ 164.1 ],
                [ 170.1 ],
                [ 166.65],
                [ 168.72],
                [ 166.85],
                [ 167.22],
                [ 159.4 ],
                [ 162.76],
                [ 155.7 ],
                [ 158.01],
                [ 160.1 ],
                [ 155.95],
                [ 157.95],
                [ 163.53],
                [ 162.29],
                [ 153.1 ],
                [ 178.48],
                [ 161.75]]])

    """

    def __init__(self, n_intervals: int):
        self.n_intervals = n_intervals
        super().__init__()

    def transform(self, X) -> np.ndarray:
        """
        Compute the local averages of a given data.

        Args:
            X: FDataGrid or FDataBasis where we want to
                calculate the local averages.

        Returns:
            ndarray of shape (n_intervals, n_samples, n_dimensions)
            with the transformed data for FDataBasis and
            (n_intervals, n_samples) for FDataGrid.
        """
        return local_averages(X, self.n_intervals)


class OccupationMeasureTransformer(FunctionTransformer):
    """
    Transformer that works as an adapter for the occupation_measure function.

    Args:
        intervals: ndarray of tuples containing the
            intervals we want to consider. The shape should be
            (n_sequences,2)
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
        >>> from skfda.preprocessing import OccupationMeasureTransformer
        >>> occupation_measure = OccupationMeasureTransformer(
        ...     intervals=[(0.0, 1.0), (2.0, 3.0)],
        ...     n_points=501,
        ... )

        >>> np.around(occupation_measure.fit_transform(fd_grid), decimals=2)
        array([[ 0.98,  0.5 ,  6.28],
               [ 1.02,  0.52,  0.  ]])
    """

    def __init__(
        self,
        intervals: Sequence[Tuple[float, float]],
        *,
        n_points: Optional[int] = None,
    ):
        self.intervals = intervals
        self.n_points = n_points
        super().__init__()

    def transform(self, X) -> NDArrayFloat:
        """
        Compute the occupation measure of a grid.

        Args:
            X: FDataGrid or FDataBasis where we want to calculate
                the occupation measure.

        Returns:
            ndarray of shape (n_intervals, n_samples)
            with the transformed data.
        """
        return occupation_measure(X, self.intervals, n_points=self.n_points)


class NumberUpCrossingsTransformer(FunctionTransformer):
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
    >>> from skfda.preprocessing import NumberUpCrossingsTransformer
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
        super().__init__()

    def transform(self, X) -> NDArrayInt:
        """
        Compute the number of up crossings of a given data.

        Args:
            X: FDataGrid where we want to calculate
                the number of up crossings.

        Returns:
            ndarray of shape (n_samples, len(levels))\
            with the values of the counters.
        """
        return number_up_crossings(X, self.levels)
