from __future__ import annotations

from typing import Optional, Union, overload

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from typing_extensions import Literal

from ._functional_data import FData
from ._typing import ArrayLike, GridPointsLike
from .extrapolation import ExtrapolationLike
from .grid import FDataGrid


class EvaluationTransformer(
    BaseEstimator,  # type:ignore
    TransformerMixin,  # type:ignore
):
    r"""
    Transformer returning the evaluations of FData objects as a matrix.

    Args:
        eval_points (array_like): List of points where the functions are
            evaluated. If `None`, the functions must be `FDatagrid` objects
            and all points will be returned.
        extrapolation (str or Extrapolation, optional): Controls the
            extrapolation mode for elements outside the :term:`domain` range.
            By default it is used the mode defined during the instance of the
            object.
        grid (bool, optional): Whether to evaluate the results on a grid
            spanned by the input arrays, or at points specified by the
            input arrays. If true the eval_points should be a list of size
            dim_domain with the corresponding times for each axis. The
            return matrix has shape n_samples x len(t1) x len(t2) x ... x
            len(t_dim_domain) x dim_codomain. If the domain dimension is 1
            the parameter has no efect. Defaults to False.

    Attributes:
        shape\_ (tuple): original shape of coefficients per sample.

    Examples:
        >>> from skfda.representation import (FDataGrid, FDataBasis,
        ...                                   EvaluationTransformer)
         >>> from skfda.representation.basis import Monomial

        Functional data object with 2 samples
        representing a function :math:`f : \mathbb{R}\longmapsto\mathbb{R}`.

        >>> data_matrix = [[1, 2], [2, 3]]
        >>> grid_points = [2, 4]
        >>> fd = FDataGrid(data_matrix, grid_points)
        >>>
        >>> transformer = EvaluationTransformer()
        >>> transformer.fit_transform(fd)
        array([[ 1., 2.],
               [ 2., 3.]])

        Functional data object with 2 samples
        representing a function :math:`f : \mathbb{R}\longmapsto\mathbb{R}^2`.

        >>> data_matrix = [[[1, 0.3], [2, 0.4]], [[2, 0.5], [3, 0.6]]]
        >>> grid_points = [2, 4]
        >>> fd = FDataGrid(data_matrix, grid_points)
        >>>
        >>> transformer = EvaluationTransformer()
        >>> transformer.fit_transform(fd)
        array([[ 1. ,  0.3,  2. ,  0.4],
               [ 2. ,  0.5,  3. ,  0.6]])

        Representation of a functional data object with 2 samples
        representing a function :math:`f : \mathbb{R}^2\longmapsto\mathbb{R}`.

        >>> data_matrix = [[[1, 0.3], [2, 0.4]], [[2, 0.5], [3, 0.6]]]
        >>> grid_points = [[2, 4], [3, 6]]
        >>> fd = FDataGrid(data_matrix, grid_points)
        >>>
        >>> transformer = EvaluationTransformer()
        >>> transformer.fit_transform(fd)
        array([[ 1. ,  0.3,  2. ,  0.4],
               [ 2. ,  0.5,  3. ,  0.6]])

        Evaluation of a functional data object at several points.

        >>> basis = Monomial(n_basis=4)
        >>> coefficients = [[0.5, 1, 2, .5], [1.5, 1, 4, .5]]
        >>> fd = FDataBasis(basis, coefficients)
        >>>
        >>> transformer = EvaluationTransformer([0, 0.2, 0.5, 0.7, 1])
        >>> transformer.fit_transform(fd)
        array([[ 0.5   ,  0.784 ,  1.5625,  2.3515,  4.    ],
               [ 1.5   ,  1.864 ,  3.0625,  4.3315,  7.    ]])

    """

    @overload
    def __init__(
        self,
        eval_points: ArrayLike,
        *,
        extrapolation: Optional[ExtrapolationLike] = None,
        grid: Literal[False] = False,
    ) -> None:
        pass

    @overload
    def __init__(
        self,
        eval_points: GridPointsLike,
        *,
        extrapolation: Optional[ExtrapolationLike] = None,
        grid: Literal[True],
    ) -> None:
        pass

    def __init__(
        self,
        eval_points: Union[ArrayLike, GridPointsLike, None] = None,
        *,
        extrapolation: Optional[ExtrapolationLike] = None,
        grid: bool = False,
    ):
        self.eval_points = eval_points
        self.extrapolation = extrapolation
        self.grid = grid

    def fit(  # noqa: D102
        self,
        X: FData,
        y: None = None,
    ) -> EvaluationTransformer:

        if self.eval_points is None and not isinstance(X, FDataGrid):
            raise ValueError(
                "If no eval_points are passed, the functions "
                "should be FDataGrid objects.",
            )

        self._is_fitted = True

        return self

    def transform(  # noqa: D102
        self,
        X: FData,
        y: None = None,
    ) -> np.ndarray:

        check_is_fitted(self, '_is_fitted')

        if self.eval_points is None:
            evaluation = X.data_matrix.copy()
        else:
            evaluation = X(  # type: ignore
                self.eval_points,
                extrapolation=self.extrapolation,
                grid=self.grid,
            )

        return evaluation.reshape((X.n_samples, -1))
