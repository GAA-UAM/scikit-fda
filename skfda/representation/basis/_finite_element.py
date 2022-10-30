import warnings
from typing import Optional

from ...typing._base import DomainRangeLike
from ...typing._numpy import ArrayLike
from ._finite_element_basis import FiniteElementBasis


class FiniteElement(FiniteElementBasis):
    """Finite element basis.

    Given a n-dimensional grid made of simplices, each element of the basis
    is a piecewise linear function that takes the value 1 at exactly one
    vertex and 0 in the other vertices.

    .. deprecated:: 0.8
        Use :class:`~skfda.representation.basis.FiniteElementBasis` instead.

    Parameters:
        vertices: The vertices of the grid.
        cells: A list of individual cells, consisting in the indexes of
            :math:`n+1` vertices for an n-dimensional domain space.

    Examples:
        >>> from skfda.representation.basis import FiniteElementBasis
        >>> basis = FiniteElementBasis(
        ...     vertices=[[0, 0], [0, 1], [1, 0], [1, 1]],
        ...     cells=[[0, 1, 2], [1, 2, 3]],
        ...     domain_range=[(0, 1), (0, 1)],
        ... )

        Evaluates all the functions in the basis in a list of discrete
        values.

        >>> basis([[0.1, 0.1], [0.6, 0.6], [0.1, 0.2], [0.8, 0.9]])
        array([[[ 0.8],
                [ 0. ],
                [ 0.7],
                [ 0. ]],
               [[ 0.1],
                [ 0.4],
                [ 0.2],
                [ 0.2]],
               [[ 0.1],
                [ 0.4],
                [ 0.1],
                [ 0.1]],
               [[ 0. ],
                [ 0.2],
                [ 0. ],
                [ 0.7]]])


        >>> from scipy.spatial import Delaunay
        >>> import numpy as np
        >>>
        >>> n_points = 10
        >>> points = np.random.uniform(size=(n_points, 2))
        >>> delaunay = Delaunay(points)
        >>> basis = FiniteElementBasis(
        ...     vertices=delaunay.points,
        ...     cells=delaunay.simplices,
        ... )
        >>> basis.n_basis
        10

     """

    def __init__(
        self,
        vertices: ArrayLike,
        cells: ArrayLike,
        domain_range: Optional[DomainRangeLike] = None,
    ) -> None:
        super().__init__(
            vertices=vertices,
            cells=cells,
            domain_range=domain_range,
        )
        warnings.warn(
            "The FiniteElement class is deprecated. Use "
            "FiniteElementBasis instead.",
            DeprecationWarning,
        )
