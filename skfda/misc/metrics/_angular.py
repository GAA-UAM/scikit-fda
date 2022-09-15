from __future__ import annotations

from typing import Optional, TypeVar, Union

import numpy as np
from typing_extensions import Final

from ...representation import FData
from ...typing._numpy import NDArrayFloat
from .._math import cosine_similarity, cosine_similarity_matrix
from ._utils import pairwise_metric_optimization

T = TypeVar("T", bound=Union[NDArrayFloat, FData])


class AngularDistance():
    r"""
    Calculate the angular distance between two objects.

    For each pair of observations x and y the angular distance between them is
    defined as the normalized "angle" between them:

    .. math::
        d(x, y) = \frac{\arccos \left(\frac{\langle x, y \rangle}{
        \sqrt{\langle x, x \rangle \langle y, y \rangle}} \right)}{\pi}

    where :math:`\langle {}\cdot{}, {}\cdot{} \rangle` is the inner product.
    This distance is defined in the interval [0, 1].

    Args:
        e1: First object.
        e2: Second object.

    Returns:
        Numpy vector where the i-th coordinate has the angular distance between
        the i-th element of the first object and the i-th element of the second
        one.

    Examples:
        Computes the angular distances between an object containing functional
        data corresponding to the functions y = 1 and y = x defined over the
        interval [0, 1] and another ones containing data of the functions y
        = 0 and y = x/2. The result then is an array of size 2 with the
        computed l2 distance between the functions in the same position in
        both.

        >>> import skfda
        >>> import numpy as np
        >>>
        >>> x = np.linspace(0, 1, 1001)
        >>> fd = skfda.FDataGrid([np.ones(len(x)), x], x)
        >>> fd2 =  skfda.FDataGrid([2*np.ones(len(x)), np.cos(x)], x)
        >>>
        >>> skfda.misc.metrics.angular_distance(fd, fd2).round(2)
        array([ 0.  ,  0.22])

    """

    def __call__(
        self,
        e1: T,
        e2: T,
    ) -> NDArrayFloat:
        """Compute the distance."""
        return np.arccos(cosine_similarity(e1, e2)) / np.pi

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}()"
        )


angular_distance: Final = AngularDistance()


@pairwise_metric_optimization.register
def _pairwise_metric_optimization_angular(
    metric: AngularDistance,
    elem1: Union[NDArrayFloat, FData],
    elem2: Optional[Union[NDArrayFloat, FData]],
) -> NDArrayFloat:

    return np.arccos(cosine_similarity_matrix(elem1, elem2)) / np.pi
