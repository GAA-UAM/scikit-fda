from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.integrate

from ..._utils._sklearn_adapter import BaseEstimator, InductiveTransformerMixin
from ...representation import FDataGrid
from ...representation.basis import Basis
from ...typing._numpy import ArrayLike
from ..validation import check_fdata_dimensions
from ._operators import Operator


class SRSF(
    Operator[FDataGrid, FDataGrid],
    BaseEstimator,
    InductiveTransformerMixin[FDataGrid, FDataGrid, object],
):
    r"""Square-Root Slope Function (SRSF) transform.

    Let :math:`f : [a,b] \rightarrow \mathbb{R}` be an absolutely continuous
    function, the SRSF transform is defined as

    .. math::
        SRSF(f(t)) = sgn(f(t)) \sqrt{|\dot f(t)|} = q(t)

    This representation it is used to compute the extended non-parametric
    Fisher-Rao distance between functions, wich under the SRSF representation
    becomes the usual :math:`\mathbb{L}^2` distance between functions.
    See :footcite:`srivastava+klassen_2016_analysis_square`.

    The inverse SRSF transform is defined as

    .. math::
        f(t) = f(a) + \int_{a}^t q(t)|q(t)|dt .

    This transformation is a mapping up to constant. Given the SRSF and the
    initial value :math:`f(a)` the original function can be obtained, for this
    reason it is necessary to store the value :math:`f(a)` during the fit,
    which is dropped due to derivation. If it is applied the inverse
    transformation without fit the estimator it is assumed that :math:`f(a)=0`.

    Args:
        eval_points: (array_like, optional): Set of points where the
            functions are evaluated, by default uses the sample points of
            the :class:`FDataGrid <skfda.FDataGrid>` transformed.
        initial_value (float, optional): Initial value to apply in the
            inverse transformation. If `None` there are stored the initial
            values of the functions during the transformation to apply
            during the inverse transformation. Defaults None.
        method: Method to use to compute the derivative. If ``None``
            (the default), finite differences are used. In a basis
            object is passed the grid is converted to a basis
            representation and the derivative is evaluated using that
            representation.

    Attributes:
        eval_points: Set of points where the
            functions are evaluated, by default uses the grid points of the
            fdatagrid.
        initial_value: Initial value to apply in the
            inverse transformation. If `None` there are stored the initial
            values of the functions during the transformation to apply
            during the inverse transformation. Defaults None.

    Note:
        Due to the use of derivatives it is recommended that the samples are
        sufficiently smooth, or have passed a smoothing preprocessing before,
        in order to achieve good results.

    References:
        .. footbibliography::

    Examples:
        Create a toy dataset and apply the transformation and its inverse.

        >>> from skfda.datasets import make_sinusoidal_process
        >>> from skfda.misc.operators import SRSF
        >>> fd = make_sinusoidal_process(error_std=0, random_state=0)
        >>> srsf = SRSF()
        >>> srsf
        SRSF(...)

        Fits the estimator (to apply the inverse transform) and apply the SRSF

        >>> q = srsf.fit_transform(fd)

        Apply the inverse transform.

        >>> fd_pull_back = srsf.inverse_transform(q)

        The original and the pull back `fd` are almost equal

        >>> zero = fd - fd_pull_back
        >>> zero.data_matrix.flatten().round(3)
        array([ 0.,  0.,  0., ..., -0., -0., -0.])

    """

    def __init__(
        self,
        *,
        output_points: Optional[ArrayLike] = None,
        initial_value: Optional[float] = None,
        method: Optional[Basis] = None,
    ) -> None:
        self.output_points = output_points
        self.initial_value = initial_value
        self.method = method

    def __call__(self, vector: FDataGrid) -> FDataGrid:  # noqa: D102
        return self.fit_transform(vector)

    def fit(self, X: FDataGrid, y: object = None) -> SRSF:
        """
        Return self. This transformer does not need to be fitted.

        Args:
            X: Present for API conventions.
            y: Present for API conventions.

        Returns:
            (Estimator): self

        """
        return self

    def transform(self, X: FDataGrid, y: object = None) -> FDataGrid:
        r"""
        Compute the square-root slope function (SRSF) transform.

        Let :math:`f : [a,b] \rightarrow \mathbb{R}` be an absolutely
        continuous function, the SRSF transform is defined as
        :footcite:`srivastava+klassen_2016_analysis_square`:

        .. math::

            SRSF(f(t)) = sgn(f(t)) \sqrt{\dot f(t)|} = q(t)

        Args:
            X: Functions to be transformed.
            y: Present for API conventions.

        Returns:
            SRSF functions.

        Raises:
            ValueError: If functions are not univariate.

        """
        check_fdata_dimensions(
            X,
            dim_domain=1,
            dim_codomain=1,
        )

        if self.output_points is None:
            output_points = X.grid_points[0]
        else:
            output_points = np.asarray(self.output_points)

        g = X.derivative(method=self.method)

        # Evaluation with the corresponding interpolation
        data_matrix = g(output_points)[..., 0]

        # SRSF(f) = sign(f) * sqrt|Df| (avoiding multiple allocation)
        sign_g = np.sign(data_matrix)
        data_matrix = np.abs(data_matrix, out=data_matrix)
        data_matrix = np.sqrt(data_matrix, out=data_matrix)
        data_matrix *= sign_g

        # Store the values of the transformation
        if self.initial_value is None:
            a = X.domain_range[0][0]
            self.initial_value_ = X(a).reshape(X.n_samples, 1, X.dim_codomain)

        return X.copy(data_matrix=data_matrix, grid_points=output_points)

    def inverse_transform(self, X: FDataGrid, y: None = None) -> FDataGrid:
        r"""
        Compute the inverse SRSF transform.

        Given the srsf and the initial value the original function can be
        obtained as :footcite:`srivastava+klassen_2016_analysis_square`:

        .. math::
            f(t) = f(a) + \int_{a}^t q(t)|q(t)|dt

        where :math:`q(t)=SRSF(f(t))`.

        If it is applied this inverse transformation without fitting the
        estimator it is assumed that :math:`f(a)=0`.

        Args:
            X: SRSF to be transformed.
            y: Present for API conventions.

        Returns:
            Functions in the original space.

        Raises:
            ValueError: If functions are multidimensional.
        """
        check_fdata_dimensions(
            X,
            dim_domain=1,
            dim_codomain=1,
        )

        stored_initial_value = getattr(self, 'initial_value_', None)

        if self.initial_value is None and stored_initial_value is None:
            raise AttributeError(
                "When initial_value=None is expected a "
                "previous transformation of the data to "
                "store the initial values to apply in the "
                "inverse transformation. Also it is possible "
                "to fix these values setting the attribute"
                "initial value without a previous "
                "transformation.",
            )

        if self.output_points is None:
            output_points = X.grid_points[0]
        else:
            output_points = np.asarray(self.output_points)

        data_matrix = X(output_points)

        data_matrix *= np.abs(data_matrix)

        f_data_matrix = scipy.integrate.cumtrapz(
            data_matrix,
            x=output_points,
            axis=1,
            initial=0,
        )

        # If the transformer was fitted, sum the initial value
        if self.initial_value is None:
            f_data_matrix += self.initial_value_
        else:
            f_data_matrix += self.initial_value

        return X.copy(data_matrix=f_data_matrix, grid_points=output_points)
