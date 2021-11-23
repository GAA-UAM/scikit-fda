"""Functional Mahalanobis Distance Module."""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator

from ...representation import FData
from ...representation._typing import ArrayLike, NDArrayFloat
from ...representation.basis import Basis
from .._math import inner_product
from ..regularization._regularization import TikhonovRegularization

WeightsCallable = Callable[[np.ndarray], np.ndarray]


class FMahalanobisDistance(BaseEstimator):  # type: ignore
    r"""Functional Mahalanobis distance.

    Class that implements functional Mahalanobis distance for both
    basis and grid representations of the data
    :footcite:`berrendero+bueno-larraz+cuevas_2020_mahalanobis`.

    Parameters:
        n_components: Number of eigenfunctions to keep from
            functional principal component analysis. Defaults to 10.
        centering: If ``True`` then calculate the mean of the functional
            data object and center the data first. Defaults to ``True``.
        regularization: Regularization object to be applied.
        components_basis: The basis in which we want the eigenfunctions.
            We can use a different basis than the basis contained in the
            passed FDataBasis object. This parameter is only used when
            fitting a FDataBasis.
        weights: the weights vector used for discrete integration.
            If none then the trapezoidal rule is used for computing the
            weights. If a callable object is passed, then the weight
            vector will be obtained by evaluating the object at the sample
            points of the passed FDataGrid object in the fit method. This
            parameter is only used when fitting a FDataGrid.

    Attributes:
        ef\_: Eigenfunctions of the covariance operator.
        ev\_: Eigenvalues of the covariance operator.
        mean\_: Mean of the stochastic process.

    Examples:
        >>> from skfda.misc.metrics import FMahalanobisDistance
        >>> import numpy as np
        >>> from skfda.representation.grid import FDataGrid
        >>> data_matrix = np.array([[1.0, 0.0], [0.0, 2.0]])
        >>> grid_points = [0, 1]
        >>> fd = FDataGrid(data_matrix, grid_points)
        >>> fmah = FMahalanobisDistance(2)
        >>> fmah.fit(fd)
        FMahalanobisDistance(n_components=2)
        >>> fmah.ef_
        FDataGrid(
            array([[[-0.63245553],
                    [ 1.26491106]],
                   [[ 1.26491106],
                    [ 0.63245553]]]),
            grid_points=(array([ 0., 1.]),),
            domain_range=((0.0, 1.0),),
            dataset_name=None,
            argument_names=(None,),
            coordinate_names=(None,),
            extrapolation=None,
            interpolation=SplineInterpolation(interpolation_order=1,
                smoothness_parameter=0, monotone=False))
        >>> fmah.ev_
        array([ 1.25000000e+00, 1.95219693e-33])
        >>> fmah.mean_
        FDataGrid(
            array([[[ 0.5], [ 1. ]]]),
        grid_points=(array([ 0.,  1.]),),
        domain_range=((0.0, 1.0),),
        dataset_name=None,
        argument_names=(None,),
        coordinate_names=(None,),
        extrapolation=None,
        interpolation=SplineInterpolation(interpolation_order=1,
            smoothness_parameter=0, monotone=False))

    References:
        .. footbibliography::
    """

    def __init__(
        self,
        n_components: int = 10,
        centering: bool = True,
        regularization: Optional[TikhonovRegularization[FData]] = None,
        weights: Optional[Union[ArrayLike, WeightsCallable]] = None,
        components_basis: Optional[Basis] = None,
        alpha: float = 0.001,
        k: int = 10,
    ) -> None:
        self.n_components = n_components
        self.centering = centering
        self.regularization = regularization
        self.weights = weights
        self.components_basis = components_basis
        self.alpha = alpha
        self.k = k

    def fit(
        self,
        X: FData,
        y: None = None,
    ) -> FMahalanobisDistance:
        """Fit the functional Mahalanobis distance to X.

        We extract the eigenfunctions and corresponding eigenvalues
        of the covariance operator by using FPCA.

        Args:
            X: Stochastic process.
            y: Ignored.

        Returns:
            self
        """
        from ...preprocessing.dim_reduction.feature_extraction import FPCA

        fpca = FPCA(
            self.n_components,
            self.centering,
            self.regularization,
            self.weights,
            self.components_basis,
        )
        fpca.fit(X)
        self.ev_ = fpca.explained_variance_
        self.ef_ = fpca.components_
        self.mean_ = fpca.mean_

        return self

    def mahalanobis_distance(
        self,
        e1: FData,
        e2: FData,
    ) -> NDArrayFloat:
        """Compute the squared functional Mahalanobis distances of given observations.

        Args:
            e1: First object.
            e2: Second object.

        Returns:
            Squared functional Mahalanobis distances of the observations.
        """
        return np.sum(
            self.ev_ * inner_product(e1 - e2, self.ef_) ** 2
            / (self.ev_ + self.alpha)**2,
        )

    def mahalanobis_depth(self, e1: FData):
        """Compute the Mahalanobis depth of a given observations.

        Args:
            e1: First object.

        Returns:
            Depth of the observations.
        """
        return 1 / (1 + self.mahalanobis(e1, self.mean_))
