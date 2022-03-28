"""Functional Mahalanobis Distance Module."""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from ...representation import FData
from ...representation._typing import ArrayLike, NDArrayFloat
from ...representation.basis import Basis
from .._math import inner_product
from ..regularization._regularization import TikhonovRegularization

WeightsCallable = Callable[[np.ndarray], np.ndarray]


class MahalanobisDistance(BaseEstimator):  # type: ignore
    r"""Functional Mahalanobis distance.

    Class that implements functional Mahalanobis distance for both
    basis and grid representations of the data
    :footcite:`berrendero+bueno-larraz+cuevas_2020_mahalanobis`.

    Parameters:
        n_components: Number of eigenvectors to keep from
            functional principal component analysis. Defaults to 10.
        centering: If ``True`` then calculate the mean of the functional
            data object and center the data first. Defaults to ``True``.
        regularization: Regularization object to be applied.
        components_basis: The basis in which we want the eigenvectors.
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
        eigen_vectors\_: Eigenvectors of the covariance operator.
        eigen_values\_: Eigenvalues of the covariance operator.
        mean\_: Mean of the stochastic process.

    Examples:
        >>> from skfda.misc.metrics import MahalanobisDistance
        >>> import numpy as np
        >>> from skfda.representation.grid import FDataGrid
        >>> data_matrix = np.array([[1.0, 0.0], [0.0, 2.0]])
        >>> grid_points = [0, 1]
        >>> fd = FDataGrid(data_matrix, grid_points)
        >>> mahalanobis = MahalanobisDistance(2)
        >>> mahalanobis.fit(fd)
        MahalanobisDistance(n_components=2)
        >>> mahalanobis(fd[0], fd[1])
        1.9968038359080937

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
    ) -> None:
        self.n_components = n_components
        self.centering = centering
        self.regularization = regularization
        self.weights = weights
        self.components_basis = components_basis
        self.alpha = alpha

    def fit(
        self,
        X: FData,
        y: None = None,
    ) -> MahalanobisDistance:
        """Fit the functional Mahalanobis distance to X.

        We extract the eigenvectors and corresponding eigenvalues
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
        self.eigen_values_ = fpca.explained_variance_
        self.eigen_vectors_ = fpca.components_
        self.mean_ = fpca.mean_

        return self

    def __call__(
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
        check_is_fitted(self)

        return np.sum(
            self.eigen_values_
            * inner_product(e1 - e2, self.eigen_vectors_) ** 2
            / (self.eigen_values_ + self.alpha)**2,
        )
