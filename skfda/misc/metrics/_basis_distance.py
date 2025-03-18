from __future__ import annotations

from typing import Optional, Union

import numpy as np
from ...typing._numpy import NDArrayFloat
from ...representation import FData, FDataBasis, FDataGrid
from ..._utils._sklearn_adapter import BaseEstimator


from typing import Optional, Union
import numpy as np
from ...representation.grid import FDataGrid
from ...preprocessing.dim_reduction import FPCA
from ...representation.basis import Basis
from ...misc.regularization import TikhonovRegularization


class MultivariateMahalanobisDistance(BaseEstimator):
    def __init__(
        self,
        n_components: int = 10,
        centering: bool = True,
        regularization: Optional[TikhonovRegularization[FDataGrid]] = None,
        weights: Optional[np.ndarray] = None,
        components_basis: Optional[Basis] = None,
        alpha: float = 0.001,
        eigenvalues: Optional[np.ndarray] = None,
        eigenvectors: Optional[FDataGrid] = None,
        p: float = 1.0,
    ) -> None:
        self.n_components = n_components
        self.centering = centering
        self.regularization = regularization
        self.weights = weights
        self.components_basis = components_basis
        self.alpha = alpha
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.p = p

    def fit(self, X: FDataGrid, y: None = None) -> "MultivariateMahalanobisDistance":
        if self.eigenvalues is None or self.eigenvectors is None:
            fpca = FPCA(
                n_components=self.n_components,
                centering=self.centering,
                regularization=self.regularization,
                components_basis=self.components_basis,
                _weights=self.weights,
            )
            fpca.fit(X)
            self.eigenvalues_ = fpca.explained_variance_
            self.eigenvectors_ = fpca.components_
        else:
            self.eigenvalues_ = self.eigenvalues
            self.eigenvectors_ = self.eigenvectors
        return self

    def _compute_h_l(self, lambda_l: float) -> float:
        """Compute the h_l(p) function as defined in the paper."""
        return lambda_l / (lambda_l + 1 / self.p)

    def __call__(self, e1: FDataGrid, e2: FDataGrid) -> np.ndarray:
        if not hasattr(self, "eigenvalues_") or not hasattr(self, "eigenvectors_"):
            raise ValueError("The model has not been fitted yet.")

        distances = []
        for l, (eigenvalue, eigenvector) in enumerate(
            zip(self.eigenvalues_, self.eigenvectors_)
        ):
            #d_M_l = (inner_product / eigenvalue) ** 2
            h_l = self._compute_h_l(eigenvalue)
            #distances.append(d_M_l * h_l)

        return np.sqrt(np.sum(distances, axis=0))


class BasisBasedDistance:
    r"""
    Class for computing the weighted distance between two functional observations 
    represented as FDataBasis objects in the same basis.

    Given two functional observations \(X_1, X_2\), their basis representations are:

    .. math::
        X_1(t) = \sum_{k=1}^{K} c_{1,k} \phi_k(t), \quad 
        X_2(t) = \sum_{k=1}^{K} c_{2,k} \phi_k(t).

    A natural distance between \(X_1\) and \(X_2\) is defined in terms of the 
    basis coefficients \(c_{1,k}\) and \(c_{2,k}\):

    .. math::
        d_{\nu}(\mathbf{X}_1, \mathbf{X}_2) = \sqrt{\sum_{k=1}^{K} \nu_k \left(c_{1,k} - c_{2,k}\right)^2 }.

    Where:
        - \( c_{n,k} = \langle X_n, \phi_k \rangle_\mu \) are the basis coefficients.
        - \( \nu_k \) is a weighting function controlling the contribution of each basis function.

    If the basis \( \{\phi_k\}_{k=1}^K \) is not orthonormal, the distance is generalized as:

    .. math::
        d_{\nu}(\mathbf{X}_1, \mathbf{X}_2) = 
        \sqrt{\sum_{i=1}^{K} \nu_i \left( \sum_{j=1}^K (c_{1,j} - c_{2,j}) \langle \phi_j, \phi_i \rangle \right)^2 }.

    This can be compactly written as:

    .. math::
        d_{\nu}^2(\mathbf{X}_1, \mathbf{X}_2) = 
        \langle \mathbf{\nu}, \left( (\mathbf{c_1 - c_2})^T M \right)^2 \rangle,

    where \( M \) is the Gram matrix of inner products between the basis functions:

    .. math::
        M_{i,j} = \langle \phi_i, \phi_j \rangle.

    The Gram matrix \( M \) has the following form:

    .. math::
        M =
        \begin{pmatrix}
        \langle \phi_1, \phi_1 \rangle & \langle \phi_1, \phi_2 \rangle & \cdots & \langle \phi_1, \phi_K \rangle \\
        \langle \phi_2, \phi_1 \rangle & \langle \phi_2, \phi_2 \rangle & \cdots & \langle \phi_2, \phi_K \rangle \\
        \vdots & \vdots & \ddots & \vdots \\
        \langle \phi_K, \phi_1 \rangle & \langle \phi_K, \phi_2 \rangle & \cdots & \langle \phi_K, \phi_K \rangle
        \end{pmatrix}.

    This class takes two `FDataBasis` objects with the same basis and computes 
    the functional distance between them.

    Attributes:
        fd1 (FDataBasis): First functional data object.
        fd2 (FDataBasis): Second functional data object.
        weights (ndarray): Weighting function \( \nu_k \).
        gram_matrix (ndarray): Gram matrix \( M \) of the basis functions.

    Methods:
        compute_distance(): Computes the functional distance \( d_{\nu}(X_1, X_2) \).

    Example:
        >>> from skfda.representation.basis import FDataBasis, BSplineBasis
        >>> import numpy as np
        >>> basis = BSplineBasis(n_basis=5)
        >>> fd1 = FDataBasis(basis, np.array([1, 2, 3, 4, 5]))
        >>> fd2 = FDataBasis(basis, np.array([2, 3, 1, 5, 4]))
        >>> weights = np.ones(5)
        >>> dist = BasisBasedDistance(fd1, fd2, weights)
        >>> dist.compute_distance()
        3.1622776601683795
    """

    def __init__(self, weights: Union[NDArrayFloat, None] = None) -> None:
        self.weights = weights

    def __repr__(self) -> str:
        return f"{type(self).__name__}(weights={self.weights})"

    def __call__(self, fd1: FDataBasis, fd2: FDataBasis) -> float:

        if fd1.basis != fd2.basis:
            raise ValueError("Both functional data objects must have the same basis.")

        c1 = fd1.coefficients
        c2 = fd2.coefficients
        M = fd1.basis.gram_matrix()

        if self.weights is None:
            self.weights = np.ones(c1.shape[1])
        else:
            aux = np.zeros(c1.shape[1])
            aux[: self.weights.shape[0]] = self.weights
            self.weights = aux

        return float(np.sqrt(np.sum(self.weights * np.square(np.dot((c1 - c2), M)))))


def basis_based_distance(
    fd1: FDataBasis,
    fd2: FDataBasis,
    weights: Optional[NDArrayFloat] = None,
) -> float:
    r"""
    Computes the distance between two functional data objects represented in the same basis.

    Args:
        fd1 (FDataBasis): First functional data object.
        fd2 (FDataBasis): Second functional data object.
        weights (ndarray, optional): Weighting function \( \nu_k \). Defaults to None.

    Returns:
        float: The computed distance between the two functional data objects.
    """
    distance = BasisBasedDistance(weights)
    return distance(fd1, fd2)
