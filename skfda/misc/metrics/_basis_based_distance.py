from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...representation import FDataBasis
    from ...typing._numpy import NDArrayFloat


class BasisBasedDistance:
    r"""
    Weighted distance between two FDataBasis observations.

    This class computes the distance between two functional data objects
    represented in the same basis. Given two functional observations
    \(X_1, X_2\), their basis representations are:

    .. math::
        X_1(t) = \sum_{k=1}^{K} c_{1,k} \phi_k(t), \quad
        X_2(t) = \sum_{k=1}^{K} c_{2,k} \phi_k(t).

    If the basis is orthogonal, a natural distance between \(X_1\) and \(X_2\)
    is defined in terms of the basis coefficients \(c_{1,k}\) and \(c_{2,k}\):

    .. math::
        d_{\nu}(\mathbf{X}_1, \mathbf{X}_2) = \sqrt{\sum_{k=1}^{K} \nu_k
        \left(c_{1,k} - c_{2,k}\right)^2 }.

    Where:
        - \( c_{n,k} = \langle X_n, \phi_k \rangle_\mu \) are the basis
         coefficients.
        - \( \nu_k \) is a weighting function controlling the contribution of
         each basis function.

    If the basis \( \{\phi_k\}_{k=1}^K \) is not orthonormal, the distance is
      generalized as:

    .. math::
        d_{\nu}(\mathbf{X}_1, \mathbf{X}_2) =
        \sqrt{\sum_{i=1}^{K} \nu_i \left( \sum_{j=1}^K (c_{1,j} - c_{2,j})
         \langle \phi_j, \phi_i \rangle \right)^2 }.

    This can be compactly written as:

    .. math::
        d_{\nu}^2(\mathbf{X}_1, \mathbf{X}_2) =
        \langle \mathbf{\nu}, \left( (\mathbf{c_1 - c_2})^T M \right)^2
          \rangle,

    where \( M \) is the Gram matrix of inner products between the basis
     functions:

    .. math::
        M_{i,j} = \langle \phi_i, \phi_j \rangle.

    The Gram matrix \( M \) has the following form:

    .. math::
        M =
        \begin{pmatrix}
        \langle \phi_1, \phi_1 \rangle & \langle \phi_1, \phi_2 \rangle &
        \cdots & \langle \phi_1, \phi_K \rangle \\
        \langle \phi_2, \phi_1 \rangle & \langle \phi_2, \phi_2 \rangle &
        \cdots & \langle \phi_2, \phi_K \rangle \\
        \vdots & \vdots & \ddots & \vdots \\
        \langle \phi_K, \phi_1 \rangle & \langle \phi_K, \phi_2 \rangle &
        \cdots & \langle \phi_K, \phi_K \rangle
        \end{pmatrix}.

    This class takes two `FDataBasis` objects with the same basis and computes
    the functional distance between them.

    Attributes:
        fd1: First functional data object.
        fd2: Second functional data object.
        weights: Weighting function \( \nu_k \).
        gram_matrix: Gram matrix \( M \) of the basis functions.

    Example:
        >>> from skfda.representation.basis import FDataBasis, BSplineBasis
        >>> import numpy as np
        >>> from skfda.misc.metrics import BasisBasedDistance
        >>> basis = BSplineBasis(n_basis=5)
        >>> fd1 = FDataBasis(basis, np.array([1, 2, 3, 4, 5]))
        >>> fd2 = FDataBasis(basis, np.array([2, 3, 1, 5, 4]))
        >>> weights = np.ones(5)
        >>> dist = BasisBasedDistance(weights)(fd1, fd2)
        >>> dist
        0.13226345847958026
    """

    def __init__(self, weights: NDArrayFloat | None = None) -> None:
        self.weights = weights

    def __repr__(self) -> str:
        return f"{type(self).__name__}(weights={self.weights})"

    def __call__(self, fd1: FDataBasis, fd2: FDataBasis) -> NDArrayFloat:

        if fd1.basis != fd2.basis:
            msg = "Both functional data objects must have the same basis."
            raise ValueError(msg)

        c1 = fd1.coefficients  # shape (n_samples_1, n_basis)
        c2 = fd2.coefficients  # shape (n_samples_2, n_basis)

        M = fd1.basis.gram_matrix()  # shape (n_basis, n_basis)  # noqa: N806

        # Prepare weights
        n_basis = c1.shape[1]
        if self.weights is None:
            weights = np.ones(n_basis)
        else:
            weights = np.zeros(n_basis)
            weights[: self.weights.shape[0]] = self.weights

        # If c2 has one sample and c1 has many, broadcast c2
        if c1.shape[0] != c2.shape[0]:
            if c2.shape[0] == 1:
                c2 = np.repeat(c2, c1.shape[0], axis=0)
            else:
                msg = (
                    "fd1 and fd2 must have the same number"
                    " of samples or fd2 must have one sample."
                )
                raise ValueError(msg)

        delta = np.dot((c1 - c2), M)  # shape (n_samples, n_basis)

        # Weighted squared norm per sample
        squared_dists = np.sum(
            weights * delta**2, axis=1,
        )  # shape (n_samples,)
        res = np.sqrt(squared_dists)
        return res[0] if len(res) == 1 else res  # type: ignore[no-any-return]
