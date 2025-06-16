import numpy as np

from ...typing._numpy import NDArrayFloat
from .._functional_data import FData
from ..basis import Basis


class FPCABasis(Basis):
    """
    Functional Principal Component basis.

    Basis formed by the first `n_basis` principal components obtained
    through Functional Principal Component Analysis (FPCA) on a given
    sample of functional data.

    The functions in this basis depend on the data used during fitting,
    and represent the directions of maximum variance in the sample.

    Attributes:
        domain_range: Tuple defining the interval over which the functions
            are defined, inferred from the input functional data.
        n_basis: Number of principal components (basis functions) used.
        _fpca: FPCA object used to compute the basis.

    Parameters:
        X: Functional data object used to fit the FPCA model.
        n_basis: Number of FPCA components to retain.

    Examples:
        Constructs a FPCABasis and adjusts it to the Canadian Weather

        >>> from skfda.datasets import fetch_weather
        >>> X, _ = fetch_weather(return_X_y=True)
        >>> X = X.coordinates[0]
        >>> basis = FPCABasis(X=X, n_basis=4)
        >>> basis.plot()
    """

    def __init__(
        self,
        *,
        X: FData,
        n_basis: int = 1,
    ) -> None:
        from skfda.preprocessing.dim_reduction.feature_extraction import FPCA

        super().__init__(domain_range=X.domain_range, n_basis=n_basis)

        self._fpca = FPCA(n_components=n_basis)
        self._fpca.fit(X)

    def _evaluate(self, eval_points: NDArrayFloat) -> NDArrayFloat:
        """
        Evaluate the FPCA basis functions at the given points.

        Args:
            eval_points: Points at which to evaluate the basis functions.

        Returns:
            Array of shape (n_basis, n_eval_points, 1) with the evaluations.
        """
        return self._fpca.components_(eval_points)

    def _derivative_basis_and_coefs(
        self,
        coefs: NDArrayFloat,
        order: int = 1,
    ) -> tuple["FPCABasis", NDArrayFloat]:
        """
        Compute the basis and coefficients of the derivative.

        Note:
            FPCA basis functions are not guaranteed to be differentiable
            depending on the original data representation.

        Args:
            coefs: Coefficients with respect to the original FPCA basis.
            order: Order of the derivative.

        Returns:
            A new FPCABasis object and the new coefficients.
        """
        new_components = self._fpca.components_.derivative(order)
        new_basis = self.copy()
        new_basis._fpca.components_ = new_components  # noqa: SLF001
        return new_basis, coefs

    def _gram_matrix(self) -> NDArrayFloat:
        """
        Return the Gram matrix of the basis.

        Since FPCA components are orthonormal by construction,
        the Gram matrix is the identity matrix.

        Returns:
            Identity matrix of shape (n_basis, n_basis).
        """
        return np.identity(self.n_basis)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FPCABasis):
            return False

        return (
            super().__eq__(other)
            and self._fpca.components_ == other._fpca.components_
        )
