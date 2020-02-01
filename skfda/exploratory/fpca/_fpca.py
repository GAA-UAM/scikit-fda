"""Functional Principal Component Analysis Module."""

import numpy as np
from abc import ABC, abstractmethod
from skfda.representation.basis import FDataBasis
from skfda.representation.grid import FDataGrid
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA


__author__ = "Yujian Hong"
__email__ = "yujian.hong@estudiante.uam.es"


class FPCA(ABC, BaseEstimator, ClassifierMixin):
    # TODO doctring
    # TODO doctest
    # TODO directory examples create test
    """Defines the common structure shared between classes that do functional
    principal component analysis

    Attributes:
        n_components (int): number of principal components to obtain from
            functional principal component analysis. Defaults to 3.
        centering (bool): if True then calculate the mean of the functional data
            object and center the data first
        components (FDataGrid or FDataBasis): this contains the principal
            components either in a basis form or discretized form
        component_values (array_like): this contains the values (eigenvalues)
            associated with the principal components
        pca (sklearn.decomposition.PCA): object for principal component analysis.
            In both cases (discretized FPCA and basis FPCA) the problem can be
            reduced to a regular PCA problem and use the framework provided by
            sklearn to continue.
    """

    def __init__(self, n_components=3, centering=True):
        """FPCA constructor

        Args:
            n_components (int): number of principal components to obtain from
                functional principal component analysis
            centering (bool): if True then calculate the mean of the functional
                data object and center the data first. Defaults to True
        """
        self.n_components = n_components
        self.centering = centering
        self.components = None
        self.component_values = None
        self.pca = PCA(n_components=self.n_components)

    @abstractmethod
    def fit(self, X, y=None):
        """Computes the n_components first principal components and saves them
        inside the FPCA object.

        Args:
            X (FDataGrid or FDataBasis):
                the functional data object to be analysed
            y (None, not used):
                only present for convention of a fit function

        Returns:
            self (object)
        """
        pass

    @abstractmethod
    def transform(self, X, y=None):
        """Computes the n_components first principal components score and
        returns them.

        Args:
            X (FDataGrid or FDataBasis):
                the functional data object to be analysed
            y (None, not used):
                only present because of fit function convention

        Returns:
            (array_like): the scores of the data with reference to the
            principal components
        """
        pass

    def fit_transform(self, X, y=None):
        """
        Computes the n_components first principal components and their scores
        and returns them.
            Args:
                X (FDataGrid or FDataBasis):
                    the functional data object to be analysed
                y (None, not used):
                    only present for convention of a fit function

            Returns:
                (array_like): the scores of the data with reference to the
                principal components
        """
        self.fit(X, y)
        return self.transform(X, y)


class FPCABasis(FPCA):
    """Defines the common structure shared between classes that do functional
    principal component analysis

    Attributes:
        n_components (int): number of principal components to obtain from
            functional principal component analysis. Defaults to 3.
        centering (bool): if True then calculate the mean of the functional data
            object and center the data first. Defaults to True. If True the
            passed FDataBasis object is modified.
        components (FDataBasis): this contains the principal components either
            in a basis form or discretized form
        component_values (array_like): this contains the values (eigenvalues)
            associated with the principal components
        pca (sklearn.decomposition.PCA): object for principal component analysis.
            In both cases (discretized FPCA and basis FPCA) the problem can be
            reduced to a regular PCA problem and use the framework provided by
            sklearn to continue.
    """

    def __init__(self, n_components=3, components_basis=None, centering=True):
        """FPCABasis constructor

        Args:
            n_components (int): number of principal components to obtain from
                functional principal component analysis
            components_basis (skfda.representation.Basis): the basis in which we
                want the principal components. Defaults to None. If so, the
                basis contained in the passed FDataBasis object for the fit
                function will be used.
            centering (bool): if True then calculate the mean of the functional
                data object and center the data first. Defaults to True
        """
        super().__init__(n_components, centering)
        # basis that we want to use for the principal components
        self.components_basis = components_basis

    def fit(self, X: FDataBasis, y=None):
        """Computes the n_components first principal components and saves them
            inside the FPCA object.

        Args:
            X (FDataBasis):
                the functional data object to be analysed in basis
                representation
            y (None, not used):
                only present for convention of a fit function

        Returns:
            self (object)

        References:
            .. [RS05-8-4-2] Ramsay, J., Silverman, B. W. (2005). Basis function
                expansion of the functions. In *Functional Data Analysis*
                (pp. 161-164). Springer.

        """
        # check that the number of components is smaller than the sample size
        if self.n_components > X.n_samples:
            raise AttributeError("The sample size must be bigger than the "
                                 "number of components")

        # check that we do not exceed limits for n_components as it should
        # be smaller than the number of attributes of the basis
        n_basis = self.components_basis.n_basis if self.components_basis \
            else X.basis.n_basis
        if self.n_components > n_basis:
            raise AttributeError("The number of components should be "
                                 "smaller than the number of attributes of "
                                 "target principal components' basis.")


        # if centering is True then subtract the mean function to each function
        # in FDataBasis
        if self.centering:
            meanfd = X.mean()
            # consider moving these lines to FDataBasis as a centering function
            # subtract from each row the mean coefficient matrix
            X.coefficients -= meanfd.coefficients

        # for reference, X.coefficients is the C matrix
        n_samples, n_basis = X.coefficients.shape

        # setup principal component basis if not given
        if self.components_basis:
            # First fix domain range if not already done
            self.components_basis.domain_range = X.basis.domain_range
            g_matrix = self.components_basis.gram_matrix()
            # the matrix that are in charge of changing the computed principal
            # components to target matrix is essentially the inner product
            # of both basis.
            j_matrix = X.basis.inner_product(self.components_basis)
        else:
            # if no other basis is specified we use the same basis as the passed
            # FDataBasis Object
            self.components_basis = X.basis.copy()
            g_matrix = self.components_basis.gram_matrix()
            j_matrix = g_matrix

        # make g matrix symmetric, referring to Ramsay's implementation
        g_matrix = (g_matrix + np.transpose(g_matrix))/2

        # obtain triangulation using cholesky
        l_matrix = np.linalg.cholesky(g_matrix)

        # L^{-1}
        l_matrix_inv = np.linalg.inv(l_matrix)

        # The following matrix is needed: L^{-1}*J^T
        l_inv_j_t = l_matrix_inv @ np.transpose(j_matrix)

        # the final matrix, C(L-1Jt)t for svd or (L-1Jt)-1CtC(L-1Jt)t for PCA
        final_matrix = X.coefficients @ np.transpose(l_inv_j_t) / \
            np.sqrt(n_samples)

        self.pca.fit(final_matrix)
        self.component_values = self.pca.singular_values_ ** 2
        self.components = X.copy(basis=self.components_basis,
                                 coefficients=self.pca.components_
                                 @ l_matrix_inv)
        """
        if self.svd:
            # vh contains the eigenvectors transposed
            # s contains the singular values, which are square roots of eigenvalues
            u, s, vh = np.linalg.svd(final_matrix, full_matrices=True, compute_uv=True)
            principal_components = vh @ l_matrix_inv
            self.components = X.copy(basis=self.components_basis,
                                     coefficients=principal_components[:self.n_components, :])
            self.component_values = s ** 2
        else:
            final_matrix = np.transpose(final_matrix) @ final_matrix

            # perform eigenvalue and eigenvector analysis on this matrix
            # eigenvectors is a numpy array, such that its columns are eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(final_matrix)

            # sort the eigenvalues and eigenvectors from highest to lowest
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            principal_components_t = np.transpose(l_matrix_inv) @ eigenvectors

            # we only want the first ones, determined by n_components
            principal_components_t = principal_components_t[:, :self.n_components]

            self.components = X.copy(basis=self.components_basis,
                                     coefficients=np.transpose(principal_components_t))

            self.component_values = eigenvalues
        """

        return self

    def transform(self, X, y=None):
        # in this case it is the inner product of our data with the components
        return X.inner_product(self.components)


class FPCADiscretized(FPCA):
    def __init__(self, n_components=3, weights=None, centering=True):
        super().__init__(n_components, centering)
        self.weights = weights

    # noinspection PyPep8Naming
    def fit(self, X: FDataGrid, y=None):
        """Computes the n_components first principal components and saves them
            inside the FPCA object.

        Args:
            X (FDataBasis):
                the functional data object to be analysed in basis
                representation
            y (None, not used):
                only present for convention of a fit function

        Returns:
            self (object)

        References:
            .. [RS05-8-4-1] Ramsay, J., Silverman, B. W. (2005). Discretizing
            the functions. In *Functional Data Analysis* (p. 161). Springer.
        """

        # check that the number of components is smaller than the sample size
        if self.n_components > X.n_samples:
            raise AttributeError("The sample size must be bigger than the "
                                 "number of components")

        # check that we do not exceed limits for n_components as it should
        # be smaller than the number of attributes of the funcional data object
        if self.n_components > X.data_matrix.shape[1]:
            raise AttributeError("The number of components should be "
                                 "smaller than the number of discretization "
                                 "points of the functional data object.")


        # data matrix initialization
        fd_data = np.squeeze(X.data_matrix)

        # get the number of samples and the number of points of descretization
        n_samples, n_points_discretization = fd_data.shape

        # if centering is True then subtract the mean function to each function
        # in FDataBasis
        if self.centering:
            meanfd = X.mean()
            # consider moving these lines to FDataBasis as a centering function
            # subtract from each row the mean coefficient matrix
            fd_data -= np.squeeze(meanfd.data_matrix)

        # establish weights for each point of discretization
        if not self.weights:
            # sample_points is a list with one array in the 1D case
            # in trapezoidal rule, suppose \deltax_k = x_k - x_{k-1}, the weight
            # vector is as follows: [\deltax_1/2, \deltax_1/2 + \deltax_2/2,
            # \deltax_2/2 + \deltax_3/2, ... , \deltax_n/2]
            differences = np.diff(X.sample_points[0])
            self.weights = [sum(differences[i:i + 2]) / 2 for i in
                            range(len(differences))]
            self.weights = np.concatenate(([differences[0] / 2], self.weights))

        weights_matrix = np.diag(self.weights)

        # k_estimated is not used for the moment
        # k_estimated = fd_data @ np.transpose(fd_data) / n_samples

        final_matrix = fd_data @ np.sqrt(weights_matrix) / np.sqrt(n_samples)
        self.pca.fit(final_matrix)
        self.components = X.copy(data_matrix=self.pca.components_)
        self.component_values = self.pca.singular_values_ ** 2

        """
        if self.svd:
            # vh contains the eigenvectors transposed
            # s contains the singular values, which are square roots of eigenvalues
            u, s, vh = np.linalg.svd(final_matrix, full_matrices=True, compute_uv=True)
            self.components = X.copy(data_matrix=vh[:self.n_components, :])
            self.component_values = s**2
        else:
            # perform eigenvalue and eigenvector analysis on this matrix
            # eigenvectors is a numpy array, such that its columns are eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(np.transpose(final_matrix) @ final_matrix)

            # sort the eigenvalues and eigenvectors from highest to lowest
            # the eigenvectors are the principal components
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            principal_components_t = eigenvectors[:, idx]

            # we only want the first ones, determined by n_components
            principal_components_t = principal_components_t[:, :self.n_components]

            # prepare the computed principal components
            self.components = X.copy(data_matrix=np.transpose(principal_components_t))
            self.component_values = eigenvalues
        """
        return self

    def transform(self, X, y=None):
        # in this case its the coefficient matrix multiplied by the principal
        # components as column vectors
        return np.squeeze(X.data_matrix) @ np.transpose(
            np.squeeze(self.components.data_matrix))
