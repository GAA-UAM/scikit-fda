import numpy as np
import skfda
from skfda.representation.basis import FDataBasis
from skfda.datasets._real_datasets import fetch_growth
from matplotlib import pyplot

class FPCA:
    def __init__(self, n_components, components_basis=None, centering=True):
        self.n_components = n_components
        # component_basis is the basis that we want to use for the principal components
        self.components_basis = components_basis
        self.centering = centering
        self.components = None
        self.component_values = None

    def fit(self, X, y=None):
        # for now lets consider that X is a FDataBasis Object

        # if centering is True then substract the mean function to each function in FDataBasis
        if self.centering:
            meanfd = X.mean()
            # consider moving these lines to FDataBasis as a centering function
            # substract from each row the mean coefficient matrix
            X.coefficients -= meanfd.coefficients

        # for reference, X.coefficients is the C matrix
        n_samples, n_basis = X.coefficients.shape

        # setup principal component basis if not given
        if not self.components_basis:
            self.components_basis = X.basis.copy()

        # if the principal components are in the same basis, this is essentially the gram matrix
        j_matrix = X.basis.inner_product(self.components_basis)

        g_matrix = self.components_basis.gram_matrix()
        l_matrix = np.linalg.cholesky(g_matrix)
        l_matrix_inv = np.linalg.inv(l_matrix)

        # The following matrix is needed: L^(-1)*J^T
        l_inv_j_t = np.matmul(l_matrix_inv, np.transpose(j_matrix))

        # the final matrix (L-1Jt)-1CtC(L-1Jt)t
        final_matrix = (l_inv_j_t @ np.transpose(X.coefficients)
                        @ X.coefficients @ np.transpose(l_inv_j_t))/n_samples

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

        return self

    def transform(self, X, y=None):
        total = sum(self.component_values)
        self.component_values /= total
        return self.component_values[:self.n_components]

    def fit_transform(self, X, y=None):
        pass


if __name__ == '__main__':
    dataset = fetch_growth()
    fd = dataset['data']
    y = dataset['target']

    basis = skfda.representation.basis.BSpline(n_basis=7)
    basisfd = fd.to_basis(basis)
    # print(basisfd.basis.gram_matrix())
    # print(basis.gram_matrix())

    basisfd.plot()
    pyplot.show()

    meanfd = basisfd.mean()

    fpca = FPCA(2)
    fpca.fit(basisfd)

    # fpca.components.plot()
    # pyplot.show()

    meanfd.plot()
    pyplot.show()

    meanfd.coefficients = np.vstack([meanfd.coefficients,
                                     meanfd.coefficients[0, :] + 10 * fpca.components.coefficients[0, :]])

    meanfd.plot()
    pyplot.show()

    # print(fpca.transform(basisfd))






