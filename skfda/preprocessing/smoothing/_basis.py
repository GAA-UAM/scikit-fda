# -*- coding: utf-8 -*-
"""Basis smoother.

This module contains the class for the basis smoothing.

"""
from enum import Enum
from typing import Union

import scipy.linalg

import numpy as np

from ... import FDataBasis
from ... import FDataGrid
from ._linear import _LinearSmoother, _check_r_to_r


class _Cholesky():
    """Solve the linear equation using cholesky factorization"""

    def __call__(self, *, basis_values, weight_matrix, data_matrix,
                 penalty_matrix, **_):

        right_matrix = basis_values.T @ weight_matrix @ data_matrix
        left_matrix = basis_values.T @ weight_matrix @ basis_values

        # Adds the roughness penalty to the equation
        if penalty_matrix is not None:
            left_matrix += penalty_matrix

        coefficients = scipy.linalg.cho_solve(scipy.linalg.cho_factor(
            left_matrix, lower=True), right_matrix)

        # The ith column is the coefficients of the ith basis for each
        #  sample
        coefficients = coefficients.T

        return coefficients


class _QR():
    """Solve the linear equation using qr factorization"""

    def __call__(self, *, basis_values, weight_matrix, data_matrix,
                 penalty_matrix, ndegenerated, **_):

        if weight_matrix is not None:
            # Decompose W in U'U and calculate UW and Uy
            upper = scipy.linalg.cholesky(weight_matrix)
            basis_values = upper @ basis_values
            data_matrix = upper @ data_matrix

        if penalty_matrix is not None:
            w, v = np.linalg.eigh(penalty_matrix)
            # Reduction of the penalty matrix taking away 0 or almost
            # zeros eigenvalues

            if ndegenerated:
                index = ndegenerated - 1
            else:
                index = None
            w = w[:index:-1]
            v = v[:, :index:-1]

            penalty_matrix = v @ np.diag(np.sqrt(w))
            # Augment the basis matrix with the square root of the
            # penalty matrix
            basis_values = np.concatenate([
                basis_values,
                penalty_matrix.T],
                axis=0)
            # Augment data matrix by n - ndegenerated zeros
            data_matrix = np.pad(data_matrix,
                                 ((0, len(v) - ndegenerated),
                                  (0, 0)),
                                 mode='constant')

        # Resolves the equation
        # B.T @ B @ C = B.T @ D
        # by means of the QR decomposition

        # B = Q @ R
        q, r = np.linalg.qr(basis_values)
        right_matrix = q.T @ data_matrix

        # R @ C = Q.T @ D
        coefficients = np.linalg.solve(r, right_matrix)
        # The ith column is the coefficients of the ith basis for each
        # sample
        coefficients = coefficients.T

        return coefficients


class _Matrix():
    """Solve the linear equation using matrix inversion"""

    def fit(self, estimator, X, y=None):
        if estimator.return_basis:
            estimator._cached_coef_matrix = estimator._coef_matrix(
                self.input_points_)
        else:
            # Force caching the hat matrix
            estimator.hat_matrix()

    def fit_transform(self, estimator, X, y=None):
        return estimator.fit().transform()

    def __call__(self, *, estimator, **_):
        pass

    def transform(self, estimator, X, y=None):
        if estimator.return_basis:
            coefficients = estimator._cached_coef_matrix @ X.data_matrix

            fdatabasis = FDataBasis(
                basis=self.basis, coefficients=coefficients,
                keepdims=self.keepdims)

            return fdatabasis
        else:
            # The matrix is cached
            return X.copy(data_matrix=self.hat_matrix() @ X.data_matrix,
                          sample_points=self.output_points_)


class BasisSmoother(_LinearSmoother):

    class SolverMethod(Enum):
        cholesky = _Cholesky()
        qr = _QR()
        matrix = _Matrix()

    def __init__(self, *,
                 basis,
                 smoothing_parameter: float = 0,
                 weights=None,
                 penalty: Union[int, np.ndarray,
                                'LinearDifferentialOperator'] = None,
                 penalty_matrix=None,
                 output_points=None,
                 method='cholesky',
                 keepdims=False,
                 return_basis=False):
        self.basis = basis
        self.smoothing_parameter = smoothing_parameter
        self.weights = weights
        self.penalty = penalty
        self.penalty_matrix = penalty_matrix
        self.output_points = output_points
        self.method = method
        self.keepdims = keepdims
        self.return_basis = return_basis

    def _method_function(self):
        """ Return the method function"""
        method_function = self.method
        if not callable(method_function):
            method_function = self.SolverMethod[
                method_function.lower()].value

        return method_function

    def _penalty(self):
        from ...misc import LinearDifferentialOperator

        """Get the penalty differential operator."""
        if self.penalty is None:
            penalty = LinearDifferentialOperator(order=2)
        elif isinstance(self.penalty, int):
            penalty = LinearDifferentialOperator(order=self.penalty)
        elif isinstance(self.penalty, np.ndarray):
            penalty = LinearDifferentialOperator(weights=self.penalty)
        else:
            penalty = self.penalty

        return penalty

    def _penalty_matrix(self):
        """Get the final penalty matrix.

        The smoothing parameter is already multiplied by it.

        """

        if self.penalty_matrix is not None:
            penalty_matrix = self.penalty_matrix
        else:
            penalty = self._penalty()

            if self.smoothing_parameter > 0:
                penalty_matrix = self.basis.penalty(penalty.order,
                                                    penalty.weights)
            else:
                penalty_matrix = None

        if penalty_matrix is not None:
            penalty_matrix *= self.smoothing_parameter

        return penalty_matrix

    def _coef_matrix(self, input_points):
        """Get the matrix that gives the coefficients"""
        basis_values_input = self.basis.evaluate(input_points).T

        # If no weight matrix is given all the weights are one
        weight_matrix = (self.weights if self.weights is not None
                         else np.identity(basis_values_input.shape[0]))

        inv = basis_values_input.T @ weight_matrix @ basis_values_input

        penalty_matrix = self._penalty_matrix()
        if penalty_matrix is not None:
            inv += penalty_matrix

        inv = np.linalg.inv(inv)

        return inv @ basis_values_input

    def _hat_matrix(self, input_points, output_points):
        basis_values_output = self.basis.evaluate(output_points).T

        return basis_values_output.T @ self._coef_matrix(input_points)

    def fit(self, X: FDataGrid, y=None):
        """Compute the hat matrix for the desired output points.

        Args:
            X (FDataGrid):
                The data whose points are used to compute the matrix.
            y : Ignored
        Returns:
            self (object)

        """
        _check_r_to_r(X)

        self.input_points_ = X.sample_points[0]
        self.output_points_ = (self.output_points
                               if self.output_points is not None
                               else self.input_points_)

        method = self._method_function()
        method_fit = getattr(method, "fit", None)
        if method_fit is not None:
            method_fit(estimator=self, X=X, y=y)

        return self

    def fit_transform(self, X: FDataGrid, y=None):
        """Compute the hat matrix for the desired output points.

        Args:
            X (FDataGrid):
                The data whose points are used to compute the matrix.
            y : Ignored
        Returns:
            self (object)

        """

        _check_r_to_r(X)

        self.input_points_ = X.sample_points[0]
        self.output_points_ = (self.output_points
                               if self.output_points is not None
                               else self.input_points_)

        penalty_matrix = self._penalty_matrix()

        # n is the samples
        # m is the observations
        # k is the number of elements of the basis

        # Each sample in a column (m x n)
        data_matrix = X.data_matrix[..., 0].T

        # Each basis in a column
        basis_values = self.basis.evaluate(self.input_points_).T

        # If no weight matrix is given all the weights are one
        weight_matrix = (self.weights if self.weights is not None
                         else np.identity(basis_values.shape[0]))

        # We need to solve the equation
        # (phi' W phi + lambda * R) C = phi' W Y
        # where:
        #  phi is the basis_values
        #  W is the weight matrix
        #  lambda the smoothness parameter
        #  C the coefficient matrix (the unknown)
        #  Y is the data_matrix

        if(data_matrix.shape[0] > self.basis.nbasis
           or self.smoothing_parameter > 0):

            # TODO: The penalty could be None (if the matrix is passed)
            ndegenerated = self.basis._ndegenerated(self._penalty().order)

            method = self._method_function()

            # If the method provides the complete transformation use it
            method_fit_transform = getattr(method, "fit_transform", None)
            if method_fit_transform is not None:
                return method_fit_transform(estimator=self, X=X, y=y)

            # Otherwise the method is used to compute the coefficients
            coefficients = method(estimator=self,
                                  basis_values=basis_values,
                                  weight_matrix=weight_matrix,
                                  data_matrix=data_matrix,
                                  penalty_matrix=penalty_matrix,
                                  ndegenerated=ndegenerated)

        elif data_matrix.shape[0] == self.basis.nbasis:
            # If the number of basis equals the number of points and no
            # smoothing is required
            coefficients = np.linalg.solve(basis_values, data_matrix)

        else:  # data_matrix.shape[0] < basis.nbasis
            raise ValueError(f"The number of basis functions "
                             f"({self.basis.nbasis}) "
                             f"exceed the number of points to be smoothed "
                             f"({data_matrix.shape[0]}).")

        fdatabasis = FDataBasis(
            basis=self.basis, coefficients=coefficients,
            keepdims=self.keepdims)

        if self.return_basis:
            return fdatabasis
        else:
            return fdatabasis(self.output_points_)

        return self

    def transform(self, X: FDataGrid, y=None):
        """Apply the smoothing.

        Args:
            X (FDataGrid):
                The data to smooth.
            y : Ignored
        Returns:
            self (object)

        """

        assert all(self.input_points_ == X.sample_points[0])

        method = self._method_function()

        # If the method provides the complete transformation use it
        method_transform = getattr(method, "transform", None)
        if method_transform is not None:
            return method_transform(estimator=self, X=X, y=y)

        # Otherwise use fit_transform over the data
        # Note that data leakage is not possible because the matrix only
        # depends on the input/output points
        return self.fit_transform(X, y)
