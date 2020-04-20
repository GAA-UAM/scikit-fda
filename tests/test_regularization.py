from skfda.misc.regularization import LinearDifferentialOperatorRegularization
from skfda.misc.regularization._linear_diff_op_regularization import (
    _monomial_evaluate_constant_linear_diff_op)
from skfda.representation.basis import Constant, Monomial, BSpline, Fourier
import unittest

import numpy as np


class TestLinearDifferentialOperatorRegularization(unittest.TestCase):

    # def setUp(self): could be defined for set up before any test

    def _test_penalty(self, basis, linear_diff_op, atol=0, result=None):

        regularization = LinearDifferentialOperatorRegularization(
            linear_diff_op)

        penalty = regularization.penalty_matrix(basis)
        numerical_penalty = regularization.penalty_matrix_numerical(basis)

        np.testing.assert_allclose(
            penalty,
            numerical_penalty,
            atol=atol
        )

        if result is not None:
            np.testing.assert_allclose(
                penalty,
                result,
                atol=atol
            )

    def test_constant_penalty(self):
        basis = Constant(domain_range=(0, 3))

        res = np.array([[12]])

        self._test_penalty(basis, linear_diff_op=[2, 3, 4], result=res)

    def test_monomial_linear_diff_op(self):
        n_basis = 5

        basis = Monomial(n_basis=n_basis)

        linear_diff_op = [3]
        res = np.array([[0., 0., 0., 0., 3.],
                        [0., 0., 0., 3., 0.],
                        [0., 0., 3., 0., 0.],
                        [0., 3., 0., 0., 0.],
                        [3., 0., 0., 0., 0.]])

        np.testing.assert_allclose(
            _monomial_evaluate_constant_linear_diff_op(basis, linear_diff_op),
            res
        )

        linear_diff_op = [3, 2]
        res = np.array([[0., 0., 0., 0., 3.],
                        [0., 0., 0., 3., 2.],
                        [0., 0., 3., 4., 0.],
                        [0., 3., 6., 0., 0.],
                        [3., 8., 0., 0., 0.]])

        np.testing.assert_allclose(
            _monomial_evaluate_constant_linear_diff_op(basis, linear_diff_op),
            res
        )

        linear_diff_op = [3, 0, 5]
        res = np.array([[0., 0., 0., 0., 3.],
                        [0., 0., 0., 3., 0.],
                        [0., 0., 3., 0., 10.],
                        [0., 3., 0., 30., 0.],
                        [3., 0., 60., 0., 0.]])

        np.testing.assert_allclose(
            _monomial_evaluate_constant_linear_diff_op(basis, linear_diff_op),
            res
        )

    def test_monomial_penalty(self):
        basis = Monomial(n_basis=5, domain_range=(0, 3))

        # Theorethical result
        res = np.array([[0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [0., 0., 12., 54., 216.],
                        [0., 0., 54., 324., 1458.],
                        [0., 0., 216., 1458., 6998.4]])

        self._test_penalty(basis, linear_diff_op=2, result=res)

        basis = Monomial(n_basis=8, domain_range=(1, 5))

        self._test_penalty(basis, linear_diff_op=[1, 2, 3])
        self._test_penalty(basis, linear_diff_op=7)
        self._test_penalty(basis, linear_diff_op=0)
        self._test_penalty(basis, linear_diff_op=1)
        self._test_penalty(basis, linear_diff_op=27)

    def test_fourier_penalty(self):
        basis = Fourier(n_basis=5)

        res = np.array([[0., 0., 0., 0., 0.],
                        [0., 1558.55, 0., 0., 0.],
                        [0., 0., 1558.55, 0., 0.],
                        [0., 0., 0., 24936.73, 0.],
                        [0., 0., 0., 0., 24936.73]])

        # Those comparisons require atol as there are zeros involved
        self._test_penalty(basis, linear_diff_op=2, atol=0.01, result=res)

        basis = Fourier(n_basis=9, domain_range=(1, 5))
        self._test_penalty(basis, linear_diff_op=[1, 2, 3], atol=1e-7)
        self._test_penalty(basis, linear_diff_op=[2, 3, 0.1, 1], atol=1e-7)
        self._test_penalty(basis, linear_diff_op=0, atol=1e-7)
        self._test_penalty(basis, linear_diff_op=1, atol=1e-7)
        self._test_penalty(basis, linear_diff_op=3, atol=1e-7)

    def test_bspline_penalty(self):
        basis = BSpline(n_basis=5)

        res = np.array([[96., -132., 24., 12., 0.],
                        [-132., 192., -48., -24., 12.],
                        [24., -48., 48., -48., 24.],
                        [12., -24., -48., 192., -132.],
                        [0., 12., 24., -132., 96.]])

        self._test_penalty(basis, linear_diff_op=2, result=res)

        basis = BSpline(n_basis=9, domain_range=(1, 5))
        self._test_penalty(basis, linear_diff_op=[1, 2, 3])
        self._test_penalty(basis, linear_diff_op=[2, 3, 0.1, 1])
        self._test_penalty(basis, linear_diff_op=0)
        self._test_penalty(basis, linear_diff_op=1)
        self._test_penalty(basis, linear_diff_op=3)
        self._test_penalty(basis, linear_diff_op=4)

        basis = BSpline(n_basis=16, order=8)
        self._test_penalty(basis, linear_diff_op=0, atol=1e-7)

    def test_bspline_penalty_special_case(self):
        basis = BSpline(n_basis=5)

        res = np.array([[1152., -2016., 1152., -288., 0.],
                        [-2016., 3600., -2304., 1008., -288.],
                        [1152., -2304., 2304., -2304., 1152.],
                        [-288., 1008., -2304., 3600., -2016.],
                        [0., -288., 1152., -2016., 1152.]])

        regularization = LinearDifferentialOperatorRegularization(
            basis.order - 1)
        penalty = regularization.penalty_matrix(basis)
        numerical_penalty = regularization.penalty_matrix_numerical(basis)

        np.testing.assert_allclose(
            penalty,
            res
        )

        np.testing.assert_allclose(
            numerical_penalty,
            res
        )
