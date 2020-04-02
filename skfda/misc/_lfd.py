import numbers

import scipy.linalg

import numpy as np


__author__ = "Pablo Pérez Manso"
__email__ = "92manso@gmail.com"


class LinearDifferentialOperator:
    """Defines the structure of a linear differential operator function system

    .. math::
        Lx(t) = b_0(t) x(t) + b_1(t) x'(x) +
                \\dots + b_{n-1}(t) d^{n-1}(x(t)) + b_n(t) d^n(x(t))

    Attributes:
        order (int): the order of the operator. It's the n coefficient in the
                     equation above.

        weights (list):  A FDataBasis objects list of length order + 1

    Examples:

        Create a linear differential operator that penalizes the second
        derivative (acceleration)

        >>> from skfda.misc import LinearDifferentialOperator
        >>> from skfda.representation.basis import (FDataBasis,
        ...                                         Monomial, Constant)
        >>>
        >>> LinearDifferentialOperator(2)
        LinearDifferentialOperator(
        weights=[
        FDataBasis(
            basis=Constant(domain_range=[array([0, 1])], n_basis=1),
            coefficients=[[0]],
            ...),
        FDataBasis(
            basis=Constant(domain_range=[array([0, 1])], n_basis=1),
            coefficients=[[0]],
            ...),
        FDataBasis(
            basis=Constant(domain_range=[array([0, 1])], n_basis=1),
            coefficients=[[1]],
            ...)]
        )

        Create a linear differential operator that penalizes three times
        the second derivative (acceleration) and twice the first (velocity).

        >>> LinearDifferentialOperator(weights=[0, 2, 3])
        LinearDifferentialOperator(
        weights=[
        FDataBasis(
            basis=Constant(domain_range=[array([0, 1])], n_basis=1),
            coefficients=[[0]],
            ...),
        FDataBasis(
            basis=Constant(domain_range=[array([0, 1])], n_basis=1),
            coefficients=[[2]],
            ...),
        FDataBasis(
            basis=Constant(domain_range=[array([0, 1])], n_basis=1),
            coefficients=[[3]],
            ...)]
        )

        Create a linear differential operator with non-constant weights.

        >>> constant = Constant()
        >>> monomial = Monomial((0, 1), n_basis=3)
        >>> fdlist = [FDataBasis(constant, [0]),
        ...           FDataBasis(constant, [0]),
        ...           FDataBasis(monomial, [1, 2, 3])]
        >>> LinearDifferentialOperator(weights=fdlist)
        LinearDifferentialOperator(
        weights=[
        FDataBasis(
            basis=Constant(domain_range=[array([0, 1])], n_basis=1),
            coefficients=[[0]],
            ...),
        FDataBasis(
            basis=Constant(domain_range=[array([0, 1])], n_basis=1),
            coefficients=[[0]],
            ...),
        FDataBasis(
            basis=Monomial(domain_range=[array([0, 1])], n_basis=3),
            coefficients=[[1 2 3]],
            ...)]
        )

    """

    def __init__(self, order_or_weights=None, *, order=None, weights=None,
                 domain_range=None):
        """Lfd Constructor. You have to provide one of the two first
           parameters. It both are provided, it will raise an error.
           If a positional argument is supplied it will be considered the
           order if it is an integral type and the weights otherwise.

        Args:
            order (int, optional): the order of the operator. It's the highest
                    derivative order of the operator

            weights (list, optional): A FDataBasis objects list of length
                    order + 1 items

            domain_range (tuple or list of tuples, optional): Definition
                         of the interval where the weight functions are
                         defined. If the functional weights are specified
                         and this is not, takes the domain range from them.
                         Otherwise, defaults to (0,1).
        """

        from ..representation.basis import (FDataBasis, Constant,
                                            _same_domain)

        num_args = sum(
            [a is not None for a in [order_or_weights, order, weights]])

        if num_args > 1:
            raise ValueError("You have to provide the order or the weights, "
                             "not both")

        real_domain_range = (domain_range if domain_range is not None
                             else (0, 1))

        if order_or_weights is not None:
            if isinstance(order_or_weights, numbers.Integral):
                order = order_or_weights
            else:
                weights = order_or_weights

        if order is None and weights is None:
            self.weights = (FDataBasis(Constant(real_domain_range), 0),)

        elif weights is None:
            if order < 0:
                raise ValueError("Order should be an non-negative integer")

            self.weights = [
                FDataBasis(Constant(real_domain_range),
                           0 if (i < order) else 1)
                for i in range(order + 1)]

        else:
            if len(weights) == 0:
                raise ValueError("You have to provide one weight at least")

            if all(isinstance(n, numbers.Real) for n in weights):
                self.weights = (FDataBasis(Constant(real_domain_range),
                                           np.array(weights)
                                           .reshape(-1, 1)).to_list())

            elif all(isinstance(n, FDataBasis) for n in weights):
                if all([_same_domain(weights[0].domain_range,
                                     x.domain_range) and x.n_samples == 1 for x
                        in weights]):
                    self.weights = weights

                    real_domain_range = weights[0].domain_range
                    if (domain_range is not None
                            and real_domain_range != domain_range):
                        raise ValueError("The domain range provided for the "
                                         "linear operator does not match the "
                                         "domain range of the weights")

                else:
                    raise ValueError("FDataBasis objects in the list have "
                                     "not the same domain_range")

            else:
                raise ValueError("The elements of the list are neither "
                                 "integers or FDataBasis objects")

        self.domain_range = real_domain_range

    def __repr__(self):
        """Representation of Lfd object."""

        bwtliststr = ""
        for w in self.weights:
            bwtliststr = bwtliststr + "\n" + repr(w) + ","

        return (f"{self.__class__.__name__}("
                f"\nweights=[{bwtliststr[:-1]}]"
                f"\n)").replace('\n', '\n    ')

    def __eq__(self, other):
        """Equality of Lfd objects"""
        return (self.weights == other.weights)

    def constant_weights(self):
        """
        Return the scalar weights of the linear differential operator if they
        are constant basis.
        Otherwise, return None.

        This function is mostly useful for basis which want to override
        the _penalty method in order to use an analytical expression
        for constant weights.
        """
        from ..representation.basis import Constant

        coefs = [w.coefficients[0, 0] if isinstance(w.basis, Constant)
                 else None
                 for w in self.weights]

        return np.array(coefs) if coefs.count(None) == 0 else None

    def __call__(self, f):
        """Return the function that results of applying the operator."""
        def applied_lfd(t):
            return sum(w(t) * f(t, derivative=i)
                       for i, w in enumerate(self.weights))

        return applied_lfd


def _apply_lfd(X, basis, penalty):
    """
    Apply the lfd to a single data type.
    """
    penalty_method = getattr(basis, "penalty", None)

    if penalty_method:
        return penalty_method(penalty)
    else:
        # Multivariate objects have no penalty
        return np.zeros((X.shape[1], X.shape[1]))


def compute_lfd_matrix(X, basis, regularization_parameter,
                       penalty, penalty_matrix):
    """
    Computes the regularization matrix for a linear differential operator.

    X can be a list of mixed data.
    """
    from skfda.representation.basis import Basis

    # If there is no regularization, return 0 and rely on broadcasting
    if regularization_parameter == 0:
        return 0

    # Compute penalty matrix if not provided
    if penalty_matrix is None:

        # Convert the linear differential operator if necessary
        if penalty is None:
            penalty = LinearDifferentialOperator(order=2)
        elif not isinstance(penalty, LinearDifferentialOperator):
            penalty = LinearDifferentialOperator(penalty)

        if isinstance(basis, Basis):
            penalty_matrix = _apply_lfd(X, basis, penalty)
        else:
            # If X and basis are lists

            penalty_blocks = [_apply_lfd(x, b, penalty)
                              for x, b in zip(X, basis)]
            penalty_matrix = scipy.linalg.block_diag(*penalty_blocks)

    return regularization_parameter * penalty_matrix
