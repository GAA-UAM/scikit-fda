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
        nderiv=2,
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
        nderiv=2,
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
        nderiv=2,
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

    def __init__(self, order=None, *, weights=None, domain_range=None):
        """Lfd Constructor. You have to provide one of the two first
           parameters. It both are provided, it will raise an error

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

        if order is not None and weights is not None:
            raise ValueError("You have to provide the order or the weights, "
                             "not both")

        real_domain_range = (domain_range if domain_range is not None
                             else (0, 1))

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

            if all(isinstance(n, int) for n in weights):
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

    @property
    def order(self):
        return len(self.weights) - 1

    def __repr__(self):
        """Representation of Lfd object."""

        bwtliststr = ""
        for i in range(self.order + 1):
            bwtliststr = bwtliststr + "\n" + self.weights[i].__repr__() + ","

        return (f"{self.__class__.__name__}("
                f"\nnderiv={self.order},"
                f"\nweights=[{bwtliststr[:-1]}]"
                f"\n)").replace('\n', '\n    ')

    def __eq__(self, other):
        """Equality of Lfd objects"""
        return (self.order == other.nderic and
                all(self.weights[i] == other.bwtlist[i]
                    for i in range(self.order)))
