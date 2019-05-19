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

    """

    def __init__(self, order=None, weights=None, domain_range=(0, 1)):
        """Lfd Constructor. You have to provide one of the two first
           parameters. It both are provided, it will raise an error

        Args:
            order (int, optional): the order of the operator. It's the highest
                    derivative order of the operator

            weights (list, optional): A FDataBasis objects list of length
                    order + 1 items

            domain_range (tuple or list of tuples, optional): Definition
                         of the interval where the weight functions are
                         defined. Defaults to (0,1).
        """

        from ..representation.basis import (FDataBasis, Constant,
                                            _same_domain)

        if order is not None and weights is not None:
            raise ValueError("You have to provide the order or the weights, "
                             "not both")

        self.domain_range = domain_range

        if order is None and weights is None:
            self.order = 0
            self.weights = []

        elif weights is None:
            if order < 0:
                raise ValueError("Order should be an non-negative integer")

            self.order = order
            self.weights = [
                FDataBasis(Constant(domain_range), 0 if (i < order) else 1) for
                i in range(order + 1)]

        else:
            if len(weights) is 0:
                raise ValueError("You have to provide one weight at least")

            if all(isinstance(n, int) for n in weights):
                self.order = len(weights) - 1
                self.weights = (FDataBasis(Constant(domain_range),
                                           np.array(weights)
                                           .reshape(-1, 1)).to_list())

            elif all(isinstance(n, FDataBasis) for n in weights):
                if all([_same_domain(weights[0].domain_range,
                                     x.domain_range) and x.nsamples == 1 for x
                        in weights]):
                    self.order = len(weights) - 1
                    self.weights = weights
                    self.domain_range = weights[0].domain_range

                else:
                    raise ValueError("FDataBasis objects in the list has "
                                     "not the same domain_range")

            else:
                raise ValueError("The elements of the list are neither "
                                 "integers or FDataBasis objects")

    def __repr__(self):
        """Representation of Lfd object."""

        bwtliststr = ""
        for i in range(self.order):
            bwtliststr = bwtliststr + "\n" + self.weights[i].__repr__() + ","

        return (f"{self.__class__.__name__}("
                f"\nnderiv={self.order},"
                f"\nbwtlist=[{bwtliststr[:-1]}]"
                f"\n)").replace('\n', '\n    ')

    def __eq__(self, other):
        """Equality of Lfd objects"""
        return (self.order == other.nderic
                and all(self.weights[i] == other.bwtlist[i]
                        for i in range(self.order)))
