import numpy as np

from fda.basis import FDataBasis, Constant, _same_domain

__author__ = "Pablo Pérez Manso"
__email__ = "92manso@gmail.com"


class Lfd:
    """Defines the structure of a linear differential operator function system

    .. math::
        Lx(t) = b_0(t) x(t) + b_1(t) x'(x) + \dots + b_{n-1}(t) d^{n-1}(x(t)) + b_n(t) d^n(x(t))

    Attributes:
        nderiv (int): the order of the operator. It's the n coefficient in the equation above.

        bwtlist (list):  A FDataBasis objects list of length either nderiv

    """

    def __init__(self, param=0, domain_range=(0, 1)):
        """Lfd Constructor

        Args:
            param (int, list, FDataBasis optional): If the given parameter is an integer, it'll be the order of
                the highest derivative.
                  If its a list, of integers or FDataBasis objects, it will be taken as the weight functions
                  for the derivatives.
                  If its a FDataBasis object, each sample will be a weight function and nderiv will be the number
                  of samples

            domain_range (tuple or list of tuples, optional): Definition of the
                interval where the weight functions are defined. Defaults to (0,1).
        """

        if isinstance(param, list):
            if all(isinstance(n, int) for n in param):
                self.nderiv = len(param)
                self.bwtlist = (FDataBasis(Constant(domain_range), np.array(param).reshape(-1, 1)).to_list()
                                if self.nderiv != 0 else [])
            elif all(isinstance(n, FDataBasis) for n in param):
                if all([_same_domain(param[0].domain_range, x.domain_range) for x in param]):
                    self.nderiv = len(param)
                    self.bwtlist = param
                else:
                    raise ValueError("FDataBasis objects in the list has not the same domain_range")
            else:
                raise ValueError("The elements of the list are neither integers or FDataBasis objects")

        elif isinstance(param, FDataBasis):
            self.nderiv = param.nsamples
            self.bwtlist = param.to_list()

        elif isinstance(param, int):
            if param < 0:
                raise ValueError("The deriv order must be non-negative.")

            self.nderiv = param
            self.bwtlist = [FDataBasis(Constant(domain_range), 0) for _ in range(param)] if self.nderiv != 0 else []

        else:
            raise ValueError("Argument is neither a non-negative integer or list")

    def __repr__(self):
        """Representation of Lfd object."""

        bwtliststr = ""
        for i in range(self.nderiv):
            bwtliststr = bwtliststr + "\n" + self.bwtlist[i].__repr__() + ","

        return (f"{self.__class__.__name__}("
                f"\nnderiv={self.nderiv},"
                f"\nbwtlist=[{bwtliststr[:-1]}]"
                f"\n)").replace('\n', '\n    ')

    def __str__(self):
        """Return str(self)."""

        return self.__repr__()

    def __eq__(self, other):
        """Equality of Lfd objects"""
        return (self.nderiv == other.nderic
                and all(self.bwtlist[i] == other.bwtlist[i] for i in range(self.nderiv)))

    @property
    def domain_range(self):
        """Definition range."""

        return None if self.nderiv == 0 else self.bwtlist[0].domain_range
