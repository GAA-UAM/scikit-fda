import numbers

import numpy as np


def _transform_to_2d(t):
    """Transform 1d arrays in column vectors."""
    t = np.asarray(t)

    dim = len(t.shape)
    assert dim <= 2

    if dim < 2:
        t = np.atleast_2d(t).T

    return t


def _execute_covariance(covariance, x, y):
    """Execute a covariance function.
    """
    x = _transform_to_2d(x)
    y = _transform_to_2d(y)

    if isinstance(covariance, numbers.Number):
        return covariance
    else:
        if callable(covariance):
            result = covariance(x, y)
        else:
            # GPy kernel
            result = covariance.K(x, y)

        assert result.shape[0] == len(x)
        assert result.shape[1] == len(y)
        return result


class Brownian():
    """Brownian covariance"""

    def __init__(self, *, variance: float=1., origin: float=0.):
        self.variance = variance
        self.origin = origin

    def __call__(self, x, y):
        """Brownian covariance function"""
        x = np.asarray(x) - self.origin
        y = np.asarray(y) - self.origin

        return self.variance * (np.abs(x) + np.abs(y.T) - np.abs(x - y.T)) / 2

    def _repr_latex_(self):
        return (r"\[K(x, y) = \sigma^2 \frac{|x - \mathcal{O}| + "
                r"|y - \mathcal{O}| - |x-y|}{2}\]"
                "where:"
                r"\begin{align*}"
                fr"\qquad\sigma^2 &= {self.variance} \\"
                fr"\mathcal{{O}} &= {self.origin} \\"
                r"\end{align*}")
