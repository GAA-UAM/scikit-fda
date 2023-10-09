"""Functions for working with arrays of functions."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar

from ...misc.validation import validate_domain_range
from .. import nquad_vec

if TYPE_CHECKING:
    from ...typing._numpy import NDArrayFloat
    from ...representation import FData
    from ...typing._base import DomainRangeLike


UfuncMethod = Literal[
    "__call__",
    "reduce",
    "reduceat",
    "accumulate",
    "outer",
    "inner",
]


class _SupportsArrayUFunc(Protocol):
    def __array_ufunc__(
        self,
        ufunc: Any,
        method: UfuncMethod,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        pass


T = TypeVar("T", bound=_SupportsArrayUFunc)


class _UnaryUfunc(Protocol):

    def __call__(self, __arg: T) -> T:  # noqa: WPS112
        pass


def _average_function_ufunc(
    data: FData,
    ufunc: _UnaryUfunc,
    *,
    domain: DomainRangeLike | None = None,
) -> NDArrayFloat:

    if domain is None:
        domain = data.domain_range
    else:
        domain = validate_domain_range(domain)

    lebesgue_measure = math.prod(
        (
            (iterval[1] - iterval[0])
            for iterval in domain
        ),
    )

    try:
        data_eval = ufunc(data)
    except TypeError:

        def integrand(*args: NDArrayFloat) -> NDArrayFloat:  # noqa: WPS430
            f1 = data(args)[:, 0, :]
            return ufunc(f1)

        return nquad_vec(
            integrand,
            domain,
        ) / lebesgue_measure

    else:
        return data_eval.integrate(domain=domain) / lebesgue_measure


def average_function_value(
    data: FData,
    *,
    domain: DomainRangeLike | None = None,
) -> NDArrayFloat:
    r"""
    Calculate the average function value for each function.

    This is the value that, if integrated over the whole domain of each
    function, has the same integral as the function itself.

    .. math::
        \bar{x} = \frac{1}{\text{Vol}(\mathcal{T})}\int_{\mathcal{T}} x(t) dt

    Args:
        data: Functions where we want to calculate the expected value.
        domain: Integration domain. By default, the whole domain is used.

    Returns:
        ndarray of shape (n_dimensions, n_samples) with the values of the
        expectations.

    See also:
        `Entry on Wikipedia
        <https://en.wikipedia.org/wiki/Mean_of_a_function>`_

    """
    return _average_function_ufunc(data, ufunc=lambda x: x, domain=domain)
