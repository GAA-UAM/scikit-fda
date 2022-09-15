"""Validation utilities."""

from __future__ import annotations

import functools
import numbers
from typing import Container, Sequence, Tuple, cast

import numpy as np
from sklearn.utils import check_random_state as _check_random_state

from ..representation import FData, FDataBasis, FDataGrid
from ..typing._base import (
    DomainRange,
    DomainRangeLike,
    EvaluationPoints,
    RandomState,
    RandomStateLike,
)
from ..typing._numpy import ArrayLike


def check_fdata_dimensions(
    fd: FData,
    *,
    dim_domain: int | Container[int] | None = None,
    dim_codomain: int | Container[int] | None = None,
) -> None:
    """
    Check that a functional data object have appropriate dimensions.

    Args:
        fd: Functional data object to check.
        dim_domain: Allowed dimension(s) of the :term:`domain`.
        dim_codomain: Allowed dimension(s) of the :term:`codomain`.

    Raises:
        ValueError: If the data has not the requested dimensions.

    """
    if isinstance(dim_domain, int):
        dim_domain = {dim_domain}

    if isinstance(dim_codomain, int):
        dim_codomain = {dim_codomain}

    if dim_domain is not None:

        if fd.dim_domain not in dim_domain:
            raise ValueError(
                "Invalid domain dimension for functional data object:"
                f"{fd.dim_domain} not in {dim_domain}.",
            )

    if dim_codomain is not None:

        if fd.dim_codomain not in dim_codomain:
            raise ValueError(
                "Invalid domain dimension for functional data object:"
                f"{fd.dim_codomain} not in {dim_codomain}.",
            )


def check_fdata_same_dimensions(
    fdata1: FData,
    fdata2: FData,
) -> None:
    """
    Check that two FData have the same dimensions.

    Args:
        fdata1: First functional data object.
        fdata2: Second functional data object.

    Raises:
        ValueError: If the dimensions don't agree.

    """
    if (fdata1.dim_domain != fdata2.dim_domain):
        raise ValueError(
            f"Functional data has incompatible domain dimensions: "
            f"{fdata1.dim_domain} != {fdata2.dim_domain}",
        )

    if (fdata1.dim_codomain != fdata2.dim_codomain):
        raise ValueError(
            f"Functional data has incompatible codomain dimensions: "
            f"{fdata1.dim_codomain} != {fdata2.dim_codomain}",
        )

    if (fdata1.domain_range != fdata2.domain_range):
        raise ValueError(
            f"Functional data has incompatible domain range: "
            f"{fdata1.domain_range} != {fdata2.domain_range}",
        )


@functools.singledispatch
def _check_fdata_same_kind_specific(
    fdata1: FData,
    fdata2: FData,
) -> None:
    """Specific comparisons for subclasses."""


@_check_fdata_same_kind_specific.register
def _check_fdatagrid_same_kind_specific(
    fdata1: FDataGrid,
    fdata2: FData,
) -> None:

    # First we do an identity comparison to speed up the common case
    if fdata1.grid_points is not fdata2.grid_points:
        if not all(
            np.array_equal(g1, g2)
            for g1, g2 in zip(fdata1.grid_points, fdata2.grid_points)
        ):
            raise ValueError(
                f"Incompatible grid points between functional data objects:"
                f"{fdata1.grid_points} != {fdata2.grid_points}",
            )


@_check_fdata_same_kind_specific.register
def _check_fdatabasis_same_kind_specific(
    fdata1: FDataBasis,
    fdata2: FData,
) -> None:

    if fdata1.basis != fdata2.basis:
        raise ValueError(
            f"Incompatible basis between functional data objects:"
            f"{fdata1.basis} != {fdata2.basis}",
        )


def check_fdata_same_kind(
    fdata1: FData,
    fdata2: FData,
) -> None:
    """
    Check that two functional objects are of the same kind.

    This compares everything to ensure that the data is compatible: dimensions,
    grids, basis, when applicable. Every parameter that must be fixed for all
    samples should be the same.

    In other words: it should be possible to concatenate the two functional
    objects together.

    Args:
        fdata1: First functional data object.
        fdata2: Second functional data object.

    Raises:
        ValueError: If some attributes don't agree.

    """
    check_fdata_same_dimensions(fdata1, fdata2)

    # If the second is a subclass, reverse the order
    if isinstance(fdata2, type(fdata1)):
        fdata1, fdata2 = fdata2, fdata1

    _check_fdata_same_kind_specific(fdata1, fdata2)

    # If there is no subclassing, execute both checks
    if not isinstance(fdata1, type(fdata2)):
        _check_fdata_same_kind_specific(fdata2, fdata1)


def _valid_eval_points_shape(
    shape: Sequence[int],
    *,
    dim_domain: int,
) -> bool:
    """Check that the shape for aligned evaluation points is ok."""
    return (
        (len(shape) == 2 and shape[-1] == dim_domain)  # noqa: WPS222
        or (len(shape) <= 1 and dim_domain == 1)  # Domain ommited
        or (len(shape) == 1 and shape == (dim_domain,))  # Num. points ommited
    )


def validate_evaluation_points(
    eval_points: ArrayLike,
    *,
    aligned: bool,
    n_samples: int,
    dim_domain: int,
) -> EvaluationPoints:
    """Convert and reshape the eval_points to ndarray.

    Args:
        eval_points: Evaluation points to be reshaped.
        aligned: Boolean flag. True if all the samples
            will be evaluated at the same evaluation_points.
        n_samples: Number of observations.
        dim_domain: Dimension of the domain.

    Returns:
        Numpy array with the eval_points, if
        evaluation_aligned is True with shape `number of evaluation points`
        x `dim_domain`. If the points are not aligned the shape of the
        points will be `n_samples` x `number of evaluation points`
        x `dim_domain`.

    """
    eval_points = np.asarray(eval_points)

    shape: Sequence[int]
    if aligned:
        if _valid_eval_points_shape(eval_points.shape, dim_domain=dim_domain):
            shape = (-1, dim_domain)
        else:
            raise ValueError(
                "Invalid shape for evaluation points."
                f"An array with size (n_points, dim_domain (={dim_domain})) "
                "was expected (both can be ommited if they are 1)."
                "Instead, the received evaluation points have shape "
                f"{eval_points.shape}.",
            )
    else:
        if eval_points.shape[0] == n_samples and _valid_eval_points_shape(
            eval_points.shape[1:],
            dim_domain=dim_domain,
        ):
            shape = (n_samples, -1, dim_domain)
        else:
            raise ValueError(
                "Invalid shape for unaligned evaluation points."
                f"An array with size (n_samples (={n_samples}), "
                f"n_points, dim_domain (={dim_domain})) "
                "was expected (the last two can be ommited if they are 1)."
                "Instead, the received evaluation points have shape "
                f"{eval_points.shape}.",
            )

    return eval_points.reshape(shape)


def _validate_domain_range_limits(
    limits: Sequence[float],
) -> Tuple[float, float]:
    if len(limits) != 2 or limits[0] > limits[1]:
        raise ValueError(
            f"Invalid domain interval {limits}. "
            "Domain intervals should have 2 bounds for "
            "dimension: (lower, upper).",
        )

    lower, upper = limits
    return (float(lower), float(upper))


def validate_domain_range(domain_range: DomainRangeLike) -> DomainRange:
    """Convert sequence to a proper domain range."""
    if isinstance(domain_range[0], numbers.Real):
        domain_range = cast(Sequence[float], domain_range)
        domain_range = (domain_range,)

    domain_range = cast(Sequence[Sequence[float]], domain_range)

    return tuple(_validate_domain_range_limits(s) for s in domain_range)


def validate_random_state(random_state: RandomStateLike) -> RandomState:
    """Validate random state or seed."""
    if isinstance(random_state, np.random.Generator):
        return random_state

    return _check_random_state(random_state)  # type: ignore[no-any-return]
