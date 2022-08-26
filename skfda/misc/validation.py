"""Validation utilities."""

from __future__ import annotations

import functools
from typing import Container

import numpy as np

from ..representation import FData, FDataBasis, FDataGrid


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
