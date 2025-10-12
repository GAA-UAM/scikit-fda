# -*- coding: utf-8 -*-
"""To basis converter.

This module contains the abstract base class for all FData to FDatabasis
converters.

"""
from __future__ import annotations

from typing import TypeVar

from ..._utils._sklearn_adapter import TransformerMixin
from ...representation import FData, FDataBasis
from ...representation.basis import Basis

Input = TypeVar(
    "Input",
    bound=FData,
    contravariant=True,
)


class _ToBasisConverter(
    TransformerMixin[Input, FDataBasis, object],
):
    """To basis converter.

    Abstract base class for all FData to FDataBasis converters. The subclasses
    must override ``fit`` and ``transform`` to define the conversion.
    """
    basis: Basis

    def __init__(self, basis: Basis) -> None:
        self.basis = basis
        super().__init__()
