# -*- coding: utf-8 -*-
"""To basis converter.

This module contains the abstract base class for all FData to FDatabasis
converters.

"""
from __future__ import annotations

from typing import (
    Generic,
    TypeVar,
)

from ..._utils._sklearn_adapter import BaseEstimator, TransformerMixin
from ...representation import FData, FDataBasis

Input = TypeVar(
    "Input",
    bound=FData,
    contravariant=True,
)


class _ToBasisConverter(
    BaseEstimator,
    Generic[Input],
    TransformerMixin[Input, FDataBasis, object],
):
    """To basis converter.

    Abstract base class for all FData to FDataBasis converters. The subclasses
    must override ``fit`` and ``transform`` to define the conversion.
    """
