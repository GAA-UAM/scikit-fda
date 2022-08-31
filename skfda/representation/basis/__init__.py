"""Basis representation."""
from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        '_basis': ["Basis"],
        "_bspline": ["BSpline"],
        "_constant": ["Constant"],
        "_fdatabasis": ["FDataBasis", "FDataBasisDType"],
        "_finite_element": ["FiniteElement"],
        "_fourier": ["Fourier"],
        "_monomial": ["Monomial"],
        "_tensor_basis": ["Tensor"],
        "_vector_basis": ["VectorValued"],
    },
)

if TYPE_CHECKING:
    from ._basis import Basis as Basis
    from ._bspline import BSpline as BSpline
    from ._constant import Constant as Constant
    from ._fdatabasis import (
        FDataBasis as FDataBasis,
        FDataBasisDType as FDataBasisDType,
    )
    from ._finite_element import FiniteElement as FiniteElement
    from ._fourier import Fourier as Fourier
    from ._monomial import Monomial as Monomial
    from ._tensor_basis import Tensor as Tensor
    from ._vector_basis import VectorValued as VectorValued
