"""Basis representation."""
from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        '_basis': ["Basis"],
        "_bspline_basis": ["BSplineBasis"],
        "_bspline": ["BSpline"],
        "_constant_basis": ["ConstantBasis"],
        "_constant": ["Constant"],
        "_fdatabasis": ["FDataBasis", "FDataBasisDType"],
        "_finite_element_basis": ["FiniteElementBasis"],
        "_finite_element": ["FiniteElement"],
        "_fourier_basis": ["FourierBasis"],
        "_fourier": ["Fourier"],
        "_monomial_basis": ["MonomialBasis"],
        "_monomial": ["Monomial"],
        "_tensor_basis": ["TensorBasis"],
        "_tensor": ["Tensor"],
        "_vector_basis": ["VectorValuedBasis"],
        "_vector": ["VectorValued"],
    },
)

if TYPE_CHECKING:
    from ._basis import Basis as Basis
    from ._bspline import BSpline as BSpline
    from ._bspline_basis import BSplineBasis as BSplineBasis
    from ._constant import Constant as Constant
    from ._constant_basis import ConstantBasis as ConstantBasis
    from ._fdatabasis import (
        FDataBasis as FDataBasis,
        FDataBasisDType as FDataBasisDType,
    )
    from ._finite_element import FiniteElement as FiniteElement
    from ._finite_element_basis import FiniteElementBasis as FiniteElementBasis
    from ._fourier import Fourier as Fourier
    from ._fourier_basis import FourierBasis as FourierBasis
    from ._monomial import Monomial as Monomial
    from ._monomial_basis import MonomialBasis as MonomialBasis
    from ._tensor import Tensor as Tensor
    from ._tensor_basis import TensorBasis as TensorBasis
    from ._vector import VectorValued as VectorValued
    from ._vector_basis import VectorValuedBasis as VectorValuedBasis
