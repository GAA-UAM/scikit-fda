"""Basis representation."""
from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        '_basis': ["Basis"],
        "_bspline_basis": ["BSplineBasis"],
        "_constant_basis": ["ConstantBasis"],
        "_fdatabasis": ["FDataBasis", "FDataBasisDType"],
        "_finite_element_basis": ["FiniteElementBasis"],
        "_fourier_basis": ["FourierBasis"],
        "_monomial_basis": ["MonomialBasis"],
        "_tensor_basis": ["TensorBasis"],
        "_vector_basis": ["VectorValuedBasis"],
    },
)

if TYPE_CHECKING:
    from ._basis import Basis as Basis
    from ._bspline_basis import BSplineBasis as BSplineBasis
    from ._constant_basis import ConstantBasis as ConstantBasis
    from ._fdatabasis import (
        FDataBasis as FDataBasis,
        FDataBasisDType as FDataBasisDType,
    )
    from ._finite_element_basis import FiniteElementBasis as FiniteElementBasis
    from ._fourier_basis import FourierBasis as FourierBasis
    from ._monomial_basis import MonomialBasis as MonomialBasis
    from ._tensor_basis import TensorBasis as TensorBasis
    from ._vector_basis import VectorValuedBasis as VectorValuedBasis
