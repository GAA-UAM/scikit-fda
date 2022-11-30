"""Basis representation."""
from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        '_basis': ["Basis"],
        "_bspline_basis": ["BSplineBasis", "BSpline"],
        "_constant_basis": ["ConstantBasis", "Constant"],
        '_custom_basis': ["CustomBasis"],
        "_fdatabasis": ["FDataBasis", "FDataBasisDType"],
        "_finite_element_basis": ["FiniteElementBasis", "FiniteElement"],
        "_fourier_basis": ["FourierBasis", "Fourier"],
        "_monomial_basis": ["MonomialBasis", "Monomial"],
        "_tensor_basis": ["TensorBasis", "Tensor"],
        "_vector_basis": ["VectorValuedBasis", "VectorValued"],
    },
)

if TYPE_CHECKING:
    from ._basis import Basis as Basis
    from ._bspline_basis import (
        BSpline as BSpline,
        BSplineBasis as BSplineBasis,
    )
    from ._constant_basis import (
        Constant as Constant,
        ConstantBasis as ConstantBasis,
    )
    from ._custom_basis import CustomBasis as CustomBasis
    from ._fdatabasis import (
        FDataBasis as FDataBasis,
        FDataBasisDType as FDataBasisDType,
    )
    from ._finite_element_basis import (
        FiniteElement as FiniteElement,
        FiniteElementBasis as FiniteElementBasis,
    )
    from ._fourier_basis import (
        Fourier as Fourier,
        FourierBasis as FourierBasis,
    )
    from ._monomial_basis import (
        Monomial as Monomial,
        MonomialBasis as MonomialBasis,
    )
    from ._tensor_basis import Tensor as Tensor, TensorBasis as TensorBasis
    from ._vector_basis import (
        VectorValued as VectorValued,
        VectorValuedBasis as VectorValuedBasis,
    )
