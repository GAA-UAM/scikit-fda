"""Elastic metrics."""
from __future__ import annotations

from typing import Any, Optional, Tuple, TypeVar

import numpy as np
import scipy.integrate
from typing_extensions import Final

from ..._utils import normalize_scale, normalize_warping
from ...representation import FData, FDataGrid
from ...typing._numpy import NDArrayFloat
from ..operators import SRSF
from ._lp_distances import l2_distance
from ._utils import PairwiseMetric, _cast_to_grid, pairwise_metric_optimization

T = TypeVar("T", bound=FData)


def _transformation_for_fisher_rao(
    fdata1: T,
    fdata2: T,
    *,
    eval_points: NDArrayFloat | None = None,
    _check: bool = True,
) -> Tuple[FDataGrid, FDataGrid]:
    fdata1, fdata2 = _cast_to_grid(
        fdata1,
        fdata2,
        eval_points=eval_points,
        _check=_check,
    )

    # Both should have the same grid points
    eval_points_normalized = normalize_scale(fdata1.grid_points[0])

    # Calculate the corresponding srsf and normalize to (0,1)
    fdata1 = fdata1.copy(
        grid_points=eval_points_normalized,
        domain_range=(0, 1),
    )
    fdata2 = fdata2.copy(
        grid_points=eval_points_normalized,
        domain_range=(0, 1),
    )

    srsf = SRSF(initial_value=0)
    fdata1_srsf = srsf.fit_transform(fdata1)
    fdata2_srsf = srsf.transform(fdata2)

    return fdata1_srsf, fdata2_srsf


class FisherRaoDistance():
    r"""
    Compute the Fisher-Rao distance between two functional objects.

    Let :math:`f_i` and :math:`f_j` be two functional observations, and let
    :math:`q_i` and :math:`q_j` be the corresponding SRSF
    (see :class:`SRSF`), the Fisher-Rao distance is defined as

    .. math::
        d_{FR}(f_i, f_j) = \| q_i - q_j \|_2 =
        \left ( \int_0^1 sgn(\dot{f_i}(t))\sqrt{|\dot{f_i}(t)|} -
        sgn(\dot{f_j}(t))\sqrt{|\dot{f_j}(t)|} dt \right )^{\frac{1}{2}}

    If the observations are distributions of random variables the distance will
    match with the usual Fisher-Rao distance in non-parametric form for
    probability distributions :footcite:`srivastava++_2011_ficher-rao`.

    If the observations are defined in a :term:`domain` different than (0,1)
    their domains are normalized to this interval with an affine
    transformation.

    Args:
        fdata1: First FData object.
        fdata2: Second FData object.
        eval_points: Array with points of evaluation.

    Returns:
        Fisher rao distance.

    Raises:
        ValueError: If the objects are not unidimensional.

    References:
        .. footbibliography::

    """

    def __call__(
        self,
        fdata1: T,
        fdata2: T,
        *,
        eval_points: NDArrayFloat | None = None,
        _check: bool = True,
    ) -> NDArrayFloat:
        """Compute the distance."""
        # Return the L2 distance of the SRSF
        return l2_distance(*_transformation_for_fisher_rao(
            fdata1,
            fdata2,
            eval_points=eval_points,
            _check=_check,
        ))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}()"
        )


fisher_rao_distance: Final = FisherRaoDistance()


@pairwise_metric_optimization.register
def _pairwise_metric_optimization_fisher_rao(
    metric: FisherRaoDistance,
    elem1: T,
    elem2: Optional[T],
) -> NDArrayFloat:

    new_elem2 = elem1.copy() if elem2 is None else elem2

    new_elem1, new_elem2 = _transformation_for_fisher_rao(
        elem1,
        new_elem2,
    )

    pairwise = PairwiseMetric(l2_distance)

    return pairwise(new_elem1, None if elem2 is None else new_elem2)


def fisher_rao_amplitude_distance(
    fdata1: T,
    fdata2: T,
    *,
    lam: float = 0,
    eval_points: NDArrayFloat | None = None,
    _check: bool = True,
    **kwargs: Any,
) -> NDArrayFloat:
    r"""
    Compute the Fisher-Rao amplitude distance between two functional objects.

    Let :math:`f_i` and :math:`f_j` be two functional observations, and let
    :math:`q_i` and :math:`q_j` be the corresponding SRSF
    (see :class:`SRSF`), the Fisher-Rao amplitude distance is defined as

    .. math::
        d_{A}(f_i, f_j)=min_{\gamma \in \Gamma}d_{FR}(f_i \circ \gamma,f_j)

    A penalty term could be added to restrict the ammount of elasticity in the
    alignment used.

    .. math::
        d_{\lambda}^2(f_i, f_j) =min_{\gamma \in \Gamma} \{
        d_{FR}^2(f_i \circ \gamma, f_j) + \lambda \mathcal{R}(\gamma) \}


    Where :math:`d_{FR}` is the Fisher-Rao distance and the penalty term is
    given by

    .. math::
        \mathcal{R}(\gamma) = \|\sqrt{\dot{\gamma}}- 1 \|_{\mathbb{L}^2}^2

    See the :footcite:`srivastava+klassen_2016_analysis_amplitude` for a
    detailed explanation.

    If the observations are defined in a :term:`domain` different than (0,1)
    their domains are normalized to this interval with an affine
    transformation.

    Args:
        fdata1: First FData object.
        fdata2: Second FData object.
        lam: Penalty term to restric the elasticity.
        eval_points: Array with points of evaluation.
        kwargs: Name arguments to be passed to
            :func:`elastic_registration_warping`.

    Returns:
        Elastic distance.

    Raises:
        ValueError: If the objects are not unidimensional.

    References:
        .. footbibliography::

    """
    from ...preprocessing.registration import FisherRaoElasticRegistration

    fdata1, fdata2 = _cast_to_grid(
        fdata1,
        fdata2,
        eval_points=eval_points,
        _check=_check,
    )

    # Both should have the same grid points
    eval_points_normalized = normalize_scale(fdata1.grid_points[0])

    # Calculate the corresponding srsf and normalize to (0,1)
    fdata1 = fdata1.copy(
        grid_points=eval_points_normalized,
        domain_range=(0, 1),
    )
    fdata2 = fdata2.copy(
        grid_points=eval_points_normalized,
        domain_range=(0, 1),
    )

    elastic_registration = FisherRaoElasticRegistration(
        template=fdata2,
        penalty=lam,
        output_points=eval_points_normalized,
        **kwargs,
    )

    fdata1_reg = elastic_registration.fit_transform(fdata1)

    srsf = SRSF(initial_value=0)
    fdata1_reg_srsf = srsf.fit_transform(fdata1_reg)
    fdata2_srsf = srsf.transform(fdata2)
    distance = l2_distance(fdata1_reg_srsf, fdata2_srsf)

    if lam != 0:
        # L2 norm || sqrt(Dh) - 1 ||^2
        warping_deriv = elastic_registration.warping_.derivative()
        penalty = warping_deriv(eval_points_normalized)[0, ..., 0]
        penalty = np.sqrt(penalty, out=penalty)
        penalty -= 1
        penalty = np.square(penalty, out=penalty)
        penalty = scipy.integrate.simps(penalty, x=eval_points_normalized)

        distance = np.sqrt(distance**2 + lam * penalty)

    return distance


def fisher_rao_phase_distance(
    fdata1: T,
    fdata2: T,
    *,
    lam: float = 0,
    eval_points: NDArrayFloat | None = None,
    _check: bool = True,
) -> NDArrayFloat:
    r"""
    Compute the Fisher-Rao phase distance between two functional objects.

    Let :math:`f_i` and :math:`f_j` be two functional observations, and let
    :math:`\gamma_{ij}` the corresponding warping used in the elastic
    registration to align :math:`f_i` to :math:`f_j` (see
    :func:`elastic_registration`). The Fisher-Rao  phase distance between
    :math:`f_i` and :math:`f_j` is defined as

    .. math::
        d_{P}(f_i, f_j) = d_{FR}(\gamma_{ij}, \gamma_{id}) =
        arcos \left ( \int_0^1 \sqrt {\dot \gamma_{ij}(t)} dt \right )

    where :math:`\gamma_{id}` is the identity warping.

    See :footcite:`srivastava+klassen_2016_analysis_phase` for a detailed
    explanation.

    If the observations are defined in a :term:`domain` different than (0,1)
    their domains are normalized to this interval with an affine
    transformation.

    Args:
        fdata1: First FData object.
        fdata2: Second FData object.
        lam: Penalty term to restric the elasticity.
        eval_points (array_like, optional): Array with points of evaluation.

    Returns:
        Phase distance between the objects.

    Raises:
        ValueError: If the objects are not unidimensional.

    References:
        .. footbibliography::

    """
    from ...preprocessing.registration import FisherRaoElasticRegistration

    fdata1, fdata2 = _cast_to_grid(
        fdata1,
        fdata2,
        eval_points=eval_points,
        _check=_check,
    )

    # Rescale in the interval (0,1)
    eval_points_normalized = normalize_scale(fdata1.grid_points[0])

    # Calculate the corresponding srsf and normalize to (0,1)
    fdata1 = fdata1.copy(
        grid_points=eval_points_normalized,
        domain_range=(0, 1),
    )
    fdata2 = fdata2.copy(
        grid_points=eval_points_normalized,
        domain_range=(0, 1),
    )

    elastic_registration = FisherRaoElasticRegistration(
        penalty=lam,
        template=fdata2,
        output_points=eval_points_normalized,
    )

    elastic_registration.fit_transform(fdata1)

    warping_deriv = elastic_registration.warping_.derivative()
    derivative_warping = warping_deriv(eval_points_normalized)[0, ..., 0]

    derivative_warping = np.sqrt(derivative_warping, out=derivative_warping)

    d = scipy.integrate.simps(derivative_warping, x=eval_points_normalized)
    d = np.clip(d, -1, 1)

    return np.arccos(d)  # type: ignore[no-any-return]


def _fisher_rao_warping_distance(
    warping1: T,
    warping2: T,
    *,
    eval_points: NDArrayFloat | None = None,
    _check: bool = True,
) -> NDArrayFloat:
    r"""
    Compute the Fisher-Rao distance between warpings functions.

    Let :math:`\gamma_i` and :math:`\gamma_j` be two warpings, defined in
    :math:`\gamma_i:[0,1] \rightarrow [0,1]`. The distance in the
    space of warping functions, :math:`\Gamma`, with the Riemannian metric
    given by the Fisher-Rao inner product can be computed using the structure
    of Hilbert sphere in their SRSF's.

    .. math::
        d_{\Gamma}(\gamma_i, \gamma_j) = cos^{-1} \left ( \int_0^1
        \sqrt{\dot \gamma_i(t)\dot \gamma_j(t)}dt \right )

    See :footcite:`srivastava+klassen_2016_analysis_probability` for a detailed
    explanation.

    If the warpings are not defined in [0,1], an affine transformation is made
    to change the :term:`domain`.

    Args:
        warping1: First warping.
        warping2: Second warping.
        eval_points: Array with points of evaluation.

    Returns:
        Distance between warpings:

    Raises:
        ValueError: If the objects are not unidimensional.

    References:
        .. footbibliography::

    """
    warping1, warping2 = _cast_to_grid(
        warping1,
        warping2,
        eval_points=eval_points,
        _check=_check,
    )

    # Normalization of warping to (0,1)x(0,1)
    warping1 = normalize_warping(warping1, (0, 1))
    warping2 = normalize_warping(warping2, (0, 1))

    warping1_data = warping1.derivative().data_matrix[0, ..., 0]
    warping2_data = warping2.derivative().data_matrix[0, ..., 0]

    # Derivative approximations can have negatives, specially in the
    # borders.
    warping1_data[warping1_data < 0] = 0
    warping2_data[warping2_data < 0] = 0

    # In this case the srsf is the sqrt(gamma')
    srsf_warping1 = np.sqrt(warping1_data, out=warping1_data)
    srsf_warping2 = np.sqrt(warping2_data, out=warping2_data)

    product = np.multiply(srsf_warping1, srsf_warping2, out=srsf_warping1)

    d = scipy.integrate.simps(product, x=warping1.grid_points[0])
    d = np.clip(d, -1, 1)

    return np.arccos(d)  # type: ignore[no-any-return]
