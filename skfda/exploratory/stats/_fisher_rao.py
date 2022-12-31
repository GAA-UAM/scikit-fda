from __future__ import annotations

from typing import Any

import numpy as np
import scipy.integrate
from fdasrsf.utility_functions import optimum_reparam

from ..._utils import invert_warping, normalize_scale
from ...misc.operators import SRSF
from ...misc.validation import check_fdata_dimensions
from ...representation import FDataGrid
from ...representation.interpolation import SplineInterpolation
from ...typing._numpy import NDArrayFloat

###############################################################################
# Based on the original implementation of J. Derek Tucker in                  #
# *fdasrsf_python* (https://github.com/jdtuck/fdasrsf_python)                 #
# and *ElasticFDA.jl* (https://github.com/jdtuck/ElasticFDA.jl).              #
###############################################################################


def _elastic_alignment_array(
    template_data: NDArrayFloat,
    q_data: NDArrayFloat,
    eval_points: NDArrayFloat,
    penalty: float,
    grid_dim: int,
) -> NDArrayFloat:
    """
    Wrap the :func:`optimum_reparam` function of fdasrsf.

    Selects the corresponding routine depending on the dimensions of the
    arrays.

    Args:
        template_data: Array with the srsf of the template.
        q_data: Array with the srsf of the curves
                to be aligned.
        eval_points: Discretisation points of the functions.
        penalty: Penalisation term.
        grid_dim: Dimension of the grid used in the alignment algorithm.

    Returns:
        Array with the same shape than q_data with the srsf of
        the functions aligned to the template(s).

    """
    return optimum_reparam(  # type: ignore[no-any-return]
        np.ascontiguousarray(template_data.T),
        np.ascontiguousarray(eval_points),
        np.ascontiguousarray(q_data.T),
        method="DP2",
        lam=penalty,
        grid_dim=grid_dim,
    ).T


def _fisher_rao_warping_mean(
    warping: FDataGrid,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    step_size: float = 0.3,
) -> FDataGrid:
    r"""
    Compute the karcher mean of a set of warpings.

    Let :math:`\gamma_i i=1...n` be a set of warping functions
    :math:`\gamma_i:[a,b] \rightarrow [a,b]` in :math:`\Gamma`, i.e.,
    monotone increasing and with the restriction :math:`\gamma_i(a)=a \,
    \gamma_i(b)=b`.

    The karcher mean :math:`\bar \gamma` is defined as the warping that
    minimises locally the sum of Fisher-Rao squared distances
    :footcite:`srivastava+klassen_2016_analysis_orbit`.

    .. math::
        \bar \gamma = argmin_{\gamma \in \Gamma} \sum_{i=1}^{n}
         d_{FR}^2(\gamma, \gamma_i)

    The computation is performed using the structure of Hilbert Sphere obtained
    after a transformation of the warpings, see
    :footcite:`srivastava++_2011_ficher-rao_orbit`.

    Args:
        warping: Set of warpings.
        max_iter: Maximum number of interations. Defaults to 100.
        tol: Convergence criterion, if the norm of the mean of the
            shooting vectors, :math:`| \bar v |<tol`, the algorithm will stop.
        step_size: Step size :math:`\epsilon` used to update the mean.

    Returns:
        Fdatagrid with the mean of the warpings. If
        shooting is True the shooting vectors will be returned in a tuple with
        the mean.

    References:
        .. footbibliography::

    """
    eval_points = warping.grid_points[0]
    original_eval_points = eval_points

    # Rescale warping to (0, 1)
    if warping.grid_points[0][0] != 0 or warping.grid_points[0][-1] != 1:

        eval_points = normalize_scale(eval_points)
        warping = FDataGrid(
            normalize_scale(warping.data_matrix[..., 0]),
            normalize_scale(warping.grid_points[0]),
        )

    # Compute srsf of warpings and their mean
    srsf = SRSF(output_points=eval_points, initial_value=0)
    psi = srsf.fit_transform(warping)

    # Find psi closest to the mean
    psi_centered = psi - srsf.fit_transform(warping.mean())
    psi_centered_data = psi_centered.data_matrix[..., 0]
    np.square(psi_centered_data, out=psi_centered_data)
    d = psi_centered_data.sum(axis=1).argmin()

    # Get raw values to calculate
    mu = np.atleast_2d(psi[d].data_matrix[0, ..., 0])
    psi_data = psi.data_matrix[..., 0]
    vmean = np.empty((1, len(eval_points)))

    # Construction of shooting vectors
    for _ in range(max_iter):

        vmean[0] = 0
        # Compute shooting vectors
        for psi_i in psi_data:

            inner = scipy.integrate.simps(mu * psi_i, x=eval_points)
            inner = max(min(inner, 1), -1)

            theta = np.arccos(inner)

            if theta > 1e-10:
                vmean += theta / np.sin(theta) * (psi_i - np.cos(theta) * mu)

        # Mean of shooting vectors
        vmean /= warping.n_samples
        v_norm = np.sqrt(scipy.integrate.simps(np.square(vmean)))

        # Convergence criterion
        if v_norm < tol:
            break

        # Calculate exponential map of mu
        a = np.cos(step_size * v_norm)
        b = np.sin(step_size * v_norm) / v_norm
        mu = a * mu + b * vmean

    # Recover mean in original gamma space
    warping_mean_ret = scipy.integrate.cumtrapz(
        np.square(mu, out=mu)[0],
        x=eval_points,
        initial=0,
    )

    # Affine traslation to original scale
    warping_mean_ret = normalize_scale(
        warping_mean_ret,
        a=original_eval_points[0],
        b=original_eval_points[-1],
    )

    monotone_interpolation = SplineInterpolation(
        interpolation_order=3,
        monotone=True,
    )

    return FDataGrid(
        [warping_mean_ret],
        grid_points=original_eval_points,
        interpolation=monotone_interpolation,
    )


def fisher_rao_karcher_mean(
    fdatagrid: FDataGrid,
    *,
    penalty: float = 0,
    center: bool = True,
    max_iter: int = 20,
    tol: float = 1e-3,
    initial: float | None = None,
    grid_dim: int = 7,
    **kwargs: Any,
) -> FDataGrid:
    r"""
    Compute the Karcher mean under the elastic metric.

    Calculates the Karcher mean of a set of functional samples in the amplitude
    space :math:`\mathcal{A}=\mathcal{F}/\Gamma`.

    Let :math:`q_i` the corresponding SRSF of the observation :math:`f_i`.
    The space :math:`\mathcal{A}` is defined using the equivalence classes
    :math:`[q_i]=\{ q_i \circ \gamma \| \gamma \in \Gamma \}`, where
    :math:`\Gamma` denotes the space of warping functions. The karcher mean
    in this space is defined as

    .. math::
        [\mu_q] = argmin_{[q] \in \mathcal{A}} \sum_{i=1}^n
        d_{\lambda}^2([q],[q_i])

    Once :math:`[\mu_q]` is obtained it is selected the element of the
    equivalence class which makes the mean of the warpings employed be the
    identity.

    See :footcite:`srivastava+klassen_2016_analysis_karcher` and
    :footcite:`srivastava++_2011_ficher-rao_karcher`.

    Args:
        fdatagrid: Set of functions to compute the
            mean.
        penalty: Penalisation term. Defaults to 0.
        center: If ``True`` it is computed the mean of the warpings and
            used to select a central mean. Defaults ``True``.
        max_iter: Maximum number of iterations. Defaults to 20.
        tol: Convergence criterion, the algorithm will stop if
            :math:`|mu_{(\nu)} - mu_{(\nu - 1)}|_2 / | mu_{(\nu-1)} |_2 < tol`.
        initial: Value of the mean at the starting point. By default
            takes the average of the initial points of the samples.
        grid_dim: Dimension of the grid used in the alignment
            algorithm. Defaults 7.
        kwargs: Named options to be pased to :func:`_fisher_rao_warping_mean`.

    Returns:
        FDatagrid with the mean of the functions.

    Raises:
        ValueError: If the object is multidimensional or the shape of the srsf
            do not match with the fdatagrid.

    References:
        .. footbibliography::

    """
    check_fdata_dimensions(
        fdatagrid,
        dim_domain=1,
        dim_codomain=1,
    )

    srsf_transformer = SRSF(initial_value=0)
    fdatagrid_srsf = srsf_transformer.fit_transform(fdatagrid)
    eval_points = fdatagrid.grid_points[0]

    eval_points_normalized = normalize_scale(eval_points)
    y_scale = eval_points[-1] - eval_points[0]

    interpolation = SplineInterpolation(interpolation_order=3, monotone=True)

    # Discretisation points
    fdatagrid_normalized = FDataGrid(
        fdatagrid(eval_points) / y_scale,
        grid_points=eval_points_normalized,
    )

    srsf = fdatagrid_srsf(eval_points)[..., 0]

    # Initialize with function closest to the L2 mean with the L2 distance
    centered = (srsf.T - srsf.mean(axis=0, keepdims=True).T).T

    distances = scipy.integrate.simps(
        np.square(centered, out=centered),
        eval_points_normalized,
        axis=1,
    )

    # Initialization of iteration
    mu = srsf[np.argmin(distances)]
    mu_aux = np.empty(mu.shape)
    mu_1 = np.empty(mu.shape)

    # Main iteration
    for _ in range(max_iter):

        gammas_matrix = _elastic_alignment_array(
            mu,
            srsf,
            eval_points_normalized,
            penalty,
            grid_dim,
        )

        gammas = FDataGrid(
            gammas_matrix,
            grid_points=eval_points_normalized,
            interpolation=interpolation,
        )

        fdatagrid_normalized = fdatagrid_normalized.compose(gammas)
        srsf = srsf_transformer.transform(
            fdatagrid_normalized,
        ).data_matrix[..., 0]

        # Next iteration
        mu_1 = srsf.mean(axis=0, out=mu_1)

        # Convergence criterion
        mu_norm = np.sqrt(
            scipy.integrate.simps(
                np.square(mu, out=mu_aux),
                eval_points_normalized,
            ),
        )

        mu_diff = np.sqrt(
            scipy.integrate.simps(
                np.square(mu - mu_1, out=mu_aux),
                eval_points_normalized,
            ),
        )

        if mu_diff / mu_norm < tol:
            break

        mu = mu_1

    if initial is None:
        initial = fdatagrid.data_matrix[:, 0].mean()

    srsf_transformer.set_params(initial_value=initial)

    # Karcher mean orbit in space L2/Gamma
    karcher_mean = srsf_transformer.inverse_transform(
        fdatagrid.copy(
            data_matrix=[mu],
            grid_points=eval_points,
            sample_names=("Karcher mean",),
        ),
    )

    if center:
        # Gamma mean in Hilbert Sphere
        mean_normalized = _fisher_rao_warping_mean(gammas, **kwargs)

        gamma_mean = FDataGrid(
            normalize_scale(
                mean_normalized.data_matrix[..., 0],
                a=eval_points[0],
                b=eval_points[-1],
            ),
            grid_points=eval_points,
        )

        gamma_inverse = invert_warping(gamma_mean)

        karcher_mean = karcher_mean.compose(gamma_inverse)

    # Return center of the orbit
    return karcher_mean
