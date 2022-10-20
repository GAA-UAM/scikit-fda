from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.integrate
import scipy.stats
from numpy import linalg as la
from sklearn.covariance import MinCovDet

from ..._utils._sklearn_adapter import BaseEstimator, OutlierMixin
from ...misc.validation import validate_random_state
from ...representation import FDataGrid
from ...typing._base import RandomStateLike
from ...typing._numpy import NDArrayFloat, NDArrayInt
from ..depth.multivariate import Depth, ProjectionDepth
from . import _directional_outlyingness_experiment_results as experiments


@dataclass
class DirectionalOutlyingnessStats:
    """Directional outlyingness statistical measures."""

    directional_outlyingness: NDArrayFloat
    functional_directional_outlyingness: NDArrayFloat
    mean_directional_outlyingness: NDArrayFloat
    variation_directional_outlyingness: NDArrayFloat


def directional_outlyingness_stats(  # noqa: WPS218
    fdatagrid: FDataGrid,
    *,
    multivariate_depth: Depth[NDArrayFloat] | None = None,
    pointwise_weights: NDArrayFloat | None = None,
) -> DirectionalOutlyingnessStats:
    r"""
    Compute the directional outlyingness of the functional data.

    Furthermore, it calculates functional, mean and the variational
    directional outlyingness of the samples in the data set, which are also
    returned.

    The functional directional outlyingness can be seen as the overall
    outlyingness, analog to other functional outlyingness measures.

    The mean directional outlyingness, describes the relative
    position (including both distance and direction) of the samples on average
    to the center curve; its norm can be regarded as the magnitude
    outlyingness.

    The variation of the directional outlyingness, measures
    the change of the directional outlyingness in terms of both norm and
    direction across the whole design interval and can be regarded as the
    shape outlyingness.

    Firstly, the directional outlyingness is calculated as follows:

    .. math::
        \mathbf{O}\left(\mathbf{X}(t) , F_{\mathbf{X}(t)}\right) =
        \left\{\frac{1}{d\left(\mathbf{X}(t) , F_{\mathbf{X}(t)}\right)} - 1
        \right\} \cdot \mathbf{v}(t)

    where :math:`\mathbf{X}` is a stochastic process with probability
    distribution :math:`F`, :math:`d` a depth function and :math:`\mathbf{v}(t)
    = \left\{ \mathbf{X}(t) - \mathbf{Z}(t)\right\} / \lVert \mathbf{X}(t) -
    \mathbf{Z}(t) \rVert` is the spatial sign of :math:`\left\{\mathbf{X}(t) -
    \mathbf{Z}(t)\right\}`, :math:`\mathbf{Z}(t)` denotes the median and
    :math:`\lVert \cdot \rVert` denotes the :math:`L_2` norm.

    From the above formula, we define the mean directional outlyingness as:

    .. math::
        \mathbf{MO}\left(\mathbf{X} , F_{\mathbf{X}}\right) = \int_I
        \mathbf{O}\left(\mathbf{X}(t) , F_{\mathbf{X}(t)}\right) \cdot w(t) dt;

    and the variation of the directional outlyingness as:

    .. math::
        VO\left(\mathbf{X} , F_{\mathbf{X}}\right) = \int_I  \lVert\mathbf{O}
        \left(\mathbf{X}(t), F_{\mathbf{X}(t)}\right)-\mathbf{MO}\left(
        \mathbf{X} , F_{\mathbf{X}}\right)  \rVert^2 \cdot w(t) dt

    where :math:`w(t)` a weight function defined on the :term:`domain` of
    :math:`\mathbf{X}`, :math:`I`.

    Then, the total functional outlyingness can be computed using these values:

    .. math::
        FO\left(\mathbf{X} , F_{\mathbf{X}}\right) = \lVert \mathbf{MO}\left(
        \mathbf{X} , F_{\mathbf{X}}\right)\rVert^2 +  VO\left(\mathbf{X} ,
        F_{\mathbf{X}}\right) .

    Args:
        fdatagrid (FDataGrid): Object containing the samples to be ordered
            according to the directional outlyingness.
        multivariate_depth (:ref:`depth measure <depth-measures>`, optional):
            Method used to order the data. Defaults to :func:`projection
            depth <skfda.exploratory.depth.multivariate.ProjectionDepth>`.
        pointwise_weights (array_like, optional): an array containing the
            weights of each point of discretisation where values have been
            recorded. Defaults to the same weight for each of the points:
            1/len(interval).

    Returns:
        DirectionalOutlyingnessStats object.

    Example:

        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2],
        ...                [0.5, 0.5, 1, 2, 1.5, 1],
        ...                [-1, -1, -0.5, 1, 1, 0.5],
        ...                [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> grid_points = [0, 2, 4, 6, 8, 10]
        >>> fd = FDataGrid(data_matrix, grid_points)
        >>> stats = directional_outlyingness_stats(fd)
        >>> stats.directional_outlyingness
        array([[[ 1.33333333],
                [ 1.33333333],
                [ 2.33333333],
                [ 1.5       ],
                [ 1.66666667],
                [ 1.66666667]],
               [[ 0.        ],
                [ 0.        ],
                [ 0.        ],
                [ 0.        ],
                [ 0.        ],
                [ 0.        ]],
               [[-1.33333333],
                [-1.33333333],
                [-1.        ],
                [-0.5       ],
                [-0.33333333],
                [-0.33333333]],
               [[-0.66666667],
                [-0.66666667],
                [-1.        ],
                [-2.5       ],
                [-3.        ],
                [-2.33333333]]])

    >>> stats.functional_directional_outlyingness
    array([ 6.58864198,  6.4608642 ,  6.63753086,  7.40481481])

    >>> stats.mean_directional_outlyingness
    array([[ 1.66666667],
           [ 0.        ],
           [-0.8       ],
           [-1.74444444]])

    >>> stats.variation_directional_outlyingness
    array([ 0.12777778,  0.        ,  0.17666667,  0.94395062])

    References:
        Dai, Wenlin, and Genton, Marc G. "Directional outlyingness for
        multivariate functional data." Computational Statistics & Data
        Analysis 131 (2019): 50-65.

    """
    if fdatagrid.dim_domain > 1:
        raise NotImplementedError("Only support 1 dimension on the domain.")

    if multivariate_depth is None:
        multivariate_depth = ProjectionDepth()

    if (
        pointwise_weights is not None
        and (
            len(pointwise_weights) != len(fdatagrid.grid_points[0])
            or pointwise_weights.sum() != 1
        )
    ):
        raise ValueError(
            "There must be a weight in pointwise_weights for each recorded "
            "time point and altogether must integrate to 1.",
        )

    if pointwise_weights is None:
        pointwise_weights = np.ones(
            len(fdatagrid.grid_points[0]),
        ) / (
            fdatagrid.domain_range[0][1] - fdatagrid.domain_range[0][0]
        )

    depth_pointwise = multivariate_depth(fdatagrid.data_matrix)
    assert depth_pointwise.shape == fdatagrid.data_matrix.shape[:-1]

    # Obtaining the pointwise median sample Z, to calculate
    # v(t) = {X(t) − Z(t)}/|| X(t) − Z(t) ||
    median_index = np.argmax(depth_pointwise, axis=0)
    pointwise_median = fdatagrid.data_matrix[
        median_index,
        range(fdatagrid.data_matrix.shape[1]),
    ]
    assert pointwise_median.shape == fdatagrid.data_matrix.shape[1:]
    v = fdatagrid.data_matrix - pointwise_median
    assert v.shape == fdatagrid.data_matrix.shape
    v_norm = la.norm(v, axis=-1, keepdims=True)

    # To avoid ZeroDivisionError, the zeros are substituted by ones (the
    # reference implementation also does this).
    v_norm[np.where(v_norm == 0)] = 1

    v_unitary = v / v_norm

    # Calculation directinal outlyingness
    dir_outlyingness = (1 / depth_pointwise[..., np.newaxis] - 1) * v_unitary

    # Calculation mean directional outlyingness
    weighted_dir_outlyingness = (
        dir_outlyingness * pointwise_weights[:, np.newaxis]
    )
    assert weighted_dir_outlyingness.shape == dir_outlyingness.shape

    mean_dir_outlyingness = scipy.integrate.simps(
        weighted_dir_outlyingness,
        fdatagrid.grid_points[0],
        axis=1,
    )
    assert mean_dir_outlyingness.shape == (
        fdatagrid.n_samples,
        fdatagrid.dim_codomain,
    )

    # Calculation variation directional outlyingness
    norm = np.square(la.norm(
        dir_outlyingness
        - mean_dir_outlyingness[:, np.newaxis, :],
        axis=-1,
    ))
    weighted_norm = norm * pointwise_weights
    variation_dir_outlyingness = scipy.integrate.simps(
        weighted_norm,
        fdatagrid.grid_points[0],
        axis=1,
    )
    assert variation_dir_outlyingness.shape == (fdatagrid.n_samples,)

    functional_dir_outlyingness = (
        np.square(la.norm(mean_dir_outlyingness))
        + variation_dir_outlyingness
    )
    assert functional_dir_outlyingness.shape == (fdatagrid.n_samples,)

    return DirectionalOutlyingnessStats(
        directional_outlyingness=dir_outlyingness,
        functional_directional_outlyingness=functional_dir_outlyingness,
        mean_directional_outlyingness=mean_dir_outlyingness,
        variation_directional_outlyingness=variation_dir_outlyingness,
    )


class MSPlotOutlierDetector(  # noqa: WPS230
    BaseEstimator,
    OutlierMixin[FDataGrid],
):
    r"""Outlier detector using directional outlyingness.

    Considering :math:`\mathbf{Y} = \left(\mathbf{MO}^T, VO\right)^T`, the
    outlier detection method is implemented as described below.

    First, the square robust Mahalanobis distance is calculated based on a
    sample of size :math:`h \leq fdatagrid.n_samples`:

    .. math::
        {RMD}^2\left( \mathbf{Y}, \mathbf{\tilde{Y}}^*_J\right) = \left(
        \mathbf{Y} - \mathbf{\tilde{Y}}^*_J\right)^T  {\mathbf{S}^*_J}^{-1}
        \left( \mathbf{Y} - \mathbf{\tilde{Y}}^*_J\right)

    where :math:`J` denotes the group of :math:`h` samples that minimizes the
    determinant of the corresponding covariance matrix,
    :math:`\mathbf{\tilde{Y}}^*_J = h^{-1}\sum_{i\in{J}}\mathbf{Y}_i` and
    :math:`\mathbf{S}^*_J = h^{-1}\sum_{i\in{J}}\left( \mathbf{Y}_i - \mathbf{
    \tilde{Y}}^*_J\right) \left( \mathbf{Y}_i - \mathbf{\tilde{Y}}^*_J
    \right)^T`. The sub-sample of size h controls the robustness of the method.

    Then, the tail of this distance distribution is approximated as follows:

    .. math::
        \frac{c\left(m - p\right)}{m\left(p + 1\right)}RMD^2\left(
        \mathbf{Y}, \mathbf{\tilde{Y}}^*_J\right)\sim F_{p+1, m-p}

    where :math:`p` is the dimension of the image plus one, and :math:`c` and
    :math:`m` are parameters determining the degrees of freedom of the
    :math:`F`-distribution and the scaling factor, given by empirical results
    and an asymptotic formula.

    Finally, we choose a cutoff value to determine the outliers, C ,
    as the :math:`\alpha` quantile of :math:`F_{p+1, m-p}`. We set
    :math:`\alpha = 0.993`, which is used in the classical boxplot for
    detecting outliers under a normal distribution.

    Parameters:
        multivariate_depth: Method used to order the data. Defaults
            to :class:`projection depth
            <fda.depth_measures.multivariate.ProjectionDepth>`.
        pointwise_weights: an array containing the
            weights of each points of discretisati on where values have
            been recorded.
        cutoff_factor: Factor that multiplies the cutoff value, in order to
            consider more or less curves as outliers.
        assume_centered: If True, the support of the
            robust location and the covariance estimates is computed, and a
            covariance estimate is recomputed from it, without centering
            the data. Useful to work with data whose mean is significantly
            equal to zero but is not exactly zero. If False, default value,
            the robust location and covariance are directly computed with
            the FastMCD algorithm without additional treatment.
        support_fraction: The proportion of points to be included in the
            support of the raw MCD estimate.
            Default is None, which implies that the minimum value of
            support_fraction will be used within the algorithm:
            [n_sample + n_features + 1] / 2
        random_state: If int,
            random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number
            generator; If None, the random number generator is the
            RandomState instance used by np.random. By default, it is 0.

    Example:
        Function :math:`f : \mathbb{R}\longmapsto\mathbb{R}`.

        >>> import skfda
        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2],
        ...                [0.5, 0.5, 1, 2, 1.5, 1],
        ...                [-1, -1, -0.5, 1, 1, 0.5],
        ...                [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> grid_points = [0, 2, 4, 6, 8, 10]
        >>> fd = skfda.FDataGrid(data_matrix, grid_points)
        >>> out_detector = MSPlotOutlierDetector()
        >>> out_detector.fit_predict(fd)
        array([1, 1, 1, 1])

    References:
        Dai, Wenlin, and Genton, Marc G. "Multivariate functional data
        visualization and outlier detection." Journal of Computational
        and Graphical Statistics 27.4 (2018): 923-934.

    """

    def __init__(
        self,
        *,
        multivariate_depth: Depth[NDArrayFloat] | None = None,
        pointwise_weights: NDArrayFloat | None = None,
        assume_centered: bool = False,
        support_fraction: float | None = None,
        num_resamples: int = 1000,
        random_state: RandomStateLike = 0,
        cutoff_factor: float = 1,
        _force_asymptotic: bool = False,
    ) -> None:
        self.multivariate_depth = multivariate_depth
        self.pointwise_weights = pointwise_weights
        self.assume_centered = assume_centered
        self.support_fraction = support_fraction
        self.num_resamples = num_resamples
        self.random_state = random_state
        self.cutoff_factor = cutoff_factor
        self._force_asymptotic = _force_asymptotic

    def _compute_points(self, X: FDataGrid) -> NDArrayFloat:
        multivariate_depth = self.multivariate_depth
        if multivariate_depth is None:
            multivariate_depth = ProjectionDepth()

        # The depths of the samples are calculated giving them an ordering.
        stats = directional_outlyingness_stats(
            X,
            multivariate_depth=multivariate_depth,
            pointwise_weights=self.pointwise_weights,
        )

        mean = stats.mean_directional_outlyingness
        variation = stats.variation_directional_outlyingness[:, np.newaxis]

        return np.concatenate((mean, variation), axis=1)

    def _parameters_asymptotic(  # noqa: WPS210
        self,
        sample_size: int,
        dimension: int,
    ) -> Tuple[float, float]:
        """Return the scaling and cutoff parameters via asymptotic formula."""
        n = sample_size
        p = dimension

        h = np.floor((n + p + 1) / 2)

        # c estimation
        xi_left = scipy.stats.chi2.rvs(
            size=self.num_resamples,
            df=p + 2,
            random_state=self.random_state_,
        )
        xi_right = scipy.stats.ncx2.rvs(
            size=self.num_resamples,
            df=p,
            nc=h / n,
            random_state=self.random_state_,
        )

        c_numerator = np.sum(xi_left < xi_right) / self.num_resamples
        c_denominator = h / n

        estimated_c = c_numerator / c_denominator

        # m estimation
        alpha = (n - h) / n
        alpha_compl = 1 - alpha
        q_alpha = scipy.stats.chi2.ppf(alpha_compl, df=p)

        dist_p2 = scipy.stats.chi2.cdf(q_alpha, df=p + 2)
        dist_p4 = scipy.stats.chi2.cdf(q_alpha, df=p + 4)
        c_alpha = alpha_compl / dist_p2
        c2 = -dist_p2 / 2
        c3 = -dist_p4 / 2
        c4 = 3 * c3

        b1 = (c3 - c4) / dist_p2
        b2 = (
            0.5 + 1 / dist_p2
            * (c3 - q_alpha / p * (c2 + alpha_compl / 2))
        )

        v1 = (
            alpha_compl * b1**2
            * (alpha * (c_alpha * q_alpha / p - 1) ** 2 - 1)
            - 2 * c3 * c_alpha**2
            * (
                3 * (b1 - p * b2)**2
                + (p + 2) * b2 * (2 * b1 - p * b2)
            )
        )
        v2 = n * (b1 * (b1 - p * b2) * alpha_compl)**2 * c_alpha**2
        v = v1 / v2

        m_asympt = 2 / (c_alpha**2 * v)

        estimated_m = (
            m_asympt
            * np.exp(0.725 - 0.00663 * p - 0.078 * np.log(n))  # noqa: WPS432
        )

        dfn = p
        dfd = estimated_m - p + 1

        # Calculation of the cutoff value and scaling factor to identify
        # outliers.
        scaling = estimated_c * dfd / estimated_m / dfn
        cutoff_value = scipy.stats.f.ppf(
            0.993,  # noqa: WPS432
            dfn,
            dfd,
            loc=0,
            scale=1,
        )

        return scaling, cutoff_value

    def _parameters_numeric(
        self,
        sample_size: int,
        dimension: int,
    ) -> Tuple[float, float]:

        key = sample_size // 5

        use_asympt = True

        if not self._force_asymptotic:
            if dimension == 2:
                scaling_list = experiments.dim2_scaling_list
                cutoff_list = experiments.dim2_cutoff_list
                assert len(scaling_list) == len(cutoff_list)
                if key < len(scaling_list):
                    use_asympt = False

            elif dimension == 3:
                scaling_list = experiments.dim3_scaling_list
                cutoff_list = experiments.dim3_cutoff_list
                assert len(scaling_list) == len(cutoff_list)
                if key < len(scaling_list):
                    use_asympt = False

        if use_asympt:
            return self._parameters_asymptotic(sample_size, dimension)

        return scaling_list[key], cutoff_list[key]

    def fit_predict(  # noqa: D102
        self,
        X: FDataGrid,
        y: object = None,
    ) -> NDArrayInt:

        self.random_state_ = validate_random_state(self.random_state)
        self.points_ = self._compute_points(X)

        # The square mahalanobis distances of the samples are
        # calulated using MCD.
        self.cov_ = MinCovDet(
            store_precision=False,
            assume_centered=self.assume_centered,
            support_fraction=self.support_fraction,
            random_state=self.random_state_,
        )
        self.cov_.fit(self.points_)

        # Calculation of the degrees of freedom of the F-distribution
        # (approximation of the tail of the distance distribution).

        # One per dimension (mean dir out) plus one (variational dir out)
        dimension = X.dim_codomain + 1
        if self._force_asymptotic:
            scaling, cutoff_value = self._parameters_asymptotic(
                sample_size=X.n_samples,
                dimension=dimension,
            )
        else:
            scaling, cutoff_value = self._parameters_numeric(
                sample_size=X.n_samples,
                dimension=dimension,
            )

        self.scaling_ = scaling
        self.cutoff_value_ = cutoff_value * self.cutoff_factor

        rmd_2: NDArrayFloat = self.cov_.mahalanobis(self.points_)

        outliers = self.scaling_ * rmd_2 > self.cutoff_value_

        # Predict as scikit-learn outlier detectors
        return ~outliers + outliers * -1
