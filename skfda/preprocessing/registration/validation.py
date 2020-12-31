"""Methods and classes for validation of the registration procedures"""

from typing import NamedTuple

import numpy as np

from ..._utils import check_is_univariate, _to_grid


class RegistrationScorer():
    r"""Cross validation scoring for registration procedures.

    It calculates the score of a registration procedure, used to perform
    model validation or parameter selection.

    Attributes:
        eval_points (array_like, optional): Set of points where the
            functions are evaluated to obtain a discrete representation and
            perform the calculation.

    Args:
        estimator (Estimator): Registration method estimator. The estimator
            should be fitted.
        X (:class:`FData <skfda.FData>`): Functional data to be registered.
        y (:class:`FData <skfda.FData>`, optional): Functional data target.
            If provided should be the same as `X` in general.

    Returns:
        float: Cross validation score.

    Note:
        The scorer passes the warpings generated in the registration procedure
        to the `score_function` when necessary.

    See also:
        :class:`~AmplitudePhaseDecomposition`
        :class:`~LeastSquares`
        :class:`~SobolevLeastSquares`
        :class:`~PairwiseCorrelation`

    """

    def __init__(self, eval_points=None):
        """Initialize the transformer"""
        self.eval_points = eval_points

    def __call__(self, estimator, X, y=None):
        """Compute the score of the transformation.

        Args:
            estimator (Estimator): Registration method estimator. The estimator
                should be fitted.
            X (:class:`FData <skfda.FData>`): Functional data to be registered.
            y (:class:`FData <skfda.FData>`, optional): Functional data target.
                If provided should be the same as `X` in general.

        Returns:
            float: Cross validation score.
        """
        if y is None:
            y = X

        # Register the data
        X_reg = estimator.transform(X)

        return self.score_function(y, X_reg)


class AmplitudePhaseDecompositionStats(NamedTuple):
    r"""Named tuple to store the values of the amplitude-phase decomposition.

    Values of the amplitude phase decomposition computed in
    :func:`mse_r_squared`, returned when `return_stats` is `True`.

    Args:
        r_square (float): Squared correlation index :math:`R^2`.
        mse_amp (float): Mean square error of amplitude
            :math:`\text{MSE}_{amp}`.
        mse_pha (float): Mean square error of phase :math:`\text{MSE}_{pha}`.
        c_r (float): Constant :math:`C_R`.

    """
    r_squared: float
    mse_amp: float
    mse_pha: float
    c_r: float


class AmplitudePhaseDecomposition(RegistrationScorer):
    r"""Compute mean square error measures for amplitude and phase variation.

    Once the registration has taken place, this function computes two mean
    squared error measures, one for amplitude variation, and the other for
    phase variation and returns a squared multiple correlation index
    of the amount of variation in the unregistered functions is due to phase.

    Let :math:`x_i(t),y_i(t)` be the unregistered and registered functions
    respectively. The total mean square error measure (see [RGS09-8-5]_) is
    defined as


    .. math::
        \text{MSE}_{total}=
        \frac{1}{N}\sum_{i=1}^{N}\int[x_i(t)-\overline x(t)]^2dt

    The measures of amplitude and phase mean square error are

    .. math::
        \text{MSE}_{amp} =  C_R \frac{1}{N}
        \sum_{i=1}^{N} \int \left [ y_i(t) - \overline{y}(t) \right ]^2 dt

    .. math::
        \text{MSE}_{phase}=
        \int \left [C_R \overline{y}^2(t) - \overline{x}^2(t) \right]dt

    where the constant :math:`C_R` is defined as

    .. math::

        C_R = 1 + \frac{\frac{1}{N}\sum_{i}^{N}\int [Dh_i(t)-\overline{Dh}(t)]
        [ y_i^2(t)- \overline{y^2}(t) ]dt}
        {\frac{1}{N} \sum_{i}^{N} \int y_i^2(t)dt}

    whose structure is related to the covariation between the deformation
    functions :math:`Dh_i(t)` and the squared registered functions
    :math:`y_i^2(t)`. When these two sets of functions are independents
    :math:`C_R=1`, as in the case of shift registration.

    The total mean square error is decomposed in the two sources of
    variability.

    .. math::
        \text{MSE}_{total} = \text{MSE}_{amp} + \text{MSE}_{phase}

    The squared multiple correlation index of the proportion of the total
    variation due to phase is defined as:

    .. math::
        R^2 = \frac{\text{MSE}_{phase}}{\text{MSE}_{total}}

    See [KR08-3]_ for a detailed explanation.

    Attributes:
        return_stats (boolean, optional): If `true` returns a named tuple
            with four values: :math:`R^2`, :math:`MSE_{amp}`, :math:`MSE_{pha}`
            and :math:`C_R`. Otherwise the squared correlation index
            :math:`R^2` is returned. Default `False`.

        eval_points (array_like, optional): Set of points where the
            functions are evaluated to obtain a discrete representation and
            perform the calculation.


    Args:
        estimator (RegistrationTransformer): Registration transformer.
        X (:class:`FData`): Unregistered functions.
        y (:class:`FData`, optional): Target data, generally the same as X. By
            default 'None', which uses `X` as target.


    Returns:
        (float or :class:`NamedTuple <typing.NamedTuple>`): squared correlation
        index :math:`R^2` if `return_stats` is `False`. Otherwise a named
        tuple containing:

            * `r_squared`: Squared correlation index :math:`R^2`.
            * `mse_amp`: Mean square error of amplitude
              :math:`\text{MSE}_{amp}`.
            * `mse_pha`: Mean square error of phase :math:`\text{MSE}_{pha}`.
            * `c_r`: Constant :math:`C_R`.


    Raises:
        ValueError: If the functional data is not univariate.

    References:
        ..  [KR08-3] Kneip, Alois & Ramsay, James. (2008).  Quantifying
            amplitude and phase variation. In *Combining Registration and
            Fitting for Functional Models* (pp. 14-15). Journal of the American
            Statistical Association.
        ..  [RGS09-8-5] Ramsay J.O., Giles Hooker & Spencer Graves (2009). In
            *Functional Data Analysis with R and Matlab* (pp. 125-126).
            Springer.

    Examples:

        Calculate the score of the shift registration of a sinusoidal process
        synthetically generated.

        >>> from skfda.preprocessing.registration.validation import \
        ...                                         AmplitudePhaseDecomposition
        >>> from skfda.preprocessing.registration import ShiftRegistration
        >>> from skfda.datasets import make_sinusoidal_process
        >>> X = make_sinusoidal_process(error_std=0, random_state=0)

        Fit the registration procedure.

        >>> shift_registration = ShiftRegistration()
        >>> shift_registration.fit(X)
        ShiftRegistration(...)

        Compute the :math:`R^2` correlation index

        >>> scorer = AmplitudePhaseDecomposition()
        >>> score = scorer(shift_registration, X)
        >>> round(score, 3)
        0.972

        Also it is possible to get all the values of the decomposition.

        >>> scorer = AmplitudePhaseDecomposition(return_stats=True)
        >>> stats = scorer(shift_registration, X)
        >>> round(stats.r_squared, 3)
        0.972
        >>> round(stats.mse_amp, 3)
        0.007
        >>> round(stats.mse_pha, 3)
        0.227
        >>> round(stats.c_r, 3)
        1.0


    See also:
        :class:`~LeastSquares`
        :class:`~SobolevLeastSquares`
        :class:`~PairwiseCorrelation`

    """

    def __init__(self, return_stats=False, eval_points=None):
        """Initialize the transformer"""
        super().__init__(eval_points)
        self.return_stats = return_stats

    def __call__(self, estimator, X, y=None):
        """Compute the score of the transformation.

        Args:
            estimator (Estimator): Registration method estimator. The estimator
                should be fitted.
            X (:class:`FData <skfda.FData>`): Functional data to be registered.
            y (:class:`FData <skfda.FData>`, optional): Functional data target.
                If provided should be the same as `X` in general.

        Returns:
            float: Cross validation score.
        """
        if y is None:
            y = X

        # Register the data
        X_reg = estimator.transform(X)

        # Pass the warpings if are generated in the transformer
        if hasattr(estimator, 'warping_'):
            return self.score_function(y, X_reg, warping=estimator.warping_)
        else:
            return self.score_function(y, X_reg)

    def score_function(self, X, y, *, warping=None):
        """Compute the score of the transformation performed.

        Args:
            X (FData): Original functional data.
            y (FData): Functional data registered.

        Returns:
            float: Score of the transformation.

        """
        from scipy.integrate import simps

        check_is_univariate(X)
        check_is_univariate(y)

        if len(y) != len(X):
            raise ValueError(f"the registered and unregistered curves must have "
                             f"the same number of samples ({len(y)})!=({len(X)})")

        if warping is not None and len(warping) != len(X):
            raise ValueError(f"The registered curves and the warping functions "
                             f"must have the same number of samples "
                             f"({len(X)})!=({len(warping)})")

        # Creates the mesh to discretize the functions
        if self.eval_points is None:
            try:
                eval_points = y.grid_points[0]

            except AttributeError:
                nfine = max(y.basis.n_basis * 10 + 1, 201)
                eval_points = np.linspace(*y.domain_range[0], nfine)
        else:
            eval_points = np.asarray(self.eval_points)

        x_fine = X.evaluate(eval_points)[..., 0]
        y_fine = y.evaluate(eval_points)[..., 0]
        mu_fine = x_fine.mean(axis=0)  # Mean unregistered function
        eta_fine = y_fine.mean(axis=0)  # Mean registered function
        mu_fine_sq = np.square(mu_fine)
        eta_fine_sq = np.square(eta_fine)

        # Total mean square error of the original funtions
        # mse_total = scipy.integrate.simps(
        #    np.mean(np.square(x_fine - mu_fine), axis=0),
        #    eval_points)

        cr = 1.  # Constant related to the covariation between the deformation
        # functions and y^2

        # If the warping functions are not provided, are suppose independent
        if warping is not None:
            # Derivates warping functions
            warping_deriv = warping.derivative()
            dh_fine = warping_deriv(eval_points)[..., 0]
            dh_fine_mean = dh_fine.mean(axis=0)
            dh_fine_center = dh_fine - dh_fine_mean

            y_fine_sq = np.square(y_fine)  # y^2
            y_fine_sq_center = np.subtract(y_fine_sq, eta_fine_sq)  # y^2-E[y2]

            covariate = np.inner(dh_fine_center.T, y_fine_sq_center.T)
            covariate = covariate.mean(axis=0)
            cr += np.divide(simps(covariate, eval_points),
                            simps(eta_fine_sq, eval_points))

        # mse due to phase variation
        mse_pha = simps(cr * eta_fine_sq - mu_fine_sq, eval_points)

        # mse due to amplitude variation
        # mse_amp = mse_total - mse_pha
        y_fine_center = np.subtract(y_fine, eta_fine)
        y_fine_center_sq = np.square(y_fine_center, out=y_fine_center)
        y_fine_center_sq_mean = y_fine_center_sq.mean(axis=0)

        mse_amp = simps(y_fine_center_sq_mean, eval_points)

        # Total mean square error of the original funtions
        mse_total = mse_pha + mse_amp

        # squared correlation measure of proportion of phase variation
        rsq = mse_pha / (mse_total)

        if self.return_stats is True:
            stats = AmplitudePhaseDecompositionStats(rsq, mse_amp, mse_pha, cr)
            return stats

        return rsq


class LeastSquares(AmplitudePhaseDecomposition):
    r"""Cross-validated measure of the registration procedure.

    Computes a cross-validated measure of the level of synchronization
    [James07]_:

    .. math::
        ls=1 - \frac{1}{N} \sum_{i=1}^{N} \frac{\int\left(\tilde{f}_{i}(t)-
        \frac{1}{N-1} \sum_{j \neq i} \tilde{f}_{j}(t)\right)^{2} dt}{\int
        \left(f_{i}(t)-\frac{1}{N-1} \sum_{j \neq i} f_{j}(t)\right)^{2} dt}

    where :math:`f_i` and :math:`\tilde f_i` are the original and the
    registered data respectively.

    The :math:`ls` measures the total cross-sectional variance of the aligned
    functions, relative to the original value.
    A value of :math:`1` would indicate an identical shape for all registered
    curves, while zero corresponds to no improvement in the synchronization. It
    can be negative because the model can be arbitrarily worse.

    Attributes:
        eval_points (array_like, optional): Set of points where the
            functions are evaluated to obtain a discrete representation and
            perform the calculation.

    Args:
        estimator (RegistrationTransformer): Registration transformer.
        X (:class:`FData <skfda.FData>`): Original functional data.
        y (:class:`FData <skfda.FData>`): Registered functional data.


    Note:
        The original least square measure used in [S11-5-2-1]_ is defined as
        :math:`1 - ls`, but has been modified according to the scikit-learn
        scorers, where higher values correspond to better cross-validated
        measures.


    References:
        .. [James07] G. James. Curve alignments by moments. Annals of Applied
            Statistics, 1(2):480–501, 2007.
        .. [S11-5-2-1] Srivastava, Anuj et. al. Registration of Functional Data
            Using Fisher-Rao Metric (2011). In *Comparisons with other Methods*
            (p. 18). arXiv:1103.3817v2.

    Examples:

        Calculate the score of the shift registration of a sinusoidal process
        synthetically generated.

        >>> from skfda.preprocessing.registration.validation import \
        ...                                                        LeastSquares
        >>> from skfda.preprocessing.registration import ShiftRegistration
        >>> from skfda.datasets import make_sinusoidal_process
        >>> X = make_sinusoidal_process(error_std=0, random_state=0)

        Fit the registration procedure.

        >>> shift_registration = ShiftRegistration()
        >>> shift_registration.fit(X)
        ShiftRegistration(...)

        Compute the least squares score.
        >>> scorer = LeastSquares()
        >>> score = scorer(shift_registration, X)
        >>> round(score, 3)
        0.796


    See also:
        :class:`~AmplitudePhaseDecomposition`
        :class:`~SobolevLeastSquares`
        :class:`~PairwiseCorrelation`

    """

    def score_function(self, X, y):
        """Compute the score of the transformation performed.

        Args:
            X (FData): Original functional data.
            y (FData): Functional data registered.

        Returns:
            float: Score of the transformation.

        """
        from ...misc.metrics import pairwise_distance, lp_distance

        check_is_univariate(X)
        check_is_univariate(y)

        X, y = _to_grid(X, y, eval_points=self.eval_points)

        # Instead of compute f_i - 1/(N-1) sum(j!=i)f_j for each i = 1 ... N
        # It is used (1 + 1/(N-1))f_i - 1/(N-1) sum(j=1 ... N) f_j =
        # (1 + 1/(N-1))f_i - N/(N-1) mean(f) =
        # C1 * f_1 - C2 mean(f) for each i= 1 ... N
        N = len(X)
        C1 = 1 + 1 / (N - 1)
        C2 = N / (N - 1)

        X = C1 * X
        y = C1 * y
        mean_X = C2 * X.mean()
        mean_y = C2 * y.mean()

        # Compute distance to mean
        distance = pairwise_distance(lp_distance)
        ls_x = distance(X, mean_X).flatten()
        ls_y = distance(y, mean_y).flatten()

        # Quotient of distance
        quotient = ls_y / ls_x

        return 1 - 1. / N * quotient.sum()


class SobolevLeastSquares(RegistrationScorer):
    r"""Cross-validated measure of the registration procedure.

    Computes a cross-validated measure of the level of synchronization
    [S11-5-2-3]_:

    .. math::
        sls=1 - \frac{\sum_{i=1}^{N} \int\left(\dot{\tilde{f}}_{i}(t)-
        \frac{1}{N} \sum_{j=1}^{N} \dot{\tilde{f}}_{j}\right)^{2} dt}
        {\sum_{i=1}^{N} \int\left(\dot{f}_{i}(t)-\frac{1}{N} \sum_{j=1}^{N}
        \dot{f}_{j}\right)^{2} dt}

    where :math:`\dot f_i` and :math:`\dot \tilde f_i` are the derivatives of
    the original and the registered data respectively.

    This criterion measures the total cross-sectional variance of the
    derivatives of the aligned functions, relative to the original value.
    A value of :math:`1` would indicate an identical shape for all registered
    curves, while zero corresponds to no improvement in the registration. It
    can be negative because the model can be arbitrarily worse.

    Attributes:
        eval_points (array_like, optional): Set of points where the
            functions are evaluated to obtain a discrete representation and
            perform the calculation.

    Args:
        estimator (RegistrationTransformer): Registration transformer.
        X (:class:`FData <skfda.FData>`): Original functional data.
        y (:class:`FData <skfda.FData>`): Registered functional data.

    Note:
        The original sobolev least square measure used in [S11-5-2-3]_ is
        defined as :math:`1 - sls`, but has been modified according to the
        scikit-learn scorers, where higher values correspond to better
        cross-validated measures.


    References:
        .. [S11-5-2-3] Srivastava, Anuj et. al. Registration of Functional Data
            Using Fisher-Rao Metric (2011). In *Comparisons with other Methods*
            (p. 18). arXiv:1103.3817v2.

    Examples:

        Calculate the score of the shift registration of a sinusoidal process
        synthetically generated.

        >>> from skfda.preprocessing.registration.validation import \
        ...                                                 SobolevLeastSquares
        >>> from skfda.preprocessing.registration import ShiftRegistration
        >>> from skfda.datasets import make_sinusoidal_process
        >>> X = make_sinusoidal_process(error_std=0, random_state=0)

        Fit the registration procedure.

        >>> shift_registration = ShiftRegistration()
        >>> shift_registration.fit(X)
        ShiftRegistration(...)

        Compute the sobolev least squares score.
        >>> scorer = SobolevLeastSquares()
        >>> score = scorer(shift_registration, X)
        >>> round(score, 3)
        0.761

    See also:
        :class:`~AmplitudePhaseDecomposition`
        :class:`~LeastSquares`
        :class:`~PairwiseCorrelation`

    """

    def score_function(self, X, y):
        """Compute the score of the transformation performed.

        Args:
            X (FData): Original functional data.
            y (FData): Functional data registered.

        Returns:
            float: Score of the transformation.

        """
        from ...misc.metrics import pairwise_distance, lp_distance

        check_is_univariate(X)
        check_is_univariate(y)

        # Compute derivative
        X = X.derivative()
        y = y.derivative()

        # Discretize if needed
        X, y = _to_grid(X, y, eval_points=self.eval_points)

        # L2 distance to mean
        distance = pairwise_distance(lp_distance)

        sls_x = distance(X, X.mean())
        sls_y = distance(y, y.mean())

        return 1 - sls_y.sum() / sls_x.sum()


class PairwiseCorrelation(RegistrationScorer):
    r"""Cross-validated measure of pairwise correlation between functions.

    Computes a cross-validated pairwise correlation between functions
    to compare registration methods [S11-5-2-2]_ :

    .. math::
        pc=\frac{\sum_{i \neq j} \operatorname{cc}\left(\tilde{f}_{i}(t),
        \tilde{f}_{j}(t)\right)}{\sum_{i \neq j}
        \operatorname{cc}\left(f_{i}(t), f_{j}(t)\right)}

    where :math:`f_i` and :math:`\tilde f_i` are the original and registered
    data respectively and :math:`cc(f, g)` is the pairwise Pearson’s
    correlation between functions.

    The larger the value of :math:`pc`, the better the alignment between
    functions in general.

    Attributes:
        eval_points (array_like, optional): Set of points where the
            functions are evaluated to obtain a discrete representation and
            perform the calculation.

    Args:
        estimator (RegistrationTransformer): Registration transformer.
        X (:class:`FData <skfda.FData>`): Original functional data.
        y (:class:`FData <skfda.FData>`): Registered functional data.

    Note:
        Pearson’s correlation between functions is calculated assuming
        the samples are equiespaciated.

    References:
        .. [S11-5-2-2] Srivastava, Anuj et. al. Registration of Functional Data
            Using Fisher-Rao Metric (2011). In *Comparisons with other Methods*
            (p. 18). arXiv:1103.3817v2.

    Examples:

        Calculate the score of the shift registration of a sinusoidal process
        synthetically generated.

        >>> from skfda.preprocessing.registration.validation import \
        ...                                                 PairwiseCorrelation
        >>> from skfda.preprocessing.registration import ShiftRegistration
        >>> from skfda.datasets import make_sinusoidal_process
        >>> X = make_sinusoidal_process(error_std=0, random_state=0)

        Fit the registration procedure.

        >>> shift_registration = ShiftRegistration()
        >>> shift_registration.fit(X)
        ShiftRegistration(...)

        Compute the pairwise correlation score.
        >>> scorer = PairwiseCorrelation()
        >>> score = scorer(shift_registration, X)
        >>> round(score, 3)
        1.816

    See also:
        :class:`~AmplitudePhaseDecomposition`
        :class:`~LeastSquares`
        :class:`~SobolevLeastSquares`

    """

    def score_function(self, X, y):
        """Compute the score of the transformation performed.

        Args:
            X (FData): Original functional data.
            y (FData): Functional data registered.

        Returns:
            float: Score of the transformation.

        """
        check_is_univariate(X)
        check_is_univariate(y)

        # Discretize functional data if needed
        X, y = _to_grid(X, y, eval_points=self.eval_points)

        # Compute correlation matrices with zeros in diagonal
        # corrcoefs computes the correlation between vector, without weights
        # due to the sample points
        X_corr = np.corrcoef(X.data_matrix[..., 0])
        np.fill_diagonal(X_corr, 0.)

        y_corr = np.corrcoef(y.data_matrix[..., 0])
        np.fill_diagonal(y_corr, 0.)

        return y_corr.sum() / X_corr.sum()
