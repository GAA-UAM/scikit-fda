"""Methods and classes for validation of the registration procedures."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from ..._utils import _to_grid
from ...misc.validation import check_fdata_dimensions
from ...representation import FData
from ...typing._numpy import NDArrayFloat
from .base import RegistrationTransformer

Input = TypeVar("Input", bound=FData)
Output = TypeVar("Output", bound=FData)


class RegistrationScorer(ABC, Generic[Input, Output]):
    """Cross validation scoring for registration procedures.

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

    def __call__(
        self,
        estimator: RegistrationTransformer[Input, Output],
        X: Input,
        y: Output | None = None,
    ) -> float:
        """Compute the score of the transformation.

        Args:
            estimator: Registration method estimator. The estimator
                should be fitted.
            X: Functional data to be registered.
            y: Functional data target. If provided should be the same as
                `X` in general.

        Returns:
            Cross validation score.
        """
        if y is None:
            y = X

        # Register the data
        X_reg = estimator.transform(X)

        return self.score_function(y, X_reg)

    @abstractmethod
    def score_function(
        self,
        X: Input,
        y: Output,
    ) -> float:
        """Compute the score of the transformation performed.

        Args:
            X: Original functional data.
            y: Functional data registered.

        Returns:
            Score of the transformation.

        """
        pass


@dataclass
class AmplitudePhaseDecompositionStats():
    r"""Named tuple to store the values of the amplitude-phase decomposition.

    Values of the amplitude phase decomposition computed in
    :func:`mse_r_squared`, returned when `return_stats` is `True`.

    Args:
        r_square (float): Squared correlation index :math:`R^2`.
        mse_amplitude (float): Mean square error of amplitude
            :math:`\text{MSE}_{amp}`.
        mse_phase (float): Mean square error of phase :math:`\text{MSE}_{pha}`.
        c_r (float): Constant :math:`C_R`.

    """

    r_squared: float
    mse_amplitude: float
    mse_phase: float
    c_r: float


class AmplitudePhaseDecomposition(
    RegistrationScorer[FData, FData],
):
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
        C_R \int \overline{y}^2(t) dt - \int \overline{x}^2(t) dt

    where the constant :math:`C_R` is defined as

    .. math::

        C_R = \frac{\frac{1}{N}\sum_{i=1}^{N}\int[x_i(t)-\overline x(t)]^2dt
        }{\frac{1}{N}\sum_{i=1}^{N}\int[y_i(t)-\overline y(t)]^2dt}

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
        >>> from skfda.preprocessing.registration import (
        ...     LeastSquaresShiftRegistration,
        ... )
        >>> from skfda.datasets import make_sinusoidal_process
        >>> X = make_sinusoidal_process(error_std=0, random_state=0)

        Fit the registration procedure.

        >>> shift_registration = LeastSquaresShiftRegistration()
        >>> shift_registration.fit(X)
        LeastSquaresShiftRegistration(...)

        Compute the :math:`R^2` correlation index

        >>> scorer = AmplitudePhaseDecomposition()
        >>> score = scorer(shift_registration, X)
        >>> round(score, 3)
        0.971

        Also it is possible to get all the values of the decomposition:

        >>> X_reg = shift_registration.transform(X)
        >>> stats = scorer.stats(X, X_reg)
        >>> round(stats.r_squared, 3)
        0.971
        >>> round(stats.mse_amplitude, 3)
        0.006
        >>> round(stats.mse_phase, 3)
        0.214
        >>> round(stats.c_r, 3)
        0.976


    See also:
        :class:`~LeastSquares`
        :class:`~SobolevLeastSquares`
        :class:`~PairwiseCorrelation`

    """

    def stats(
        self,
        X: FData,
        y: FData,
    ) -> AmplitudePhaseDecompositionStats:
        """
        Compute the decomposition statistics.

        Args:
            X: Original functional data.
            y: Functional data registered.

        Returns:
            The decomposition statistics.
        """
        from ...misc.metrics import l2_distance, l2_norm

        check_fdata_dimensions(
            X,
            dim_domain=1,
            dim_codomain=1,
        )
        check_fdata_dimensions(
            y,
            dim_domain=1,
            dim_codomain=1,
        )

        if len(y) != len(X):
            raise ValueError(
                f"The registered and unregistered curves must have "
                f"the same number of samples ({len(y)})!=({len(X)})",
            )

        X_mean = X.mean()
        y_mean = y.mean()

        c_r = np.sum(l2_norm(X)**2) / np.sum(l2_norm(y)**2)

        mse_amplitude = c_r * np.mean(l2_distance(y, y.mean())**2)
        mse_phase = (c_r * l2_norm(y_mean)**2 - l2_norm(X_mean)**2).item()

        # Should be equal to np.mean(l2_distance(X, X_mean)**2)
        mse_total = mse_amplitude + mse_phase

        # squared correlation measure of proportion of phase variation
        rsq = mse_phase / mse_total

        return AmplitudePhaseDecompositionStats(
            r_squared=rsq,
            mse_amplitude=mse_amplitude,
            mse_phase=mse_phase,
            c_r=c_r,
        )

    def score_function(
        self,
        X: FData,
        y: FData,
    ) -> float:
        """Compute the score of the transformation performed.

        Args:
            X: Original functional data.
            y: Functional data registered.

        Returns:
            Score of the transformation.

        """
        return float(self.stats(X, y).r_squared)


class LeastSquares(RegistrationScorer[FData, FData]):
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
        >>> from skfda.preprocessing.registration import (
        ...     LeastSquaresShiftRegistration,
        ... )
        >>> from skfda.datasets import make_sinusoidal_process
        >>> X = make_sinusoidal_process(error_std=0, random_state=0)

        Fit the registration procedure.

        >>> shift_registration = LeastSquaresShiftRegistration()
        >>> shift_registration.fit(X)
        LeastSquaresShiftRegistration(...)

        Compute the least squares score.
        >>> scorer = LeastSquares()
        >>> score = scorer(shift_registration, X)
        >>> round(score, 3)
        0.953


    See also:
        :class:`~AmplitudePhaseDecomposition`
        :class:`~SobolevLeastSquares`
        :class:`~PairwiseCorrelation`

    """

    def score_function(self, X: FData, y: FData) -> float:
        """Compute the score of the transformation performed.

        Args:
            X (FData): Original functional data.
            y (FData): Functional data registered.

        Returns:
            float: Score of the transformation.

        """
        from ...misc.metrics import l2_distance

        check_fdata_dimensions(
            X,
            dim_domain=1,
            dim_codomain=1,
        )
        check_fdata_dimensions(
            y,
            dim_domain=1,
            dim_codomain=1,
        )

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
        ls_x = l2_distance(X, mean_X)**2
        ls_y = l2_distance(y, mean_y)**2

        # Quotient of distance
        quotient = ls_y / ls_x

        return float(1 - np.mean(quotient))


class SobolevLeastSquares(RegistrationScorer[FData, FData]):
    r"""Cross-validated measure of the registration procedure.

    Computes a cross-validated measure of the level of synchronization
    [S11-5-2-3]_:

    .. math::
        sls=1 - \frac{\sum_{i=1}^{N} \int\left(\dot{\tilde{f}}_{i}(t)-
        \frac{1}{N} \sum_{j=1}^{N} \dot{\tilde{f}}_{j}\right)^{2} dt}
        {\sum_{i=1}^{N} \int\left(\dot{f}_{i}(t)-\frac{1}{N} \sum_{j=1}^{N}
        \dot{f}_{j}\right)^{2} dt}

    where :math:`\dot{f}_i` and :math:`\dot{\tilde{f}}_i` are the derivatives
    of the original and the registered data respectively.

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
        >>> from skfda.preprocessing.registration import (
        ...     LeastSquaresShiftRegistration,
        ... )
        >>> from skfda.datasets import make_sinusoidal_process
        >>> X = make_sinusoidal_process(error_std=0, random_state=0)

        Fit the registration procedure.

        >>> shift_registration = LeastSquaresShiftRegistration()
        >>> shift_registration.fit(X)
        LeastSquaresShiftRegistration(...)

        Compute the sobolev least squares score.
        >>> scorer = SobolevLeastSquares()
        >>> score = scorer(shift_registration, X)
        >>> round(score, 3)
        0.924

    See also:
        :class:`~AmplitudePhaseDecomposition`
        :class:`~LeastSquares`
        :class:`~PairwiseCorrelation`

    """

    def score_function(self, X: FData, y: FData) -> float:
        """Compute the score of the transformation performed.

        Args:
            X (FData): Original functional data.
            y (FData): Functional data registered.

        Returns:
            float: Score of the transformation.

        """
        from ...misc.metrics import l2_distance

        check_fdata_dimensions(
            X,
            dim_domain=1,
            dim_codomain=1,
        )
        check_fdata_dimensions(
            y,
            dim_domain=1,
            dim_codomain=1,
        )

        # Compute derivative
        X = X.derivative()
        y = y.derivative()

        # L2 distance to mean
        sls_x = l2_distance(X, X.mean())**2
        sls_y = l2_distance(y, y.mean())**2

        return float(1 - sls_y.sum() / sls_x.sum())


class PairwiseCorrelation(RegistrationScorer[FData, FData]):
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
        >>> from skfda.preprocessing.registration import (
        ...     LeastSquaresShiftRegistration,
        ... )
        >>> from skfda.datasets import make_sinusoidal_process
        >>> X = make_sinusoidal_process(error_std=0, random_state=0)

        Fit the registration procedure.

        >>> shift_registration = LeastSquaresShiftRegistration()
        >>> shift_registration.fit(X)
        LeastSquaresShiftRegistration(...)

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

    def __init__(self, eval_points: NDArrayFloat | None = None) -> None:
        self.eval_points = eval_points

    def score_function(self, X: FData, y: FData) -> float:
        """Compute the score of the transformation performed.

        Args:
            X (FData): Original functional data.
            y (FData): Functional data registered.

        Returns:
            float: Score of the transformation.

        """
        check_fdata_dimensions(
            X,
            dim_domain=1,
            dim_codomain=1,
        )
        check_fdata_dimensions(
            y,
            dim_domain=1,
            dim_codomain=1,
        )

        # Discretize functional data if needed
        X, y = _to_grid(X, y, eval_points=self.eval_points)

        # Compute correlation matrices with zeros in diagonal
        # corrcoefs computes the correlation between vector, without weights
        # due to the sample points
        X_corr = np.corrcoef(X.data_matrix[..., 0])
        np.fill_diagonal(X_corr, 0)

        y_corr = np.corrcoef(y.data_matrix[..., 0])
        np.fill_diagonal(y_corr, 0)

        return float(y_corr.sum() / X_corr.sum())
