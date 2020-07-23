from skfda.representation import FDataBasis, FData
import numpy as np
import itertools
import scipy
from sklearn.utils import check_random_state


def hotelling_t2(fd1, fd2):
    r"""
        Calculates Hotelling's :math:`T^2` over two samples in
        :class:`skfda.representation.FData` objects with sizes :math:`n_1`
        and :math:`n_2`.

        .. math::
            T^2 = n(\mathbf{m}_1 - \mathbf{m}_2)^\top \mathbf{W}^{1/2}(
            \mathbf{W}^{1/2}\mathbf{K_{\operatorname{pooled}}} \mathbf{W}^{
            1/2})^+
            \mathbf{W}^{1/2} (\mathbf{m}_1 - \mathbf{m}_2),

        where :math:`(\cdot)^{+}` indicates the Moore-Penrose pseudo-inverse
        operator, :math:`n=n_1+n_2`, `W` is Gram matrix (identity in case of
        discretized data), :math:`\mathbf{m}_1, \mathbf{m}_2` are the
        means of each ample and :math:`\mathbf{K}_{\operatorname{pooled}}`
        matrix is defined as

        .. math::
            \mathbf{K}_{\operatorname{pooled}} :=
            \cfrac{n_1 - 1}{n_1 + n_2 - 2} \mathbf{K}_{n_1} +
            \cfrac{n_2 - 1}{n_1 + n_2 - 2} \mathbf{K}_{n_2},

        where :math:`\mathbf{K}_{n_1}`, :math:`\mathbf{K}_{n_2}` are the sample
        covariance matrices, computed with the basis coefficients or using
        the discrete representation, depending on the input.

        This statistic is defined in Pini, Stamm and Vantini[1].

        Args:
            fd1 (FData): Object with the first sample.
            fd2 (FData): Object containing second sample.

        Returns:
            The value of the statistic.

        Raises:
            TypeError.

        Examples:

            >>> from skfda.inference.hotelling import hotelling_t2
            >>> from skfda.representation import FDataGrid, basis

            >>> fd1 = FDataGrid([[1, 1, 1], [3, 3, 3]])
            >>> fd2 = FDataGrid([[3, 3, 3], [5, 5, 5]])
            >>> '%.2f' % hotelling_t2(fd1, fd2)
            '2.00'
            >>> fd1 = fd1.to_basis(basis.Fourier(n_basis=3))
            >>> fd2 = fd2.to_basis(basis.Fourier(n_basis=3))
            >>> '%.2f' % hotelling_t2(fd1, fd2)
            '2.00'

        References:
            [1] A. Pini, A. Stamm and S. Vantini, "Hotelling's t2 in
            separable hilbert spaces", *Jounal of Multivariate Analysis*,
            167 (2018), pp.284-305.

        """
    if not isinstance(fd1, FData):
        raise TypeError("Argument type must inherit FData.")

    if not isinstance(fd2, type(fd1)):
        raise TypeError("Both samples must be instances of the same type.")

    n1, n2 = fd1.n_samples, fd2.n_samples  # Size of each sample
    n = n1 + n2  # Size of full sample
    m = fd1.mean() - fd2.mean()  # Delta mean

    if isinstance(fd1, FDataBasis):
        if fd1.basis != fd2.basis:
            raise ValueError("Both FDataBasis objects must share the same "
                             "basis.")
        # When working on basis representation we use the coefficients
        m = m.coefficients[0]
        k1 = np.cov(fd1.coefficients, rowvar=False)
        k2 = np.cov(fd2.coefficients, rowvar=False)
        # If no weight matrix is passed, then we compute the Gram Matrix
        weights = fd1.basis.gram_matrix()
        weights = np.sqrt(weights)
    else:
        # Working with standard discretized data
        m = m.data_matrix[0, ..., 0]
        k1 = fd1.cov().data_matrix[0, ..., 0]
        k2 = fd2.cov().data_matrix[0, ..., 0]

    m = m.reshape((-1, 1))  # Reshaping the mean for a proper matrix product
    k_pool = ((n1 - 1) * k1 + (n2 - 1) * k2) / (n - 2)  # Combination of covs

    if isinstance(fd1, FDataBasis):
        # Product of pooled covariance with the weights and Moore-Penrose inv.
        k_inv = np.linalg.pinv(np.linalg.multi_dot([weights, k_pool, weights]))
        k_inv = weights.dot(k_inv).dot(weights)
    else:
        # If data is discrete no weights are needed
        k_inv = np.linalg.pinv(k_pool)

    return n1 * n2 / n * m.T.dot(k_inv).dot(m)[0][0]


def hotelling_test_ind(fd1, fd2, *, n_reps=None, random_state=None,
                       return_dist=False):
    r"""
    Calculate the :math:`T^2`-test for the means of two independent samples of
    functional data.

    This is a two-sided test for the null hypothesis that 2 independent samples
    have identical average (expected) values. This test assumes that the
    populations have identical variances by default.

    The p-value of the test is calculated using a permutation test over the
    statistic :func:`~skfda.inference.hotelling.hotelling_t2`. If a maximum
    number of repetitions of the algorithm is provided then the permutations
    tested are generated randomly.

    This procedure is from Pini, Stamm and Vantinni[1].

    Args:
        fd1,fd2 (FData): Samples of data. The FData objects must have the same
            type.

        n_reps (int, optional): Maximum number of repetitions to compute
            p-value. Default value is None.

        random_state (optional): Random state.

        return_dist (bool, optional): Flag to indicate if the function should
            return a numpy.array with the values of the statistic computed over
            each permutation.


    Returns:
        Value of the sample statistic, one tailed p-value and a collection of
        statistic values from permutations of the sample.

    Return type:
        (float, float, numpy.array)

    Raises:
        TypeError: In case of bad arguments.

    Examples:
        >>> from skfda.inference.hotelling import hotelling_t2
        >>> from skfda.representation import FDataGrid, basis
        >>> from numpy import printoptions

        >>> fd1 = FDataGrid([[1, 1, 1], [3, 3, 3]])
        >>> fd2 = FDataGrid([[3, 3, 3], [5, 5, 5]])
        >>> t2n, pval, dist = hotelling_test_ind(fd1, fd2, return_dist=True)
        >>> '%.2f' % t2n
        '2.00'
        >>> '%.2f' % pval
        '0.00'
        >>> with printoptions(precision=4):
        ...     print(dist)
        [ 2. 2. 0. 0. 2. 2.]

    References:
        [1] A. Pini, A. Stamm and S. Vantini, "Hotelling's t2 in
        separable hilbert spaces", *Jounal of Multivariate Analysis*,
        167 (2018), pp.284-305.
    """
    if not isinstance(fd1, FData):
        raise TypeError("Argument type must inherit FData.")

    if not isinstance(fd2, type(fd1)):
        raise TypeError("Both samples must be instances of the same type.")

    if n_reps is not None and n_reps < 1:
        raise ValueError("Number of repetitions must be positive.")

    n1, n2 = fd1.n_samples, fd2.n_samples
    t2_0 = hotelling_t2(fd1, fd2)
    n = n1 + n2
    sample = fd1.concatenate(fd2)
    indices = np.arange(n)

    if n_reps is not None:  # Computing n_reps random permutations
        random_state = check_random_state(random_state)
        dist = np.empty(n_reps)
        for i in range(n_reps):
            random_state.shuffle(indices)
            dist[i] = hotelling_t2(sample[indices[:n1]], sample[indices[n1:]])

    else:  # Full permutation test
        combinations = itertools.combinations(indices, n1)
        dist = np.empty(int(scipy.special.comb(n, n1)))
        for i, comb in enumerate(combinations):
            sample1_i = np.asarray(comb)  # Comb is a selection of n1 indices
            sample2_i = np.setdiff1d(indices, sample1_i)  # Remaining n2 ind.
            sample1, sample2 = sample[sample1_i], sample[sample2_i]
            dist[i] = hotelling_t2(sample1, sample2)

    p_value = np.sum(dist > t2_0) / len(dist)

    if return_dist:
        return t2_0, p_value, dist

    return t2_0, p_value
