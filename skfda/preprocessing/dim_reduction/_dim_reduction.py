

from numpy.linalg import norm
from sklearn.base import BaseEstimator, TransformerMixin

def vectorial_norm(fdatagrid, p=2):
    r"""Apply a vectorial norm to a multivariate function.

    Given a multivariate function :math:`f:\mathbb{R}^n\rightarrow \mathbb{R}^d`
    applies a vectorial norm :math:`\| \cdot \|` to produce a function
    :math:`\|f\|:\mathbb{R}^n\rightarrow \mathbb{R}`.

    For example, let :math:`f:\mathbb{R} \rightarrow \mathbb{R}^2` be
    :math:`f(t)=(f_1(t), f_2(t))` and :math:`\| \cdot \|_2` the euclidian norm.

    .. math::
        \|f\|_2(t) = \sqrt { |f_1(t)|^2 + |f_2(t)|^2 }

    In general if :math:`p \neq \pm \infty` and :math:`f:\mathbb{R}^n
    \rightarrow \mathbb{R}^d`

    .. math::
        \|f\|_p(x_1, ... x_n) = \left ( \sum_{k=1}^{d} |f_k(x_1, ..., x_n)|^p
        \right )^{(1/p)}

    Args:
        fdatagrid (:class:`FDatagrid`): Functional object to be transformed.
        p (int, optional): Exponent in the lp norm. If p is a number then
            it is applied sum(abs(x)**p)**(1./p), if p is inf then max(abs(x)),
            and if p is -inf it is applied min(abs(x)). See numpy.linalg.norm
            to more information. Defaults to 2.

    Returns:
        (:class:`FDatagrid`): FDatagrid with image dimension equal to 1.

    Examples:

        >>> from skfda.datasets import make_multimodal_samples
        >>> from skfda.preprocessing.dim_reduction import vectorial_norm

        First we will construct an example dataset with curves in
        :math:`\mathbb{R}^2`.

        >>> fd = make_multimodal_samples(ndim_image=2, random_state=0)
        >>> fd.ndim_image
        2

        We will apply the euclidean norm

        >>> fd = vectorial_norm(fd, p=2)
        >>> fd.ndim_image
        1

    """
    data_matrix = norm(fdatagrid.data_matrix, ord=p, axis=-1, keepdims=True)

    return fdatagrid.copy(data_matrix=data_matrix)

class VectorialNorm(BaseEstimator, TransformerMixin):
    r"""Sklearn transform version of vectorial_norm.

    Given a multivariate function :math:`f:\mathbb{R}^n\rightarrow \mathbb{R}^d`
    applies a vectorial norm :math:`\| \cdot \|` to produce a function
    :math:`\|f\|:\mathbb{R}^n\rightarrow \mathbb{R}`.

    For example, let :math:`f:\mathbb{R} \rightarrow \mathbb{R}^2` be
    :math:`f(t)=(f_1(t), f_2(t))` and :math:`\| \cdot \|_2` the euclidian norm.

    .. math::
        \|f\|_2(t) = \sqrt { |f_1(t)|^2 + |f_2(t)|^2 }

    In general if :math:`p \neq \pm \infty` and :math:`f:\mathbb{R}^n
    \rightarrow \mathbb{R}^d`

    .. math::
        \|f\|_p(x_1, ... x_n) = \left ( \sum_{k=1}^{d} |f_k(x_1, ..., x_n)|^p
        \right )^{(1/p)}

    Attributes:
        p (int, optional): Exponent in the lp norm. If p is a number then
            it is applied sum(abs(x)**p)**(1./p), if p is inf then max(abs(x)),
            and if p is -inf it is applied min(abs(x)). See numpy.linalg.norm
            to more information. Defaults to 2.

    """
    def __init__(self, p=2):
        r"""Initialize method.

        Args:
            p (int, optional): Exponent in the lp norm. If p is a number then
                it is applied sum(abs(x)**p)**(1./p), if p is inf then
                max(abs(x)), and if p is -inf it is applied min(abs(x)).
                See numpy.linalg.norm to more information. Defaults to 2.

        """
        self.p = p

    def fit(self, X, y=None):
        r"""This transformers does not need to be fitted, so nothing is done."""
        return self


    def transform(self, X):
        r"""Applies the transformation.

        Args:
            X (:class:`FDatagrid`): Functional object to be transformed.

        Returns:
            (:class:`FDatagrid`): FDatagrid with image dimension equal to 1.


        """

        return vectorial_norm(X, p=self.p)
