import abc
import numbers

import matplotlib.pyplot as plt

import numpy as np
import sklearn.gaussian_process.kernels as sklearn_kern

from ..exploratory.visualization._utils import _create_figure, _figure_to_svg


def _squared_norms(x, y):
    return ((x[np.newaxis, :, :] - y[:, np.newaxis, :]) ** 2).sum(2)


def _transform_to_2d(t):
    """Transform 1d arrays in column vectors."""
    t = np.asarray(t)

    dim = len(t.shape)
    assert dim <= 2

    if dim < 2:
        t = np.atleast_2d(t).T

    return t


def _execute_covariance(covariance, x, y):
    """Execute a covariance function.
    """
    x = _transform_to_2d(x)
    y = _transform_to_2d(y)

    if isinstance(covariance, numbers.Number):
        return covariance
    else:
        if callable(covariance):
            result = covariance(x, y)
        elif hasattr(covariance, "shape"):
            result = covariance
        else:
            # GPy kernel
            result = covariance.K(x, y)

        assert result.shape[0] == len(x)
        assert result.shape[1] == len(y)
        return result


class Covariance(abc.ABC):
    """Abstract class for covariance functions"""

    @abc.abstractmethod
    def __call__(self, x, y):
        pass

    def heatmap(self, limits=(-1, 1)):
        """
        Return a heatmap plot of the covariance function.

        """

        x = np.linspace(*limits, 1000)

        cov_matrix = self(x, x)

        fig = _create_figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(cov_matrix, extent=[limits[0], limits[1],
                                      limits[1], limits[0]])
        ax.set_title(f"Covariance function in [{limits[0]}, {limits[1]}]")

        return fig

    def _sample_trajectories_plot(self):
        from ..datasets import make_gaussian_process

        fd = make_gaussian_process(
            start=-1, n_samples=10, cov=self, random_state=0)
        fig = fd.plot()
        fig.axes[0].set_title("Sample trajectories")
        return fig

    def __repr__(self):

        params = ', '.join(f'{n}={getattr(self, n)}'
                           for n, _ in self._parameters)

        return (f"{self.__module__}.{type(self).__qualname__}("
                f"{params}"
                f")")

    def _latex_content(self):
        params = ''.join(fr'{l} &= {getattr(self, n)} \\'
                         for n, l in self._parameters)

        return (fr"{self._latex_formula} \\"
                r"\text{where:}"
                r"\begin{aligned}"
                fr"\qquad{params}"
                r"\end{aligned}")

    def _repr_latex_(self):
        return fr"\(\displaystyle {self._latex_content()}\)"

    def _repr_html_(self):
        fig = self.heatmap()
        heatmap = _figure_to_svg(fig)
        plt.close(fig)

        fig = self._sample_trajectories_plot()
        sample_trajectories = _figure_to_svg(fig)
        plt.close(fig)

        row_style = ''

        def column_style(percent, margin_top=0):
            return (f'style="display: inline-block; '
                    f'margin:0; '
                    f'margin-top: {margin_top}; '
                    f'width:{percent}%; '
                    f'height:auto;'
                    f'vertical-align: middle"')

        html = f"""
        <div {row_style}>
            <div {column_style(100, margin_top='25px')}>
            \\[{self._latex_content()}\\]
            </div>
        </div>
        <div {row_style}>
            <div {column_style(48)}>
            {sample_trajectories}
            </div>
            <div {column_style(48)}>
            {heatmap}
            </div>
        </div>
        """

        return html

    def to_sklearn(self):
        """Convert it to a sklearn kernel, if there is one"""
        raise NotImplementedError(f"{type(self).__name__} covariance not "
                                  f"implemented in scikit-learn")


class Brownian(Covariance):
    r"""
    Brownian covariance function.

    The covariance function is

    .. math::
        K(x, x') = \sigma^2 \frac{|x - \mathcal{O}| + |x' - \mathcal{O}|
        - |x - x'|}{2}

    where :math:`\sigma^2` is the variance at distance 1 from
    :math:`\mathcal{O}` and :math:`\mathcal{O}` is the origin point.
    If :math:`\mathcal{O} = 0` (the default) and we only
    consider positive values, the formula can be simplified as

    .. math::
        K(x, y) = \sigma^2 \min(x, y).

    Heatmap plot of the covariance function:

    .. jupyter-execute::

        from skfda.misc.covariances import Brownian
        import matplotlib.pyplot as plt

        Brownian().heatmap(limits=(0, 1))
        plt.show()

    Example of Gaussian process trajectories using this covariance:

    .. jupyter-execute::

        from skfda.misc.covariances import Brownian
        from skfda.datasets import make_gaussian_process
        import matplotlib.pyplot as plt

        gp = make_gaussian_process(
                n_samples=10, cov=Brownian(), random_state=0)
        gp.plot()
        plt.show()

    Default representation in a Jupyter notebook:

    .. jupyter-execute::

        from skfda.misc.covariances import Brownian

        Brownian()

    """
    _latex_formula = (r"K(x, x') = \sigma^2 \frac{|x - \mathcal{O}| + "
                      r"|x' - \mathcal{O}| - |x - x'|}{2}")

    _parameters = [("variance", r"\sigma^2"),
                   ("origin", r"\mathcal{O}")]

    def __init__(self, *, variance: float = 1., origin=0.):
        self.variance = variance
        self.origin = origin

    def __call__(self, x, y):
        x = _transform_to_2d(x) - self.origin
        y = _transform_to_2d(y) - self.origin

        sum_norms = np.add.outer(
            np.linalg.norm(x, axis=-1),
            np.linalg.norm(y, axis=-1))
        norm_sub = np.linalg.norm(
            x[:, np.newaxis, :] - y[np.newaxis, :, :], axis=-1)

        return self.variance * (sum_norms - norm_sub) / 2


class Linear(Covariance):
    r"""
    Linear covariance function.

    The covariance function is

    .. math::
        K(x, x') = \sigma^2 (x^T x' + c)

    where :math:`\sigma^2` is the scale of the variance and
    :math:`c` is the intercept.

    Heatmap plot of the covariance function:

    .. jupyter-execute::

        from skfda.misc.covariances import Linear
        import matplotlib.pyplot as plt

        Linear().heatmap(limits=(0, 1))
        plt.show()

    Example of Gaussian process trajectories using this covariance:

    .. jupyter-execute::

        from skfda.misc.covariances import Linear
        from skfda.datasets import make_gaussian_process
        import matplotlib.pyplot as plt

        gp = make_gaussian_process(
                n_samples=10, cov=Linear(), random_state=0)
        gp.plot()
        plt.show()

    Default representation in a Jupyter notebook:

    .. jupyter-execute::

        from skfda.misc.covariances import Linear

        Linear()

    """
    _latex_formula = r"K(x, x') = \sigma^2 (x^T x' + c)"

    _parameters = [("variance", r"\sigma^2"),
                   ("intercept", r"c")]

    def __init__(self, *, variance: float=1., intercept: float=0.):
        self.variance = variance
        self.intercept = intercept

    def __call__(self, x, y):
        x = _transform_to_2d(x)
        y = _transform_to_2d(y)

        return self.variance * (x @ y.T + self.intercept)

    def to_sklearn(self):
        """Convert it to a sklearn kernel, if there is one"""
        return (self.variance *
                (sklearn_kern.DotProduct(0) + self.intercept))


class Polynomial(Covariance):
    r"""
    Polynomial covariance function.

    The covariance function is

    .. math::
        K(x, x') = \sigma^2 (\alpha x^T x' + c)^d

    where :math:`\sigma^2` is the scale of the variance,
    :math:`\alpha` is the slope, :math:`d` the degree of the
    polynomial and :math:`c` is the intercept.

    Heatmap plot of the covariance function:

    .. jupyter-execute::

        from skfda.misc.covariances import Polynomial
        import matplotlib.pyplot as plt

        Polynomial().heatmap(limits=(0, 1))
        plt.show()

    Example of Gaussian process trajectories using this covariance:

    .. jupyter-execute::

        from skfda.misc.covariances import Polynomial
        from skfda.datasets import make_gaussian_process
        import matplotlib.pyplot as plt

        gp = make_gaussian_process(
                n_samples=10, cov=Polynomial(), random_state=0)
        gp.plot()
        plt.show()

    Default representation in a Jupyter notebook:

    .. jupyter-execute::

        from skfda.misc.covariances import Polynomial

        Polynomial()

    """
    _latex_formula = r"K(x, x') = \sigma^2 (\alpha x^T x' + c)^d"

    _parameters = [("variance", r"\sigma^2"),
                   ("intercept", r"c"),
                   ("slope", r"\alpha"),
                   ("degree", r"d")]

    def __init__(self, *, variance: float=1., intercept: float=0.,
                 slope: float=1., degree: float=2.):
        self.variance = variance
        self.intercept = intercept
        self.slope = slope
        self.degree = degree

    def __call__(self, x, y):
        x = _transform_to_2d(x)
        y = _transform_to_2d(y)

        return self.variance * (self.slope * x @ y.T
                                + self.intercept) ** self.degree

    def to_sklearn(self):
        """Convert it to a sklearn kernel, if there is one"""
        return (self.variance *
                (self.slope *
                 sklearn_kern.DotProduct(0) + + self.intercept)
                ** self.degree)


class Gaussian(Covariance):
    r"""
    Gaussian covariance function.

    The covariance function is

    .. math::
        K(x, x') = \sigma^2 \exp\left(-\frac{||x - x'||^2}{2l^2}\right)

    where :math:`\sigma^2` is the variance and :math:`l` is the length scale.

    Heatmap plot of the covariance function:

    .. jupyter-execute::

        from skfda.misc.covariances import Gaussian
        import matplotlib.pyplot as plt

        Gaussian().heatmap(limits=(0, 1))
        plt.show()

    Example of Gaussian process trajectories using this covariance:

    .. jupyter-execute::

        from skfda.misc.covariances import Gaussian
        from skfda.datasets import make_gaussian_process
        import matplotlib.pyplot as plt

        gp = make_gaussian_process(
                n_samples=10, cov=Gaussian(), random_state=0)
        gp.plot()
        plt.show()

    Default representation in a Jupyter notebook:

    .. jupyter-execute::

        from skfda.misc.covariances import Gaussian

        Gaussian()

    """
    _latex_formula = (r"K(x, x') = \sigma^2 \exp\left(-\frac{\|x - x'\|^2}{2l^2}"
                      r"\right)")

    _parameters = [("variance", r"\sigma^2"),
                   ("length_scale", r"l")]

    def __init__(self, *, variance: float=1., length_scale: float=1.):
        self.variance = variance
        self.length_scale = length_scale

    def __call__(self, x, y):
        x = _transform_to_2d(x)
        y = _transform_to_2d(y)

        x_y = _squared_norms(x, y)

        return self.variance * np.exp(-x_y / (2 * self.length_scale ** 2))

    def to_sklearn(self):
        """Convert it to a sklearn kernel, if there is one"""
        return (self.variance *
                sklearn_kern.RBF(length_scale=self.length_scale))


class Exponential(Covariance):
    r"""
    Exponential covariance function.

    The covariance function is

    .. math::
        K(x, x') = \sigma^2 \exp\left(-\frac{\|x - x'\|}{l}\right)

    where :math:`\sigma^2` is the variance and :math:`l` is the length scale.

    Heatmap plot of the covariance function:

    .. jupyter-execute::

        from skfda.misc.covariances import Exponential
        import matplotlib.pyplot as plt

        Exponential().heatmap(limits=(0, 1))
        plt.show()

    Example of Gaussian process trajectories using this covariance:

    .. jupyter-execute::

        from skfda.misc.covariances import Exponential
        from skfda.datasets import make_gaussian_process
        import matplotlib.pyplot as plt

        gp = make_gaussian_process(
                n_samples=10, cov=Exponential(), random_state=0)
        gp.plot()
        plt.show()

    Default representation in a Jupyter notebook:

    .. jupyter-execute::

        from skfda.misc.covariances import Exponential

        Exponential()

    """
    _latex_formula = (r"K(x, x') = \sigma^2 \exp\left(-\frac{||x - x'||}{l}"
                      r"\right)")

    _parameters = [("variance", r"\sigma^2"),
                   ("length_scale", r"l")]

    def __init__(self, *, variance: float=1., length_scale: float=1.):
        self.variance = variance
        self.length_scale = length_scale

    def __call__(self, x, y):
        x = _transform_to_2d(x)
        y = _transform_to_2d(y)

        x_y = _squared_norms(x, y)
        return self.variance * np.exp(-np.sqrt(x_y) / (self.length_scale))

    def to_sklearn(self):
        """Convert it to a sklearn kernel, if there is one"""
        return (self.variance *
                sklearn_kern.Matern(length_scale=self.length_scale, nu=0.5))


class WhiteNoise(Covariance):
    r"""
    Gaussian covariance function.

    The covariance function is

    .. math::
        K(x, x')= \begin{cases}
                    \sigma^2, \quad x = x' \\
                    0, \quad x \neq x'\\
                  \end{cases}

    where :math:`\sigma^2` is the variance.

    Heatmap plot of the covariance function:

    .. jupyter-execute::

        from skfda.misc.covariances import WhiteNoise
        import matplotlib.pyplot as plt

        WhiteNoise().heatmap(limits=(0, 1))
        plt.show()

    Example of Gaussian process trajectories using this covariance:

    .. jupyter-execute::

        from skfda.misc.covariances import WhiteNoise
        from skfda.datasets import make_gaussian_process
        import matplotlib.pyplot as plt

        gp = make_gaussian_process(
                n_samples=10, cov=WhiteNoise(), random_state=0)
        gp.plot()
        plt.show()

    Default representation in a Jupyter notebook:

    .. jupyter-execute::

        from skfda.misc.covariances import WhiteNoise

        WhiteNoise()

    """

    _latex_formula = (r"K(x, x')= \begin{cases} \sigma^2, \quad x = x' \\"
                      r"0, \quad x \neq x'\\ \end{cases}")

    _parameters = [("variance", r"\sigma^2")]

    def __init__(self, *, variance: float = 1.):
        self.variance = variance

    def __call__(self, x, y):
        x = _transform_to_2d(x)
        return self.variance * np.eye(x.shape[0])

    def to_sklearn(self):
        """Convert it to a sklearn kernel, if there is one"""
        return sklearn_kern.WhiteKernel(noise_level=self.variance)
