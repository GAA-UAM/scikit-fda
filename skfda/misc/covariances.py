import abc
import numbers

import matplotlib

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

    def heatmap(self):
        x = np.linspace(-1, 1, 1000)

        cov_matrix = self(x, x)

        fig = _create_figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(cov_matrix, extent=[-1, 1, 1, -1])
        ax.set_title("Covariance function in [-1, 1]")

        return fig

    def _sample_trajectories_plot(self):
        from ..datasets import make_gaussian_process

        fd = make_gaussian_process(start=-1, cov=self)
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

        fig = self._sample_trajectories_plot()
        sample_trajectories = _figure_to_svg(fig)

        row_style = 'style="position:relative; display:table-row"'

        def column_style(percent):
            return (f'style="width: {percent}%; display: table-cell; '
                    f'vertical-align: middle"')

        html = f"""
        <div {row_style}>
            <div {column_style(50)}>
            \\[{self._latex_content()}\\]
            </div>
        </div>
        <div {row_style}>
            <div {column_style(50)}>
            {sample_trajectories}
            </div>
            <div {column_style(50)}>
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
    """Brownian covariance function."""

    _latex_formula = (r"K(x, y) = \sigma^2 \frac{|x - \mathcal{O}| + "
                      r"|y - \mathcal{O}| - |x-y|}{2}")

    _parameters = [("variance", r"\sigma^2"),
                   ("origin", r"\mathcal{O}")]

    def __init__(self, *, variance: float = 1., origin: float = 0.):
        self.variance = variance
        self.origin = origin

    def __call__(self, x, y):
        x = _transform_to_2d(x) - self.origin
        y = _transform_to_2d(y) - self.origin

        return self.variance * (np.abs(x) + np.abs(y.T) - np.abs(x - y.T)) / 2


class Linear(Covariance):
    """Linear covariance function."""

    _latex_formula = r"K(x, y) = \sigma^2 (x^T y + c)"

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
    """Polynomial covariance function."""

    _latex_formula = r"K(x, y) = \sigma^2 (\alpha x^T y + c)^d"

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
    """Gaussian covariance function."""

    _latex_formula = (r"K(x, y) = \sigma^2 \exp\left(\frac{||x - y||^2}{2l^2}"
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
    """Exponential covariance function."""

    _latex_formula = (r"K(x, y) = \sigma^2 \exp\left(\frac{||x - y||}{l}"
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
