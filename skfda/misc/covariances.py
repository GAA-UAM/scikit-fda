from __future__ import annotations

import abc
from typing import Callable, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import sklearn.gaussian_process.kernels as sklearn_kern
from matplotlib.figure import Figure
from scipy.special import gamma, kv

from ..typing._numpy import ArrayLike, NDArrayFloat


def _squared_norms(x: NDArrayFloat, y: NDArrayFloat) -> NDArrayFloat:
    return (  # type: ignore[no-any-return]
        (x[:, np.newaxis, :] - y[np.newaxis, :, :]) ** 2
    ).sum(2)


CovarianceLike = Union[
    float,
    NDArrayFloat,
    Callable[[ArrayLike, ArrayLike], NDArrayFloat],
]


def _transform_to_2d(t: ArrayLike) -> NDArrayFloat:
    """Transform 1d arrays in column vectors."""
    t = np.asfarray(t)

    dim = len(t.shape)
    assert dim <= 2

    if dim < 2:
        t = np.atleast_2d(t).T

    return t


def _execute_covariance(
    covariance: CovarianceLike,
    x: ArrayLike,
    y: ArrayLike,
) -> NDArrayFloat:
    """Execute a covariance function."""
    x = _transform_to_2d(x)
    y = _transform_to_2d(y)

    if isinstance(covariance, (int, float)):
        return np.array(covariance)
    else:
        if callable(covariance):
            result = covariance(x, y)
        elif isinstance(covariance, np.ndarray):
            result = covariance
        else:
            # GPy kernel
            result = covariance.K(x, y)

        assert result.shape[0] == len(x)
        assert result.shape[1] == len(y)
        return result


class Covariance(abc.ABC):
    """Abstract class for covariance functions."""

    _parameters_str: Sequence[Tuple[str, str]]
    _latex_formula: str

    @abc.abstractmethod
    def __call__(self, x: ArrayLike, y: ArrayLike) -> NDArrayFloat:
        pass

    def heatmap(self, limits: Tuple[float, float] = (-1, 1)) -> Figure:
        """Return a heatmap plot of the covariance function."""
        from ..exploratory.visualization._utils import _create_figure

        x = np.linspace(*limits, 1000)

        cov_matrix = self(x, x)

        fig = _create_figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(
            cov_matrix,
            extent=[limits[0], limits[1], limits[1], limits[0]],
        )
        ax.set_title(f"Covariance function in [{limits[0]}, {limits[1]}]")

        return fig

    def _sample_trajectories_plot(self) -> Figure:
        from ..datasets import make_gaussian_process

        fd = make_gaussian_process(
            start=-1,
            n_samples=10,
            cov=self,
            random_state=0,
        )
        fig = fd.plot()
        fig.axes[0].set_title("Sample trajectories")
        return fig

    def __repr__(self) -> str:

        params_str = ', '.join(
            f'{n}={getattr(self, n)}' for n, _ in self._parameters_str
        )

        return (
            f"{self.__module__}.{type(self).__qualname__}("
            f"{params_str}"
            f")"
        )

    def _latex_content(self) -> str:
        params_str = ''.join(
            fr'{l} &= {getattr(self, n)} \\' for n, l in self._parameters_str
        )

        return (
            fr"{self._latex_formula} \\"
            r"\text{where:}"
            r"\begin{aligned}"
            fr"\qquad{params_str}"
            r"\end{aligned}"
        )

    def _repr_latex_(self) -> str:
        return fr"\(\displaystyle {self._latex_content()}\)"

    def _repr_html_(self) -> str:
        from ..exploratory.visualization._utils import _figure_to_svg

        fig = self.heatmap()
        heatmap = _figure_to_svg(fig)
        plt.close(fig)

        fig = self._sample_trajectories_plot()
        sample_trajectories = _figure_to_svg(fig)
        plt.close(fig)

        row_style = ''

        def column_style(percent: float, margin_top: str = "0") -> str:
            return (
                f'style="display: inline-block; '
                f'margin:0; '
                f'margin-top: {margin_top}; '
                f'width:{percent}%; '
                f'height:auto;'
                f'vertical-align: middle"'
            )

        return fr"""
        <div {row_style}>
            <div {column_style(100, margin_top='25px')}>
            \[{self._latex_content()}\]
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

    def to_sklearn(self) -> sklearn_kern.Kernel:
        """Convert it to a sklearn kernel, if there is one."""
        raise NotImplementedError(
            f"{type(self).__name__} covariance not "
            f"implemented in scikit-learn",
        )


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
    _latex_formula = (
        r"K(x, x') = \sigma^2 \frac{|x - \mathcal{O}| + "
        r"|x' - \mathcal{O}| - |x - x'|}{2}"
    )

    _parameters_str = [
        ("variance", r"\sigma^2"),
        ("origin", r"\mathcal{O}"),
    ]

    def __init__(self, *, variance: float = 1, origin: float = 0) -> None:
        self.variance = variance
        self.origin = origin

    def __call__(self, x: ArrayLike, y: ArrayLike) -> NDArrayFloat:
        x = _transform_to_2d(x) - self.origin
        y = _transform_to_2d(y) - self.origin

        sum_norms = np.add.outer(
            np.linalg.norm(x, axis=-1),
            np.linalg.norm(y, axis=-1),
        )
        norm_sub = np.linalg.norm(
            x[:, np.newaxis, :] - y[np.newaxis, :, :],
            axis=-1,
        )

        return (  # type: ignore[no-any-return]
            self.variance * (sum_norms - norm_sub) / 2
        )


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

    _parameters_str = [
        ("variance", r"\sigma^2"),
        ("intercept", "c"),
    ]

    def __init__(self, *, variance: float = 1, intercept: float = 0) -> None:
        self.variance = variance
        self.intercept = intercept

    def __call__(self, x: ArrayLike, y: ArrayLike) -> NDArrayFloat:
        x = _transform_to_2d(x)
        y = _transform_to_2d(y)

        return self.variance * (x @ y.T + self.intercept)

    def to_sklearn(self) -> sklearn_kern.Kernel:
        return (
            self.variance
            * (sklearn_kern.DotProduct(0) + self.intercept)
        )


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

    _parameters_str = [
        ("variance", r"\sigma^2"),
        ("intercept", "c"),
        ("slope", r"\alpha"),
        ("degree", "d"),
    ]

    def __init__(
        self,
        *,
        variance: float = 1,
        intercept: float = 0,
        slope: float = 1,
        degree: float = 2,
    ) -> None:
        self.variance = variance
        self.intercept = intercept
        self.slope = slope
        self.degree = degree

    def __call__(self, x: ArrayLike, y: ArrayLike) -> NDArrayFloat:
        x = _transform_to_2d(x)
        y = _transform_to_2d(y)

        return (
            self.variance
            * (self.slope * x @ y.T + self.intercept) ** self.degree
        )

    def to_sklearn(self) -> sklearn_kern.Kernel:
        return (
            self.variance
            * (self.slope * sklearn_kern.DotProduct(0) + self.intercept)
            ** self.degree
        )


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

    _latex_formula = (
        r"K(x, x') = \sigma^2 \exp\left(-\frac{\|x - x'\|^2}{2l^2}"
        r"\right)"
    )

    _parameters_str = [
        ("variance", r"\sigma^2"),
        ("length_scale", "l"),
    ]

    def __init__(self, *, variance: float = 1, length_scale: float = 1):
        self.variance = variance
        self.length_scale = length_scale

    def __call__(self, x: ArrayLike, y: ArrayLike) -> NDArrayFloat:
        x = _transform_to_2d(x)
        y = _transform_to_2d(y)

        x_y = _squared_norms(x, y)

        return self.variance * np.exp(-x_y / (2 * self.length_scale ** 2))

    def to_sklearn(self) -> sklearn_kern.Kernel:
        return (
            self.variance * sklearn_kern.RBF(length_scale=self.length_scale)
        )


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

    _latex_formula = (
        r"K(x, x') = \sigma^2 \exp\left(-\frac{||x - x'||}{l}"
        r"\right)"
    )

    _parameters_str = [
        ("variance", r"\sigma^2"),
        ("length_scale", "l"),
    ]

    def __init__(
        self,
        *,
        variance: float = 1,
        length_scale: float = 1,
    ) -> None:
        self.variance = variance
        self.length_scale = length_scale

    def __call__(self, x: ArrayLike, y: ArrayLike) -> NDArrayFloat:
        x = _transform_to_2d(x)
        y = _transform_to_2d(y)

        x_y = _squared_norms(x, y)
        return self.variance * np.exp(-np.sqrt(x_y) / (self.length_scale))

    def to_sklearn(self) -> sklearn_kern.Kernel:
        return (
            self.variance
            * sklearn_kern.Matern(length_scale=self.length_scale, nu=0.5)
        )


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

    _latex_formula = (
        r"K(x, x')= \begin{cases} \sigma^2, \quad x = x' \\"
        r"0, \quad x \neq x'\\ \end{cases}"
    )

    _parameters_str = [("variance", r"\sigma^2")]

    def __init__(self, *, variance: float = 1):
        self.variance = variance

    def __call__(self, x: ArrayLike, y: ArrayLike) -> NDArrayFloat:
        x = _transform_to_2d(x)
        return self.variance * np.eye(x.shape[0])

    def to_sklearn(self) -> sklearn_kern.Kernel:
        return sklearn_kern.WhiteKernel(noise_level=self.variance)


class Matern(Covariance):
    r"""
    Matérn covariance function.

    The covariance function is

    .. math::
        K(x, x') = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)}
        \left( \frac{\sqrt{2\nu}|x - x'|}{l} \right)^{\nu}
        K_{\nu}\left( \frac{\sqrt{2\nu}|x - x'|}{l} \right)

    where :math:`\sigma^2` is the variance, :math:`l` is the length scale
    and :math:`\nu` controls the smoothness of the related Gaussian process.
    The trajectories of a Gaussian process with Matérn covariance is 
    :math:`\lceil \nu \rceil - 1` times differentiable.


    Heatmap plot of the covariance function:

    .. jupyter-execute::

        from skfda.misc.covariances import Matern
        import matplotlib.pyplot as plt

        Matern().heatmap(limits=(0, 1))
        plt.show()

    Example of Gaussian process trajectories using this covariance:

    .. jupyter-execute::

        from skfda.misc.covariances import Matern
        from skfda.datasets import make_gaussian_process
        import matplotlib.pyplot as plt

        gp = make_gaussian_process(
                n_samples=10, cov=Matern(), random_state=0)
        gp.plot()
        plt.show()

    Default representation in a Jupyter notebook:

    .. jupyter-execute::

        from skfda.misc.covariances import Matern

        Matern()

    """
    _latex_formula = (
        r"K(x, x') = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)}"
        r"\left( \frac{\sqrt{2\nu}|x - x'|}{l} \right)^{\nu}"
        r"K_{\nu}\left( \frac{\sqrt{2\nu}|x - x'|}{l} \right)"
    )

    _parameters_str = [
        ("variance", r"\sigma^2"),
        ("length_scale", "l"),
        ("nu", r"\nu"),
    ]

    def __init__(
        self,
        *,
        variance: float = 1,
        length_scale: float = 1,
        nu: float = 1.5,
    ) -> None:
        self.variance = variance
        self.length_scale = length_scale
        self.nu = nu

    def __call__(self, x: ArrayLike, y: ArrayLike) -> NDArrayFloat:
        x = _transform_to_2d(x)
        y = _transform_to_2d(y)

        x_y_squared = _squared_norms(x, y)
        x_y = np.sqrt(x_y_squared)

        p = self.nu - 0.5
        if p.is_integer():
            # Formula for half-integers
            p = int(p)
            body = np.sqrt(2 * p + 1) * x_y / self.length_scale
            exponential = np.exp(-body)
            power_list = np.full(shape=(p,) + body.shape, fill_value=2 * body)
            power_list = np.cumprod(power_list, axis=0)
            power_list = np.concatenate(
                (power_list[::-1], np.asarray([np.ones_like(body)])),
            )
            power_list = np.moveaxis(power_list, 0, -1)
            numerator = np.cumprod(np.arange(p, 0, -1))
            numerator = np.concatenate(([1], numerator))
            denom1 = np.cumprod(np.arange(2 * p, p, -1))
            denom1 = np.concatenate((denom1[::-1], [1]))
            denom2 = np.cumprod(np.arange(1, p + 1))
            denom2 = np.concatenate(([1], denom2))

            sum_terms = power_list * numerator / (denom1 * denom2)
            return (  # type: ignore[no-any-return]
                self.variance * exponential * np.sum(sum_terms, axis=-1)
            )
        elif self.nu == np.inf:
            return (
                self.variance * np.exp(
                    -x_y_squared / (2 * self.length_scale ** 2),
                )
            )
        else:
            # General formula
            scaling = 2**(1 - self.nu) / gamma(self.nu)
            body = np.sqrt(2 * self.nu) * x_y / self.length_scale
            power = body**self.nu
            bessel = kv(self.nu, body)

            with np.errstate(invalid='ignore'):
                eval_cov = self.variance * scaling * power * bessel

            # Values with nan are where the distance is 0
            return np.nan_to_num(  # type: ignore[no-any-return]
                eval_cov,
                nan=self.variance,
            )

    def to_sklearn(self) -> sklearn_kern.Kernel:
        return (
            self.variance
            * sklearn_kern.Matern(length_scale=self.length_scale, nu=self.nu)
        )
