"""Outliergram Module.

This module contains the methods used to plot shapes in order to detect
shape outliers in our dataset. In order to do this, we plot the
Modified Band Depth and Modified Epigraph Index, that will help us detect
these outliers. The motivation of the method is that it is easy to find
magnitude outliers, but there is a necessity of capturing this other type.
"""

from typing import Optional, Sequence, Union

import numpy as np
import scipy.integrate as integrate
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import rankdata

from ... import FDataGrid
from ..depth._depth import ModifiedBandDepth
from ._baseplot import BasePlot
from ._utils import _get_figure_and_axes, _set_figure_layout_for_fdata


class Outliergram(BasePlot):
    """
    Outliergram method of visualization.

    Plots the Modified Band Depth (MBD) on the Y axis and the Modified
    Epigraph Index (MEI) on the X axis. This points will create the form of
    a parabola. The shape outliers will be the points that appear far from
    this curve.
    Args:
        fdata: functional data set that we want to examine.
        chart: figure over with the graphs are plotted or axis over
            where the graphs are plotted. If None and ax is also
            None, the figure is initialized.
        fig: figure over with the graphs are plotted in case ax is not
            specified. If None and ax is also None, the figure is
            initialized.
        axes: axis where the graphs are plotted. If None, see param fig.
        n_rows: designates the number of rows of the figure
                to plot the different dimensions of the image. Only specified
                if fig and ax are None.
        n_cols: designates the number of columns of the
                figure to plot the different dimensions of the image. Only
                specified if fig and ax are None.
    Attributes:
        mbd: result of the calculation of the Modified Band Depth on our
            dataset. Represents the mean time a curve stays between other pair
            of curves, being a good measure of centrality.
        mei: result of the calculation of the Modified Epigraph Index on our
            dataset. Represents the mean time a curve stays below other curve.
    References:
        López-Pintado S.,  Romo J.. (2011). A half-region depth for functional
        data, Computational Statistics & Data Analysis, volume 55
        (page 1679-1695).
        Arribas-Gil A., Romo J.. Shape outlier detection and visualization for
        functional data: the outliergram
        https://academic.oup.com/biostatistics/article/15/4/603/266279
    """

    def __init__(
        self,
        fdata: FDataGrid,
        chart: Union[Figure, Axes, None] = None,
        *,
        fig: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
        **kwargs,
    ) -> None:
        BasePlot.__init__(self)
        self.fdata = fdata
        self.depth = ModifiedBandDepth()
        self.depth.fit(fdata)
        self.mbd = self.depth(fdata)
        self.mei = self.modified_epigraph_index_list()
        if len(self.mbd) != len(self.mei):
            raise ValueError(
                "The size of mbd and mei should be the same.",
            )
        self.n = self.mbd.size
        distances, parable = self._compute_distances()
        self.distances = distances
        mei_ordered, parable = (
            list(el) for el in zip(*sorted(zip(self.mei, parable)))
        )
        self.parable = parable
        self.mei_ordered = mei_ordered
        self._compute_outliergram()

        self._set_figure_and_axes(chart, fig, axes, n_rows, n_cols)

    def plot(
        self,
    ) -> Figure:
        """
        Plot Outliergram.

        Plots the Modified Band Depth (MBD) on the Y axis and the Modified
        Epigraph Index (MEI) on the X axis. This points will create the form of
        a parabola. The shape outliers will be the points that appear far from
        this curve.
        Returns:
            fig: figure object in which the depths will be
            scattered.
        """
        self.artists = np.zeros(self.n_samples(), dtype=Artist)
        self.axScatter = self.axes[0]

        for i in range(self.mei.size):
            self.artists[i] = self.axScatter.scatter(
                self.mei[i],
                self.mbd[i],
                picker=2,
            )

        self.axScatter.plot(
            self.mei_ordered,
            self.parable,
        )

        self.axScatter.plot(
            self.mei_ordered,
            self.shifted_parable,
            linestyle='dashed',
        )

        # Set labels of graph
        if self.fdata.dataset_name is not None:
            self.axScatter.set_title(self.fdata.dataset_name)

        self.axScatter.set_xlabel("MEI")
        self.axScatter.set_ylabel("MBD")
        self.axScatter.set_xlim([0, 1])
        self.axScatter.set_ylim([
            self.depth.min,
            self.depth.max,
        ])

        return self.fig

    def modified_epigraph_index_list(self) -> np.ndarray:
        """
        Calculate the Modified Epigraph Index of a FData.

        The MEI represents the mean time a curve stays below other curve.
        In this case we will calculate the MEI for each curve in relation
        with all the other curves of our dataset.
        """
        interval_len = (
            self.fdata.domain_range[0][1]
            - self.fdata.domain_range[0][0]
        )

        # Array containing at each point the number of curves
        # are above it.
        num_functions_above = rankdata(
            -self.fdata.data_matrix,
            method='max',
            axis=0,
        ) - 1

        integrand = integrate.simps(
            num_functions_above,
            x=self.fdata.grid_points[0],
            axis=1,
        )

        integrand /= (interval_len * self.fdata.n_samples)

        return integrand.flatten()

    def _compute_distances(self) -> np.ndarray:
        """
        Calculate the distances of each point towards the parabola.

        The distances can be calculated with function:
            d_i = a_0 + a_1* mei_i + n^2* a_2* mei_i^2 - mb_i.
        """
        a_0 = -2 / (self.n * (self.n - 1))
        a_1 = (2 * (self.n + 1)) / (self.n - 1)
        a_2 = a_0

        parable = (
            a_0 + a_1 * self.mei + pow(self.n, 2) * a_2 * pow(self.mei, 2)
        )
        distances = parable - self.mbd

        return distances, parable

    def _compute_outliergram(self) -> None:
        """Compute the parabola under which the outliers lie."""
        first_quartile = np.percentile(self.distances, 25)  # noqa: WPS432
        third_quartile = np.percentile(self.distances, 75)  # noqa: WPS432
        iqr = third_quartile - first_quartile
        self.shifted_parable = self.parable - (third_quartile + iqr)

    def n_samples(self) -> int:
        """Get the number of instances that will be used for interactivity."""
        return self.fdata.n_samples

    def _set_figure_and_axes(
        self,
        chart: Union[Figure, Axes, None] = None,
        fig: Optional[Figure] = None,
        axes: Union[Axes, Sequence[Axes], None] = None,
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
    ) -> None:
        """
        Initialize the axes and fig of the plot.

        Args:
        chart: figure over with the graphs are plotted or axis over
            where the graphs are plotted. If None and ax is also
            None, the figure is initialized.
        fig: figure over with the graphs are plotted in case ax is not
            specified. If None and ax is also None, the figure is
            initialized.
        axes: axis where the graphs are plotted. If None, see param fig.
        n_rows: designates the number of rows of the figure
            to plot the different dimensions of the image. Only specified
            if fig and ax are None.
        n_cols: designates the number of columns of the
            figure to plot the different dimensions of the image. Only
            specified if fig and ax are None.
        """
        fig, axes = _get_figure_and_axes(chart, fig, axes)
        fig, axes = _set_figure_layout_for_fdata(
            fdata=self.fdata,
            fig=fig,
            axes=axes,
            n_rows=n_rows,
            n_cols=n_cols,
        )
        self.fig = fig
        self.axes = axes
