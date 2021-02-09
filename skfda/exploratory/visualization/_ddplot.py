"""DD-Plot Module.

This module contains the necessary functions to construct the DD-Plot.
To do this depth is calculated for the two chosen distributions, and then
a scatter plot is created of this two variables.
"""

from typing import List, Optional, TypeVar

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ...exploratory.depth.multivariate import Depth
from ._utils import _get_figure_and_axes, _set_figure_layout_for_fdata

T = TypeVar('T')
S = TypeVar('S', Figure, Axes, List[Axes])


class DDPlot:
    """
    DDPlot visualization.

    Plot the depth of our fdata elements in two
    different distributions, one in each axis. It is useful to understand
    how our data is more related with one subset of data / distribution
    than another one.
    Args:
        fdata: functional data set that we want to examine.
        dist1: functional data set that represents the first distribution that
            we want to use to compute the depth (Depth X).
        dist2: functional data set that represents the second distribution that
            we want to use to compute the depth (Depth Y).
        depth_method: method that will be used to compute the depths of the
            data with respect to the distributions.
    """

    def __init__(
        self,
        fdata: T,
        dist1: T,
        dist2: T,
        depth_method: Depth[T],
    ) -> None:
        self.fdata = fdata
        self.dist1 = dist1
        self.dist2 = dist2
        self.depth_method = depth_method
        self.depth_method.fit(fdata)

    def plot(
        self,
        chart: Optional[S] = None,
        *,
        fig: Optional[Figure] = None,
        axes: List[Axes] = None,
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
        **kwargs,
    ) -> Figure:
        """
        Plot DDPlot graph.

        Plot the depth of our fdata elements in the two different
        distributions,one in each axis. It is useful to understand how
        our data is more related with one subset of data / distribution
        than another one.
        Args:
            chart (figure object, axe or list of axes, optional): figure over
                with the graphs are plotted or axis over where the graphs are
                plotted. If None and ax is also None, the figure is
                initialized.
            fig (figure object, optional): figure over with the graphs are
                plotted in case ax is not specified. If None and ax is also
                None, the figure is initialized.
            axes (list of axis objects, optional): axis where the graphs
                are plotted. If None, see param fig.
            n_rows (int, optional): designates the number of rows of the figure
                to plot the different dimensions of the image. Only specified
                if fig and ax are None.
            n_cols(int, optional): designates the number of columns of the
                figure to plot the different dimensions of the image. Only
                specified if fig and ax are None.
            kwargs: if dim_domain is 1, keyword arguments to be passed to the
                matplotlib.pyplot.plot function; if dim_domain is 2, keyword
                arguments to be passed to the matplotlib.pyplot.plot_surface
                function.
        Returns:
            fig (figure object): figure object in which the depths will be
            scattered.
        """
        margin = 0.025
        width_aux_line = 0.35
        color_aux_line = "gray"

        fig, axes = _get_figure_and_axes(chart, fig, axes)
        fig, axes = _set_figure_layout_for_fdata(
            self.fdata, fig, axes, n_rows, n_cols,
        )

        depth_dist1 = self.depth_method(
            self.fdata, distribution=self.dist1,
        )
        depth_dist2 = self.depth_method(
            self.fdata, distribution=self.dist2,
        )

        if self.fdata.dim_domain == 1:
            for i in range(self.fdata.dim_codomain):
                axes[i].scatter(
                    depth_dist1,
                    depth_dist2,
                    **kwargs,
                )

        # Set labels of graph
        fig.suptitle("DDPlot")
        for axe in axes:
            axe.set_xlabel("X depth")
            axe.set_ylabel("Y depth")
            axe.set_xlim(
                [
                    self.depth_method.min - margin,
                    self.depth_method.max + margin,
                ],
            )
            axe.set_ylim(
                [
                    self.depth_method.min - margin,
                    self.depth_method.max + margin,
                ],
            )
            axe.plot(
                [0, 1],
                linewidth=width_aux_line,
                color=color_aux_line,
            )

        return fig
