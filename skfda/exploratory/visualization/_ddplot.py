"""DD-Plot Module.

This module contains the necessary functions to construct the DD-Plot.
To do this depth is calculated for the two chosen distributions, and then
a scatter plot is created of this two variables.
"""

from typing import Optional, TypeVar, Union

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ...exploratory.depth.multivariate import Depth
from ...representation._functional_data import FData
from ._baseplot import BasePlot

T = TypeVar('T', bound=FData)


class DDPlot(BasePlot):
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
        chart: figure over with the graphs are plotted or axis over
            where the graphs are plotted. If None and ax is also
            None, the figure is initialized.
        fig: figure over with the graphs are plotted in case ax is not
            specified. If None and ax is also None, the figure is
            initialized.
        axes: axis where the graphs are plotted. If None, see param fig.
    Attributes:
        depth_dist1: result of the calculation of the depth_method into our
            first distribution (dist1).
        depth_dist2: result of the calculation of the depth_method into our
            second distribution (dist2).
    """

    def __init__(
        self,
        fdata: T,
        dist1: T,
        dist2: T,
        chart: Union[Figure, Axes, None] = None,
        *,
        depth_method: Depth[T],
        fig: Optional[Figure] = None,
        axes: Optional[Axes] = None,
    ) -> None:
        BasePlot.__init__(
            self,
            chart,
            fig=fig,
            axes=axes,
        )
        self.fdata = fdata
        self.depth_method = depth_method
        self.depth_method.fit(fdata)
        self.depth_dist1 = self.depth_method(
            self.fdata,
            distribution=dist1,
        )
        self.depth_dist2 = self.depth_method(
            self.fdata,
            distribution=dist2,
        )

    @property
    def n_samples(self) -> int:
        return self.fdata.n_samples

    def _plot(
        self,
        fig: Figure,
        axes: Axes,
    ) -> None:
        """
        Plot DDPlot graph.

        Plot the depth of our fdata elements in the two different
        distributions,one in each axis. It is useful to understand how
        our data is more related with one subset of data / distribution
        than another one.
        Returns:
            fig (figure object): figure object in which the depths will be
            scattered.
        """
        self.artists = np.zeros(
            (self.n_samples, 1),
            dtype=Artist,
        )
        margin = 0.025
        width_aux_line = 0.35
        color_aux_line = "gray"

        ax = axes[0]

        for i, (d1, d2) in enumerate(zip(self.depth_dist1, self.depth_dist2)):
            self.artists[i, 0] = ax.scatter(
                d1,
                d2,
                picker=True,
                pickradius=2,
            )

        # Set labels of graph
        if self.fdata.dataset_name is not None:
            ax.set_title(self.fdata.dataset_name)
        ax.set_xlabel("X depth")
        ax.set_ylabel("Y depth")
        ax.set_xlim(
            [
                self.depth_method.min - margin,
                self.depth_method.max + margin,
            ],
        )
        ax.set_ylim(
            [
                self.depth_method.min - margin,
                self.depth_method.max + margin,
            ],
        )
        ax.plot(
            [0, 1],
            linewidth=width_aux_line,
            color=color_aux_line,
        )
