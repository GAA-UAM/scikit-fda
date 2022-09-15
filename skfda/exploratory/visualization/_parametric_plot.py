"""Parametric Plot Module.

This module contains the functionality in charge of plotting
two different functions as coordinates, this can be done giving
one FData, with domain 1 and codomain 2, or giving two FData, both
of them with domain 1 and codomain 1.
"""
from __future__ import annotations

from typing import Dict, Sequence, TypeVar

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ...representation import FData
from ._baseplot import BasePlot
from ._utils import ColorLike
from .representation import Indexable, _get_color_info

K = TypeVar('K', contravariant=True)


class ParametricPlot(BasePlot):
    """
    Parametric Plot visualization.

    This class contains the functionality in charge of plotting
    two different functions as coordinates, this can be done giving
    one FData, with domain 1 and codomain 2, or giving two FData, both
    of them with domain 1 and codomain 1.

    Args:
        fdata1: functional data set that we will use for the graph. If it has
            a dim_codomain = 1, the fdata2 will be needed.
        fdata2: optional functional data set, that will be needed if the fdata1
            has dim_codomain = 1.
        chart: figure over with the graphs are plotted or axis over
            where the graphs are plotted. If None and ax is also
            None, the figure is initialized.
        fig: figure over with the graphs are plotted in case ax is not
            specified. If None and ax is also None, the figure is
            initialized.
        ax: axis where the graphs are plotted. If None, see param fig.
    """

    def __init__(
        self,
        fdata1: FData,
        fdata2: FData | None = None,
        chart: Figure | Axes | None = None,
        *,
        fig: Figure | None = None,
        axes: Axes | None = None,
        group: Sequence[K] | None = None,
        group_colors: Indexable[K, ColorLike] | None = None,
        group_names: Indexable[K, str] | None = None,
        legend: bool = False,
    ) -> None:
        BasePlot.__init__(
            self,
            chart,
            fig=fig,
            axes=axes,
        )
        self.fdata1 = fdata1
        self.fdata2 = fdata2

        if self.fdata2 is not None:
            self.fd_final = self.fdata1.concatenate(
                self.fdata2, as_coordinates=True,
            )
        else:
            self.fd_final = self.fdata1

        self.group = group
        self.group_names = group_names
        self.group_colors = group_colors
        self.legend = legend

    @property
    def n_samples(self) -> int:
        return self.fd_final.n_samples

    def _plot(
        self,
        fig: Figure,
        axes: Axes,
    ) -> None:

        self.artists = np.zeros((self.n_samples, 1), dtype=Artist)

        sample_colors, patches = _get_color_info(
            self.fd_final,
            self.group,
            self.group_names,
            self.group_colors,
            self.legend,
        )

        color_dict: Dict[str, ColorLike | None] = {}

        if (
            self.fd_final.dim_domain == 1
            and self.fd_final.dim_codomain == 2
        ):
            ax = axes[0]

            for i in range(self.fd_final.n_samples):

                if sample_colors is not None:
                    color_dict["color"] = sample_colors[i]

                self.artists[i, 0] = ax.plot(
                    self.fd_final.data_matrix[i][:, 0].tolist(),
                    self.fd_final.data_matrix[i][:, 1].tolist(),
                    **color_dict,
                )[0]

        else:
            raise ValueError(
                "Error in data arguments,",
                "codomain or domain is not correct.",
            )

        if self.fd_final.dataset_name is not None:
            fig.suptitle(self.fd_final.dataset_name)

        if self.fd_final.coordinate_names[0] is None:
            ax.set_xlabel("Function 1")
        else:
            ax.set_xlabel(self.fd_final.coordinate_names[0])

        if self.fd_final.coordinate_names[1] is None:
            ax.set_ylabel("Function 2")
        else:
            ax.set_ylabel(self.fd_final.coordinate_names[1])
