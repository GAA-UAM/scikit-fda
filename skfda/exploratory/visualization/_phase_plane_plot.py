from typing import List, Optional, TypeVar

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from ...representation import FData

from ._utils import (
    _get_figure_and_axes,
    _set_figure_layout_for_fdata,
)

S = TypeVar('S', Figure, Axes, List[Axes])


class PhasePlanePlot:

    def __init__(
        self,
        fdata1: FData,
        fdata2: Optional[FData] = None,
    ) -> None:
        self.fdata1 = fdata1
        self.fdata2 = fdata2

    def plot(
        self,
        chart: Optional[S] = None,
        *,
        fig: Optional[Figure] = None,
        axes: Optional[List[Axes]] = None,
        **kwargs,
    ) -> Figure:
        fig, axes = _get_figure_and_axes(chart, fig, axes)

        if (
            self.fdata2 is not None
        ):
            if (
                self.fdata1.dim_domain == self.fdata2.dim_domain
                and self.fdata1.dim_codomain == self.fdata2.dim_codomain
                and self.fdata1.dim_domain == 1
                and self.fdata1.dim_codomain == 1
            ):
                fd = self.fdata1.concatenate()
            else:
                raise ValueError(
                    "Error in data arguments",
                )
        else:
            fd = self.fdata1

        if (
            fd.dim_domain == 1
            and fd.dim_codomain == 2
        ):
            fig, axes = _set_figure_layout_for_fdata(
                fd, fig, axes,
            )
            axes[0].plot(
                fd.data_matrix[0][0].tolist(),
                fd.data_matrix[0][1].tolist(),
                **kwargs,
            )
        else:
            raise ValueError(
                "Error in data arguments",
            )

        fig.suptitle("Phase-Plane Plot")
        axes[0].set_xlabel("Function 1")
        axes[0].set_ylabel("Function 2")

        return fig
