from typing import List, Optional, TypeVar

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._utils import (
    _get_figure_and_axes,
    _set_figure_layout,
    _set_figure_layout_for_fdata,
)

T = TypeVar('T')
S = TypeVar('S', Figure, Axes, List[Axes])


class PhasePlanePlot:

    def __init__(
        self,
        fdata1: T,
        fdata2: Optional[T] = None,
    ) -> None:
        self.fdata1 = fdata1
        self.fdata2 = fdata2

    def plot(
        self,
        chart: Optional[S] = None,
        *,
        fig: Optional[Figure] = None,
        axes: Optional[List[Axes]] = None,
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
        **kwargs,
    ) -> Figure:
        fig, axes = _get_figure_and_axes(chart, fig, axes)

        if (
            self.fdata1.dim_domain == 1
            and self.fdata1.dim_codomain == 2
            and self.fdata2 is None
        ):
            fig, axes = _set_figure_layout(
                fig,
                axes,
                dim=self.fdata1.dim_domain + 1,
                n_axes=1,
            )
            axes[0].plot(
                self.fdata1.data_matrix[0][0].tolist(),
                self.fdata1.data_matrix[0][1].tolist(),
                **kwargs,
            )

        elif (
            self.fdata1.dim_domain == self.fdata2.dim_domain
            and self.fdata1.dim_codomain == self.fdata2.dim_codomain
            and self.fdata1.dim_domain == 1
            and self.fdata1.dim_codomain == 1
        ):
            fig, axes = _set_figure_layout_for_fdata(
                self.fdata1, fig, axes,
            )
            axes[0].plot(
                self.fdata1.data_matrix[0].tolist(),
                self.fdata2.data_matrix[0].tolist(),
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
