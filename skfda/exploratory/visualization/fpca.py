from __future__ import annotations

import warnings
from typing import Sequence

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from skfda.exploratory.visualization.representation import GraphPlot
from skfda.representation import FData

from ._baseplot import BasePlot


class FPCAPlot(BasePlot):
    """
    FPCAPlot visualization.

    Args:
        mean: The functional data object containing the mean function.
            If len(mean) > 1, the mean is computed.
        components: The principal components
        factor: Multiple of the principal component curve to be added or
            subtracted.
        fig: Figure over which the graph is plotted. If not specified it will
            be initialized
        axes: Axes over where the graph is  plotted.
            If ``None``, see param fig.
        n_rows: Designates the number of rows of the figure.
        n_cols: Designates the number of columns of the figure.
    """

    def __init__(
        self,
        mean: FData,
        components: FData,
        *,
        factor: float = 1,
        multiple: float | None = None,
        chart: Figure | Axes | None = None,
        fig: Figure | None = None,
        axes: Axes | None = None,
        n_rows: int | None = None,
        n_cols: int | None = None,
    ):
        super().__init__(
            chart,
            fig=fig,
            axes=axes,
            n_rows=n_rows,
            n_cols=n_cols,
        )
        self.mean = mean
        self.components = components
        if multiple is None:
            self.factor = factor
        else:
            warnings.warn(
                "The 'multiple' parameter is deprecated, "
                "use 'factor' instead.",
                DeprecationWarning,
            )
            self.factor = multiple

    @property
    def multiple(self) -> float:
        warnings.warn(
            "The 'multiple' attribute is deprecated, use 'factor' instead.",
            DeprecationWarning,
        )
        return self.factor

    @property
    def n_subplots(self) -> int:
        return len(self.components)

    def _plot(
        self,
        fig: Figure,
        axes: Sequence[Axes],
    ) -> None:

        if len(self.mean) > 1:
            self.mean = self.mean.mean()

        for i, ax in enumerate(axes):
            perturbations = self._get_component_perturbations(i)
            GraphPlot(fdata=perturbations, axes=ax).plot()
            ax.set_title(f"Principal component {i + 1}")

    def _get_component_perturbations(self, index: int = 0) -> FData:
        """
        Compute the perturbations over the mean of a principal component.

        Args:
            index: Index of the component for which we want to compute the
                perturbations

        Returns:
            The mean function followed by the positive perturbation and
            the negative perturbation.
        """
        if not isinstance(self.mean, FData):
            raise AttributeError("X must be a FData object")
        perturbations = self.mean.copy()
        perturbations = perturbations.concatenate(
            perturbations[0] + self.multiple * self.components[index],
        )
        return perturbations.concatenate(
            perturbations[0] - self.multiple * self.components[index],
        )
