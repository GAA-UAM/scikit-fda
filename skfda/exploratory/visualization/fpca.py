from skfda.exploratory.visualization.representation import GraphPlot
from typing import Optional, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from skfda.exploratory.visualization._utils import _get_figure_and_axes
from skfda.representation import FData

from ._baseplot import BasePlot


class FPCAPlot(BasePlot):
    """
    FPCAPlot visualization.

    Args:
        mean (FDataGrid or FDataBasis):
            the functional data object containing the mean function.
            If len(mean) > 1, the mean is computed.
        components (FDataGrid or FDataBasis):
            the principal components
        multiple (float):
            multiple of the principal component curve to be added or
            subtracted.
        fig (figure object, optional):
            figure over which the graph is plotted. If not specified it will
            be initialized
        axes (axes object, optional): axis over where the graph is  plotted.
            If None, see param fig.
    """

    def __init__(
        self,
        mean, components, multiple,
        chart=None,
        fig=None,
        axes=None,
    ):
        BasePlot.__init__(self)
        self.mean = mean
        self.components = components
        self.multiple = multiple
        
        self.set_figure_and_axes(chart, fig, axes)

    def plot(self, **kwargs):
        """ 
        Plots the perturbation graphs for the principal components.
        The perturbations are defined as variations over the mean. Adding a multiple
        of the principal component curve to the mean function results in the
        positive perturbation and subtracting a multiple of the principal component
        curve results in the negative perturbation. For each principal component
        curve passed, a subplot with the mean and the perturbations is shown.

        Returns:
            (FDataGrid or FDataBasis): this contains the mean function followed
            by the positive perturbation and the negative perturbation.
        """

        if len(self.mean) > 1:
            self.mean = self.mean.mean()

        for i in range(len(self.axes)):
            aux = self._get_component_perturbations(i)
            GraphPlot(fdata=aux, axes=self.axes[i]).plot(**kwargs)
            self.axes[i].set_title('Principal component ' + str(i + 1))

        return self.fig

    def n_samples(self) -> int:
        return self.fdata.n_samples

    def set_figure_and_axes(
        self,
        chart: Union[Figure, Axes, None] = None,
        fig: Optional[Figure] = None,
        axes: Optional[Axes] = None,
    ) -> None:
        fig, axes = _get_figure_and_axes(chart, fig, axes)
        if not axes:
            axes = fig.subplots(nrows=len(self.components))

        self.fig = fig
        self.axes = axes

    def _get_component_perturbations(self, index=0):
        """ Computes the perturbations over the mean function of a principal
        component at a certain index.

        Args:
            X (FDataGrid or FDataBasis):
                the functional data object from which we obtain the mean
            index (int):
                index of the component for which we want to compute the
                perturbations
            multiple (float):
                multiple of the principal component curve to be added or
                subtracted.

        Returns:
            (FDataGrid or FDataBasis): this contains the mean function followed
            by the positive perturbation and the negative perturbation.
        """
        if not isinstance(self.mean, FData):
            raise AttributeError("X must be a FData object")
        perturbations = self.mean.copy()
        perturbations = perturbations.concatenate(
            perturbations[0] + self.multiple * self.components[index])
        perturbations = perturbations.concatenate(
            perturbations[0] - self.multiple * self.components[index])
        return perturbations
