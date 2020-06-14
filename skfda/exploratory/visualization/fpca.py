from matplotlib import pyplot as plt
from skfda.representation import FDataGrid, FDataBasis, FData
from skfda.exploratory.visualization._utils import _get_figure_and_axes


def plot_fpca_perturbation_graphs(mean, components, multiple,
                                  chart = None,
                                  fig=None,
                                  axes=None,
                                  **kwargs):
    """ Plots the perturbation graphs for the principal components.
    The perturbations are defined as variations over the mean. Adding a multiple
    of the principal component curve to the mean function results in the
    positive perturbation and subtracting a multiple of the principal component
    curve results in the negative perturbation. For each principal component
    curve passed, a subplot with the mean and the perturbations is shown.

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

    Returns:
        (FDataGrid or FDataBasis): this contains the mean function followed
        by the positive perturbation and the negative perturbation.
    """

    if len(mean) > 1:
        mean = mean.mean()

    fig, axes = _get_figure_and_axes(chart, fig, axes)

    if not axes:
        axes = fig.subplots(nrows=len(components))

    for i in range(len(axes)):
        aux = _get_component_perturbations(mean, components, i, multiple)
        aux.plot(axes[i], **kwargs)
        axes[i].set_title('Principal component ' + str(i + 1))

    return fig


def _get_component_perturbations(mean, components, index=0, multiple=30):
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
    if not isinstance(mean, FData):
        raise AttributeError("X must be a FData object")
    perturbations = mean.copy()
    perturbations = perturbations.concatenate(
        perturbations[0] + multiple * components[index])
    perturbations = perturbations.concatenate(
        perturbations[0] - multiple * components[index])
    return perturbations
