"""Creates a DataFrame custom accessor for mixed data, to incorporate
    metric information.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, Callable
import warnings

from ._functional_data import FData

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype

from ..typing._numpy import (
    ArrayLike,
    NDArrayBool,
    NDArrayFloat,
    NDArrayInt,
    NDArrayObject,
)
import matplotlib.pyplot as plt
from math import ceil, sqrt


@pd.api.extensions.register_dataframe_accessor("metric")
class MetricAccessor:
    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        self._obj = pandas_obj
        self._metrics: Dict[
            str,
            Callable[[Union[FData, NDArrayFloat], Union[FData, NDArrayFloat]], float],
        ] = {}

    def set_metric(
        self,
        column: str,
        func: Callable[[Union[FData, NDArrayFloat], Union[FData, NDArrayFloat]], float],
    ) -> None:
        if column not in self._obj.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")
        if not callable(func):
            raise TypeError("Metric function must be callable.")
        self._metrics[column] = func

    def get_metric(
        self, column: str
    ) -> Optional[
        Callable[[Union[FData, NDArrayFloat], Union[FData, NDArrayFloat]], float]
    ]:
        return self._metrics.get(column, None)

    def apply_metrics(self, row1: int, row2: int) -> Dict[str, Any]:
        if row1 not in self._obj.index or row2 not in self._obj.index:
            raise ValueError("Both row indices must exist in the DataFrame.")

        results = {}
        for col in self._metrics:
            if col in self._obj.columns:
                metric_func = self._metrics[col]
                val1, val2 = self._obj.at[row1, col], self._obj.at[row2, col]
                results[col] = metric_func(val1, val2)

        return results
    
    def count_total_plots(self, df: pd.DataFrame) -> int:
        """
        Counts the total number of plots needed for a DataFrame, 
        considering FDataGrid codomain dimensions.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing numerical and functional data.
        
        Returns:
            int: Total number of plots required.
        """
        total_plots = 0

        for col in df.columns:
            first_element = df[col].iloc[0]

            if isinstance(first_element, FData):
                total_plots += first_element.dim_codomain  # Count one plot per codomain component
            else:
                total_plots += 1  # Count as a single plot for numerical data

        return total_plots

    def plot_dataframe(self) -> None:
        """Custom plot method for Pandas DataFrame that handles numerical and functional data."""
        
        num_cols = self.count_total_plots(self._obj)
        grid_size = ceil(sqrt(num_cols))  # Determine grid size (square layout)
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 5, grid_size * 5))
        axes = axes.flatten()  # Flatten for easy iteration
        i = 0
        for col in self._obj.columns:
            ax = axes[i]
            data = self._obj[col]
            
            if isinstance(data.iloc[0], FData):
                # Functional Data Plot
                first_element = data.iloc[0]
                if first_element.dim_codomain > 1:  
                    # Create a list of dim_codomain axes
                    ax = axes[i:i + first_element.dim_codomain]
                    for fd in data:
                        fd.plot(axes=ax)
                    [ax.set_title(f"{col} - {j}") for j, ax in enumerate(ax)]
                    i += first_element.dim_codomain -1
                else:
                    for fd in data:
                        fd.plot(axes=ax)
                    ax.set_title(col)

            elif isinstance(data.iloc[0], np.ndarray) or np.isscalar(data.iloc[0]):
                # Scatter plot for numerical data
                ax.scatter(range(len(data)), data)
                ax.set_title(col)

            else:
                ax.axis("off")  # Hide axes for unsupported types
            
            i += 1
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

        
