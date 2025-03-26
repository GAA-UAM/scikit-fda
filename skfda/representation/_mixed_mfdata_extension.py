"""Creates a DataFrame custom accessor for mixed data, to incorporate
    metric information.

"""
from __future__ import annotations

from ._functional_data import FData

import numpy as np
import pandas as pd #type: ignore[import-untyped]

import matplotlib.pyplot as plt
from math import ceil, sqrt


@pd.api.extensions.register_dataframe_accessor("mxdata")
class MixedDataAccessor:
    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        self._obj = pandas_obj

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
                total_plots += first_element.dim_codomain
            else:
                total_plots += 1

        return total_plots

    def plot_dataframe(self) -> None:
        """Custom plot method for Pandas DataFrame that handles numerical and functional data."""
        
        num_cols = self.count_total_plots(self._obj)
        grid_size = ceil(sqrt(num_cols))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 5, grid_size * 5))
        axes = axes.flatten()
        i = 0
        for col in self._obj.columns:
            ax = axes[i]
            data = self._obj[col]
            
            if isinstance(data.iloc[0], FData):
        
                first_element = data.iloc[0]
                if first_element.dim_codomain > 1:  
            
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
        
                ax.scatter(range(len(data)), data)
                ax.set_title(col)

            else:
                ax.axis("off")
            
            i += 1
        

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

        
