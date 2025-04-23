import numpy as np
import pandas as pd

from skfda import datasets
from skfda.exploratory.visualization.representation import plot_mixed_data
from skfda.representation.grid import FDataGrid

X, y = datasets.fetch_weather(return_X_y=True, as_frame=True)
fd = X.iloc[:, 0].values
fd_temperatures = fd.coordinates[0]
fd_precipitations = fd.coordinates[1]
fd_temperatures.plot()
fd_precipitations.plot()



fd_D_temperatures = fd_temperatures.derivative(order=1)
fd_D_precipitations = fd_precipitations.derivative(order=1)

fd_D_temperatures.plot()
fd_D_precipitations.plot()

# Create a FDataGrid vector valued with a function and a derivative

data_matrix = np.concatenate([
    fd_temperatures.data_matrix,
    fd_D_temperatures.data_matrix,
], axis=2) 

fd_vector = FDataGrid(
    data_matrix=data_matrix,
    grid_points=fd_temperatures.grid_points,
    coordinate_names=["temperatures", "temperature_derivatives"],
)

# Plot the vector-valued function
fd_vector.plot().show()
input("Wait for input...")


# Create the dataframe
# 

mixed_fd = pd.DataFrame(
    {
        "category": y,
        "temperatures": fd_temperatures,
        "temperature_derivatives": fd_D_temperatures,
        "precipitations_vector": fd_vector,
    }
)

fd = mixed_fd["precipitations_vector"]

mixed_fd.head()


fig = plot_mixed_data(mixed_fd)

print(fig)

input("Wait for input...")
