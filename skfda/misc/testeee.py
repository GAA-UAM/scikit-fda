import numpy as np
from skfda.misc.scoring import root_mean_squared_error

# Exemple de données
y_true = np.array( [3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

# Appel de la fonction
rmse = root_mean_squared_error(y_true, y_pred)

print(f"Root Mean Squared Error (NumPy): {rmse}")

from skfda.representation.grid import FDataGrid
from skfda.misc.scoring import root_mean_squared_error

# Exemple de données FDataGrid
y_true = np.array([[0.5, 1], [-1, 1], [7, -6]]) #FDataGrid(data_matrix=[[0.5, 1], [-1, 1], [7, -6]], grid_points=[0, 1])
y_pred = np.array([[0, 2],[-1, 2],[8, -5]])

# Appel de la fonction
rmse = root_mean_squared_error(y_true, y_pred)

print(f"Root Mean Squared Error (FData): {rmse}")