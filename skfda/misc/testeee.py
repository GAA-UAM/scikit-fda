import numpy as np
from sklearn.metrics import mean_squared_error
from skfda.misc.scoring import root_mean_squared_error

# Données
y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
y_pred = np.array([[0, 2], [-1, 2], [8, -5]])

# Calcul avec scikit-learn
mse = mean_squared_error(y_true, y_pred, multioutput='uniform_average')
rmse_sklearn = np.sqrt(mse)

# Calcul avec votre fonction
rmse_custom = root_mean_squared_error(y_true, y_pred)

print(f"RMSE (scikit-learn): {rmse_sklearn}")
print(f"RMSE (custom): {rmse_custom}")