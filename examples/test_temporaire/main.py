import numpy as np
import matplotlib.pyplot as plt
from skfda import FDataGrid
from skfda.misc.scoring import root_mean_squared_error, root_mean_squared_log_error

# Générer des données fonctionnelles de test
np.random.seed(0)
n_samples, n_features = 10, 100
time_points = np.linspace(0, 1, n_features)
data_true = np.sin(2 * np.pi * time_points)  # Fonction cible
data_pred = data_true + np.random.normal(size=(n_samples, n_features)) * 0.1  # Prédictions perturbées

# Ajouter une petite constante pour éviter les valeurs négatives dans les log (en particulier pour RMSLE)
data_true[data_true <= 0] += 1e-10  # Ajouter une petite constante aux valeurs négatives ou nulles
data_pred[data_pred <= 0] += 1e-10

# Créer des objets FDataGrid pour les données vraies et prédites
fd_true = FDataGrid(data_true, time_points)
fd_pred = FDataGrid(data_pred, time_points)

# Visualiser les données fonctionnelles vraies et prédites
fd_true.plot(label="Vraies données")
fd_pred.plot(label="Données prédites", linestyle='--')
plt.title("Données fonctionnelles vraies vs. prédites")
plt.legend()
plt.show()

# Calculer les erreurs
rmse = root_mean_squared_error(fd_true, fd_pred)
rmsle = root_mean_squared_log_error(fd_true, fd_pred)

print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Root Mean Squared Log Error (RMSLE): {rmsle}")

# Visualiser l'erreur absolue
error = fd_pred - fd_true
error.plot()
plt.title("Erreur entre les données prédites et les vraies données")
plt.show()
