import numpy as np
import matplotlib.pyplot as plt
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA

# Générer des données fonctionnelles de test
np.random.seed(0)
n_samples, n_features = 10, 100
time_points = np.linspace(0, 1, n_features)
data = np.sin(2 * np.pi * time_points) + np.random.normal(size=(n_samples, n_features))

# Créer un objet FDataGrid
fd = FDataGrid(data, time_points)

# Visualiser les données fonctionnelles
fd.plot()
plt.title("Données fonctionnelles de test")
plt.show()

# Effectuer une analyse en composantes principales fonctionnelles (FPCA)
fpca = FPCA(n_components=2)
fpca.fit(fd)

# Visualiser les composantes principales
fpca.components_.plot()
plt.title("Composantes principales fonctionnelles")
plt.show()
