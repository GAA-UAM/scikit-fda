"""
PACE algorithm for cluster analysis
=======================================================================

Explores the possibility to apply clustering techniques to sparse,
irregularly sampled data using PACE.
"""

# Author: Alejandro Arias Gomez
# License: MIT

# sphinx_gallery_thumbnail_number = 8

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.utils import Bunch

from skfda.datasets._real_datasets import fetch_country_height
from skfda.exploratory.visualization.clustering import ClusterPlot
from skfda.ml.clustering import FuzzyCMeans
from skfda.preprocessing.dim_reduction import PACE
from skfda.representation import FDataIrregular

# %%
# In this example, we will use the Country Height dataset, which contains the
# average height of 144 countries grouped by decades, from 1700 to 2000.
# The dataset is irregularly sampled, meaning that not all countries have
# measurements for all decades.
#
# The dataset is available in the `skfda.datasets` module.
#
# Our goal is to analyse the relationship between the countries and their
# continent using clustering techniques. However, because the dataset is
# irregularly sampled, we will use the PACE algorithm to perform FPCA analysis
# and reconstruct the underlying curves for each country and convert the data
# into a regular grid before applying clustering, in order to be able to use
# the package's clustering algorithms.
#
# We will first load the dataset and plot each country's height curve, divided
# by continent.
country_height_bunch: Bunch = fetch_country_height()
country_height: FDataIrregular = country_height_bunch.data
assert isinstance(
    country_height, FDataIrregular
), "Expected an FDataIrregular object"

target = country_height_bunch.target

country_categories = country_height_bunch.target_names.categories
n_groups = len(country_categories)
cmap = plt.get_cmap("Set2")
country_colors = [cmap(i / (n_groups - 1)) for i in range(n_groups)]

country_height.plot(
    group=target,
    group_colors=country_colors,
    group_names=country_categories,
)
plt.show()


# %%
# To reinforce the irregularity of the data, we will plot the data points
# (years and heights) for all countries, where it can be seen that ever since
# the 1850s, the measurements are much more frequent.
plt.figure()
for t, v in zip(country_height.points, country_height.values, strict=True):
    t_i = np.asarray(t)
    v_i = np.asarray(v)
    plt.scatter(t_i, v_i, alpha=0.7, s=10, color="black")

plt.xlabel("year")
plt.ylabel("height_cm")
plt.title("Average male height by country")
plt.tight_layout()
plt.show()


# %%
# We can now apply the PACE method to the dataset. Due to the fact that the
# domain points are so sparse, bandwidths are going to be very high, which
# highlights the importance of having a dense dataset, but also exemplifies
# the idea that PACE can work even on sparse situations.
pace = PACE(
    n_components=0.99,
    bandwidth_mean=np.array([0.1, 100.0]),
    bandwidth_cov=np.array([0.1, 100.0]),
)
pace.fit(country_height)

fpc_scores = pace.transform(country_height)

# %%
# Plotting the covariance surface of the data, we can see that, in the stages
# where the measurements are more sparse, the covariance is quite low,
# contrasting with the more recent years, where the covariance is much higher.
covariance_x, covariance_y = np.meshgrid(
    pace.t_covariance_,
    pace.t_covariance_,
    indexing="ij",
)
covariance = pace.covariance_.squeeze()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(
    covariance_x,
    covariance_y,
    covariance,
    cmap="viridis",
    alpha=0.7,
)
ax.set_title("Smoothed Covariance Surface via PACE")
plt.tight_layout()
plt.show()


# %%
# We can further inspect the principal components of the data, where we can see
# that the first component takes the form of a steady increase over the years,
# while the second component takes the shape of an increase over the first
# years, followed by an accentuated decrease after the 1900s, which could be
# supported by the first world war, a period of hunger. The third principal
# component also takes the shape of an increase, but starting a bit later,
# which could model countries where abundant food or resources arrived at a
# latter stage, followed by a stabilization and slight decrease since 1900.
# Combined, the last two components explain the more subtle variations in the
# data, while the first component explains the main trend.
fig, ax = plt.subplots()
for i, sample in enumerate(pace.components_):
    label = pace.components_.sample_names[i]
    sample.plot(axes=ax, label=label)

ax.set_xlabel(pace.components_.argument_names[0] or "Domain")
ax.set_ylabel(pace.components_.coordinate_names[0] or "Value")
ax.legend()
plt.show()

print(pace.explained_variance_ratio[:3])

# %%
# From the FPC scores, we can reconstruct the whole dataset, allowing us to
# view the data in a regular grid, which is a necessary tool for clustering
# algorithms.
reconstructed = pace.inverse_transform(fpc_scores)

reconstructed.plot()
plt.show()

# %%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

curve_name = "country"
X = pd.DataFrame({
    curve_name: reconstructed,
}).iloc[:, [0]]
X = X.iloc[:, 0].array

X_train, X_test, y_train, y_test = train_test_split(
    X,
    target,
    test_size=0.3,
    stratify=target,
    random_state=8,
)

X_train.plot(
    group=y_train,
    group_names=country_categories,
    group_colors=country_colors,
)
plt.show()

# %%
X_test.plot()
plt.show()

# %%
from skfda.ml.classification import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print(knn_pred)
print(f"The score of KNN is {knn.score(X_test, y_test):2.2%}")

fig = X_test.plot(
    group=knn_pred,
    group_names=country_categories,
    group_colors=country_colors,
)

X_test_0 = X_test[knn_pred == 0]
X_test_1 = X_test[knn_pred == 1]
X_test_2 = X_test[knn_pred == 2]
X_test_3 = X_test[knn_pred == 3]

X_test_0.mean().plot(fig=fig, color="#157a5b", linewidth=3)
X_test_1.mean().plot(fig=fig, color="#666666", linewidth=3)
X_test_2.mean().plot(fig=fig, color="#7fbc6a", linewidth=3)
X_test_3.mean().plot(fig=fig, color="#c49c4d", linewidth=3)
plt.show()

# %%
# References
# ----------
#
# .. footbibliography::
