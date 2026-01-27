"""
PACE algorithm for classification analysis
=======================================================================

Explores the possibility to apply clustering techniques to sparse,
irregularly sampled data using PACE.
"""

# Author: Alejandro Arias Gomez
# License: MIT

# sphinx_gallery_thumbnail_number = 5

# %%
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

from skfda.datasets import fetch_country_height
from skfda.ml.classification import KNeighborsClassifier
from skfda.preprocessing.dim_reduction import PACE
from skfda.representation import FDataIrregular

# %%
# This example explores the possibility to apply clustering techniques to
# sparse, irregularly sampled data using the PACE algorithm, as described in
# :footcite:ts:`yao+muller+wang_2005_pace`.
#
# Throughout this analysis, we will use the Country Height dataset, which
# contains the average height of 144 countries grouped by decades, from 1810 to
# 1989. The dataset is irregularly sampled, meaning that not all countries have
# measurements for all decades.
#
# The dataset is available in the `skfda.datasets` module.
#
# Our goal is to analyse the relationship between the countries and their
# continent using classification techniques. However, because the dataset is
# irregularly sampled, we will use the PACE algorithm to perform FPCA analysis
# and reconstruct the underlying curves for each country, converting the data
# into a regular grid before classification, in order to be able to use the
# package's algorithms.
#
# We will first load the dataset and plot each country's height curve, divided
# by continent. In addition, we will represent how many countries we have per
# continent, in order to compare the classification results with a
# classificator by majority.
country_height_bunch: Bunch = fetch_country_height()
country_height: FDataIrregular = country_height_bunch.data
assert isinstance(
    country_height,
    FDataIrregular,
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

counter = Counter(target)
total = len(target)

for i in range(5):
    quantity = counter.get(i, 0)
    perc = (quantity / total) * 100
    print(f"Continent {i}: {quantity} countries ({perc:.2f}%)")


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
# while the second and third components account for the smaller variations in
# the different subjects. These could represent periods of hunger, wars,
# industrialisation or other affecting factors. Because no direct event can be
# linked to these variations, we are unable to determine the sign of these
# principal components.
#
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

print(pace.explained_variance_ratio_[:3])

# %%
# From the FPC scores, we can reconstruct the whole dataset, allowing us to
# view the data in a regular grid, which is a necessary tool for classification
# algorithms in the package.
n_components_list = [1, 2, 3]
reconstructed_all = []
titles = [
    "Objective Classification",
    "PACE (1 component)",
    "PACE (2 components)",
    "PACE (3 components)",
]
mean_colors = ["#00522c", "#6c84b5", "#238b45", "#8c6d31", "#666666"]
knn_scores = []
knn_preds = []

for n in n_components_list:
    pace = PACE(
        n_components=n,
        bandwidth_mean=22.74,
        bandwidth_cov=28.53,
    )
    pace_scores = pace.fit_transform(country_height)
    reconstructed = pace.inverse_transform(pace_scores)

    curve_name = "country"
    X = pd.DataFrame(
        {
            curve_name: reconstructed,
        }
    ).iloc[:, [0]]
    X = X.iloc[:, 0].array

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        target,
        test_size=0.3,
        stratify=target,
        random_state=8,
    )

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)

    knn_scores.append(knn.score(X_test, y_test))
    knn_preds.append(knn_pred)

    reconstructed_all.append(X_test)
    if n == 3:
        reconstructed_all.append(reconstructed)

# %%
# We finish the analysis by showcasing a visual comparison of classification
# results based on the number of principal components used. The first panel
# represents the true classes (continent) with average curves per group, while
# the remaining panels show the predicted groups using 1, 2, and 3 FPCs
# respectively.
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, ax in enumerate(axes.flat):
    if i == 0:
        reconstructed_all[3].plot(
            axes=ax,
            group=target,
            group_names=country_categories,
            group_colors=country_colors,
        )

        for label, color in zip(range(5), mean_colors, strict=True):
            reconstructed_all[3][target == label].mean().plot(
                axes=ax,
                color=color,
                linewidth=5,
            )
    else:
        reconstructed_all[i - 1].plot(
            axes=ax,
            group=knn_preds[i - 1],
            group_names=country_categories,
            group_colors=country_colors,
        )

        for label, color in zip(range(5), mean_colors, strict=True):
            reconstructed_all[i - 1][knn_preds[i - 1] == label].mean().plot(
                axes=ax,
                color=color,
                linewidth=5,
            )

    ax.set_title(titles[i])

plt.tight_layout()
plt.show()

print(knn_scores)
# %%
# Analysing the classification scores for each experiment, we observe that,
# even in the worst case scenario, the classification is more accurate than
# a random classificator (20% success) or a classificator by majority (32.17%
# success). In addition, these results show a clear upward trend in
# classification accuracy as the number of components increases. Each
# additional component captures further variability in the data, which helps to
# better differentiate between countries. This highlights a trade-off between
# dimensionality and expressiveness: even a small number of components (3)
# achieves over 60% accuracy despite the original data's irregularity,
# showcasing PACE effectiveness in recovering discriminative features from
# sparse data.
#
# Focusing solely on the analysis of this particular problem, the results
# confirm the presence of continent-level patterns in the data, but the
# moderate accuracy also suggests that intra-continental diversity—such as
# socioeconomic or ethnic differences—limits the strength of the correlation,
# leaving room for further refinement or more granular modeling.

# %%
# References
# ----------
#
# .. footbibliography::
