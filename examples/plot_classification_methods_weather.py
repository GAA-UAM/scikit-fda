"""
Classification methods
==================================

This tutorial shows a comparison between the accuracies of different
classification methods.
There has been selected one method of each kind present in the library.
In particular, there is one based on depths, Maximum Depth Classifier,
another one based on centroids, Nearest Centroid Classifier, another one
based on the K-Nearest Neighbors, K-Nearest Neighbors Classifier, and
finally, one based on the quadratic discriminant analysis, Parameterized
Functional QDA.

The Canadian Weather dataset is used as input data.
"""

# Author:Álvaro Castillo García
# License: MIT

import matplotlib.pyplot as plt
import pandas as pd
from GPy.kern import RBF
from sklearn.model_selection import train_test_split

from skfda.datasets import fetch_weather
from skfda.exploratory.depth import ModifiedBandDepth
from skfda.ml.classification import (
    KNeighborsClassifier,
    MaximumDepthClassifier,
    NearestCentroid,
    ParameterizedFunctionalQDA,
)

##############################################################################
# The Canadian Weather dataset is formed by the measures of the daily
# temperatures and precipitations over a year at 35 different locations in
# Canada. It also includes targets which identify the climate present on the
# location of each observation. The four kinds of climates that were
# identified are: Arctic, Atlantic, Continental and Pacific
X, y = fetch_weather(return_X_y=True, as_frame=True)
X = X.iloc[:, 0].values
X = X.coordinates[0]
categories = y.values.categories
y = y.values.codes


##############################################################################
# As in many ML algorithms, we split the dataset into train and test. In this
# graph, we can see the training dataset. These temperature curves will be
# used to train the model. Hence, the predictions will be data-driven.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=0,
)

# Plot samples grouped by climate
X_train.plot(group=y_train, group_names=categories).show()


##############################################################################
# Below are the  temperature graphs that we would like to classify.
# They belong to artic, atlantic, continental and pacific climates.
X_test.plot().show()


##############################################################################
# As said above, we are trying to compare four different methods:
# :class:`~skfda.ml.classification.MaximumDepthClassifier`,
# :class:`~skfda.ml.classification.KNeighborsClassifier`,
# :class:`~skfda.ml.classification.NearestCentroid` and
# :class:`~skfda.ml.classification.ParameterizedFunctionalQDA`


##############################################################################
# The first method we are going to use is the Maximum Depth Classifier.
# As depth method we will consider the Modified Band Depth.

depth = MaximumDepthClassifier(depth_method=ModifiedBandDepth())
depth.fit(X_train, y_train)
depth_pred = depth.predict(X_test)
print(depth_pred)
print('The score of Maximum Depth Classifier is {0:2.2%}'.format(
    depth.score(X_test, y_test),
))

##############################################################################
# The second method to consider is the K-Nearest Neighbours Classifier.


knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print(knn_pred)
print('The score of KNN is {0:2.2%}'.format(knn.score(X_test, y_test)))


##############################################################################
# The third method we are going to use is the Nearest Centroid Classifier

centroid = NearestCentroid()
centroid.fit(X_train, y_train)
centroid_pred = centroid.predict(X_test)
print(centroid_pred)
print('The score of Nearest Centroid Classifier is {0:2.2%}'.format(
    centroid.score(X_test, y_test),
))


##############################################################################
# The fourth method considered is a Parameterized functional quadratic
# discriminant.
# We have selected a gaussian kernel with initial parameters: variance=6 and
# mean=1. The selection of the initial parameters does not really affect the
# results as the algorithm will automatically optimize them.
# As regularizer a small value 0.05 has been chosen.

pfqda = ParameterizedFunctionalQDA(
    kernel=RBF(input_dim=1, variance=6, lengthscale=1),
    regularizer=0.05,
)
pfqda.fit(X_train, y_train)
pfqda_pred = pfqda.predict(X_test)
print(pfqda_pred)
print('The score of Parameterized Functional QDA is {0:2.2%}'.format(
    pfqda.score(X_test, y_test),
))


##############################################################################
# In this experiment we are able to observe that KNN classifies perfectly the
# dataset while the other classifiers accumulate less accuracy. Maximum Depth
# classifier is giving the worst results with a 72.73 % of accuracy whereas
# Parameterized Functional QDA and Maximum Depth Classifier obtain the same
# accuracy for the test dataset, 81.82 %.

accuracies = pd.DataFrame({
    'Classification methods':
        [
            'Maximum Depth Classifier',
            'K-Nearest-Neighbors',
            'Nearest Centroid Classifier',
            'Parameterized Functional QDA',
        ],
    'Accuracy':
        [
            '{0:2.2%}'.format(
                depth.score(X_test, y_test),
            ),
            '{0:2.2%}'.format(
                knn.score(X_test, y_test),
            ),
            '{0:2.2%}'.format(
                centroid.score(X_test, y_test),
            ),
            '{0:2.2%}'.format(
                pfqda.score(X_test, y_test),
            ),
        ],
})

accuracies


# The figure below shows the results of the classification for the test set on
# the four methods considered.
# It can be seen that the curves are similarly classified by all of them.


fig, axs = plt.subplots(2, 2)
plt.subplots_adjust(hspace=0.45, bottom=0.06)

X_test.plot(group=centroid_pred, group_names=categories, axes=axs[0][1])
axs[0][1].set_title('Nearest Centroid Classifier', loc='left')

X_test.plot(group=depth_pred, group_names=categories, axes=axs[0][0])
axs[0][0].set_title('Maximum Depth Classifier', loc='left')

X_test.plot(group=knn_pred, group_names=categories, axes=axs[1][0])
axs[1][0].set_title('KNN', loc='left')

X_test.plot(group=pfqda_pred, group_names=categories, axes=axs[1][1])
axs[1][1].set_title('Parameterized Functional QDA', loc='left')

plt.show()
