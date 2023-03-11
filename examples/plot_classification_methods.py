"""
Classification methods
==================================

It shows a comparison between the accuracies of different classification
methods.
There has been selected one method of each kind present in the library.
In particular, there is one based on depths, Maximum Depth Classifier,
another one based on centroids, Nearest Centroid Classifier, another one
based on the K-Nearest Neighbors, K-Nearest Neighbors Classifier, and
finally, one based on the quadratic discriminant analysis, Parameterized
Functional QDA.

The Berkeley Growth Study dataset is used as input data.
"""

# Author:Álvaro Castillo García
# License: MIT

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from skfda.datasets import fetch_growth
from skfda.exploratory.depth import ModifiedBandDepth
from skfda.exploratory.stats.covariance import ParametricGaussianCovariance
from skfda.misc.covariances import Gaussian
from skfda.ml.classification import (
    KNeighborsClassifier,
    MaximumDepthClassifier,
    NearestCentroid,
    QuadraticDiscriminantAnalysis,
)

##############################################################################
# The Berkeley Growth Study data contains the heights of 39 boys and 54 girls
# from age 1 to 18 and the ages at which they were collected. Males are
# assigned the numeric value 0 while females are assigned a 1. In our
# comparison of the different methods, we will try to learn the sex of a person
# by using its growth curve.
X, y = fetch_growth(return_X_y=True, as_frame=True)
X = X.iloc[:, 0].values
categories = y.values.categories
y = y.values.codes


##############################################################################
# As in many ML algorithms, we split the dataset into train and test. In this
# graph, we can see the training dataset. These growth curves will be used to
# train the model. Hence, the predictions will be data-driven.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=0,
)

# Plot samples grouped by sex
X_train.plot(group=y_train, group_names=categories).show()

##############################################################################
# Below are the growth graphs of those individuals that we would like to
# classify. Some of them will be male and some female.
X_test.plot().show()


##############################################################################
# As said above, we are trying to compare four different methods:
# :class:`~skfda.ml.classification.MaximumDepthClassifier`,
# :class:`~skfda.ml.classification.KNeighborsClassifier`,
# :class:`~skfda.ml.classification.NearestCentroid` and
# :class:`~skfda.ml.classification.QuadraticDiscriminantAnalysis`


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
# The third method we are going to use is the Nearest Centroid Classifier.

centroid = NearestCentroid()
centroid.fit(X_train, y_train)
centroid_pred = centroid.predict(X_test)
print(centroid_pred)
print('The score of Nearest Centroid Classifier is {0:2.2%}'.format(
    centroid.score(X_test, y_test),
))


##############################################################################
# The fourth method considered is a functional quadratic discriminant, where
# the covariance is assumed as having a parametric form, specified with a
# kernel, or covariance function.
#
# We have selected a Gaussian kernel with initial hyperparameters: variance=6
# and lengthscale=1. The selection of the initial parameters does not really
# affect the results as the algorithm will automatically optimize them.
# As regularizer a small value 0.05 has been chosen.

qda = QuadraticDiscriminantAnalysis(
    ParametricGaussianCovariance(
        Gaussian(variance=6, length_scale=1),
    ),
    regularizer=0.05,
)
qda.fit(X_train, y_train)
qda_pred = qda.predict(X_test)
print(qda_pred)
print('The score of functional QDA is {0:2.2%}'.format(
    qda.score(X_test, y_test),
))


##############################################################################
# As it can be seen, the classifier with the lowest score is the Maximum
# Depth Classifier. It obtains a 82.14% accuracy for the test set.
# KNN and the functional QDA can be seen as the best classifiers
# for this problem, with an accuracy of 96.43%.
# Instead, the Nearest Centroid Classifier is not as good as the others.
# However, it obtains an accuracy of 85.71% for the test set.
# It can be concluded that all classifiers work well for this problem, as they
# achieve more than an 80% of score, but the most robust ones are KNN and
# functional QDA.

accuracies = pd.DataFrame({
    'Classification methods':
        [
            'Maximum Depth Classifier',
            'K-Nearest-Neighbors',
            'Nearest Centroid Classifier',
            'Functional QDA',
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
                qda.score(X_test, y_test),
            ),
        ],
})

accuracies


##############################################################################
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

X_test.plot(group=qda_pred, group_names=categories, axes=axs[1][1])
axs[1][1].set_title('Functional QDA', loc='left')

plt.show()
