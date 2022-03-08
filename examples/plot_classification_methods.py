"""
Classification methods.

==================================

Shows a comparison between different classification methods.
The Berkeley Growth Study dataset is used as input data.
Classification methods KNN, Maximum Depth, Nearest Centroid and
Gaussian classifier are compared.
"""

# Author:Álvaro Castillo García
# License: MIT

from GPy.kern import Linear
from sklearn.model_selection import train_test_split

from skfda.datasets import fetch_growth
from skfda.exploratory.depth import ModifiedBandDepth
from skfda.ml.classification import (
    GaussianClassifier,
    KNeighborsClassifier,
    MaximumDepthClassifier,
    NearestCentroid,
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
# :class:`~skfda.ml.classification.GaussianClassifier`


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

# Plot the prediction
X_test.plot(group=depth_pred, group_names=categories).show()


##############################################################################
# The second method to consider is the K-Nearest Neighbours Classifier.


knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print(knn_pred)
print('The score of KNN is {0:2.2%}'.format(knn.score(X_test, y_test)))

# Plot the prediction
X_test.plot(group=knn_pred, group_names=categories).show()


##############################################################################
# The third method we are going to use is the Nearest Centroid Classifier

centroid = NearestCentroid()
centroid.fit(X_train, y_train)
centroid_pred = centroid.predict(X_test)
print(centroid_pred)
print('The score of Nearest Centroid Classifier is {0:2.2%}'.format(
    centroid.score(X_test, y_test),
))

# Plot the prediction
X_test.plot(group=centroid_pred, group_names=categories).show()


##############################################################################
# The fourth method considered is a Gaussian Process based Classifier.
# As the data set tends to be linear we have selected a linear kernel with
# initial parameters: variance=6 and mean=1
# As regularizer a small value 0.05 has been chosen.

gaussian = GaussianClassifier(
    kernel=Linear(1, variances=6),
    regularizer=0.05,
)
gaussian.fit(X_train, y_train)
gaussian_pred = gaussian.predict(X_test)
print(gaussian_pred)
print('The score of Gaussian Process Classifier is {0:2.2%}'.format(
    gaussian.score(X_test, y_test),
))

# Plot the prediction
X_test.plot(group=gaussian_pred, group_names=categories).show()
