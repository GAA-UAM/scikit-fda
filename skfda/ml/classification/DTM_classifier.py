"""Distance to trimmed means (DTM) classification."""

from ..._neighbors.classification import NearestCentroid
from ...exploratory.depth import Depth, ModifiedBandDepth
from ...exploratory.stats import trim_mean
from ...misc.metrics import lp_distance


class DTMClassifier(NearestCentroid):
    """Distance to trimmed means (DTM) classification.

    Test samples are classified to the class that minimizes the distance of
    the observation to the trimmed mean of the group.

    Parameters:
        proportiontocut (float): indicates the percentage of functions to
            remove. It is not easy to determine as it varies from dataset to
            dataset.
        depth_method (Depth, default
            :class:`ModifiedBandDepth <skfda.depth.ModifiedBandDepth>`):
            The depth class used to order the data. See the documentation of
            the depths module for a list of available depths. By default it
            is ModifiedBandDepth.
        metric (function, default
            :func:`lp_distance <skfda.misc.metrics.lp_distance>`):
            Distance function between two functional objects. See the
            documentation of the metrics module for a list of available
            metrics.

    Examples:
        Firstly, we will import and split the Berkeley Growth Study dataset

        >>> from skfda.datasets import fetch_growth
        >>> from sklearn.model_selection import train_test_split
        >>> dataset = fetch_growth()
        >>> fd = dataset['data']
        >>> y = dataset['target']
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     fd, y, test_size=0.25, stratify=y, random_state=0)

        We will fit a Distance to trimmed means classifier

        >>> from skfda.ml.classification import DTMClassifier
        >>> clf = DTMClassifier(proportiontocut=0.25)
        >>> clf.fit(X_train, y_train)
        DTMClassifier(...)

        We can predict the class of new samples

        >>> clf.predict(X_test) # Predict labels for test samples
        array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1,
                1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1])

        Finally, we calculate the mean accuracy for the test data

        >>> clf.score(X_test, y_test)
        0.875

    See also:
        :class:`~skfda.ml.classification.MaximumDepthClassifier
    """

    def __init__(self, proportiontocut,
                 depth_method: Depth = ModifiedBandDepth(),
                 metric=lp_distance):
        """Initialize the classifier."""
        super().__init__(metric=metric,
                         centroid=lambda fdatagrid: trim_mean(fdatagrid,
                                                              proportiontocut,
                                                              depth_method))
