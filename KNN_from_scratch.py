import numpy as np
from collection import Counter


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k=3):
        """K-Nearest-Neighbour constructor

        :param k: number of nearestneighbour taken into account for each case
        : type: int
        """
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Sort by distance and return indices of the first k neighbours
        k_idx = np.argsort(distances)[: self.k]

        # Extract the labels of the k nearest neighbour training n_samples
        k_neighbour_labels = [self.y_train[i] for i in k_idx]

        # return the most common class label
        most_common = Counter(k_neighbour_labels).most_common(1)
        return most_common[0][0]
