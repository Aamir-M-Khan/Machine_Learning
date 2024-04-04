import numpy as np


class NaiveBayes:
    def __init__(self, X, y):
        """NaiveBayes constructor

        X: (number of training examples, number of features)
        type: (int,int)

        y: (number of classes)
        type: int
        """

        self.num_examples, self.num_features = X.shape
        self.num_classes = len(np.unique(y))

    def fit(self, X):
        self.classes_mean = {}
        self.classes_variance = {}
        self.classes_prior = {}

        for c in range(self.num_classes):
            X_c = X[y == c]

            self.classes_mean[str(c)] = np.mean(X_c, axis=0)
            self.classes_variance[str(c)] = np.var(X_c, axis=0)
            self.classes_prior[str(c)] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        prob = np.zeros((self.num_examples, self.num_classses))

        for c in range(self.num_classes):
            prior = self.classes_prior[str(c)]
            prob_c = self.density_function(
                X, self.classes_mean[str(c)], self.classes_variance[str(c)]
            )
            prob[:, c] = prob_c + np.log(prior)

            return np.argmax(prob, 1)

    def density_function(self, x, mean, sigma):
        # Calculate probability from Gaussian density density_function
        const = -self.num_features / 2 * np.log(2 * np.pi) - 0.5 * np.sum(
            np.log(sigma + self.eps)
        )

        probs = 0.5 * np.sum(np.power(x - mean, 2) / (sigma + self.eps), 1)
        return const - probs


if __name__ == "__main__":
    X = np.loadtxt("example_data/data.txt", delimiter=",")
    y = np.loadtxt("example_data/targets.txt") - 1

    NB = NaiveBayes(X, y)
    NB.fit(X)
    y_pred = NB.predict(X)

    print(f"Accuracy: {sum(y_pred==y)/X.shape[0]}")
