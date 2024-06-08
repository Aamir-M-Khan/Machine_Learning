import numpy as np

class SVM:
    """
    Support Vector Machine (SVM) classifier using gradient descent optimization.

    Parameters:
    -----------
    learning_rate : float, optional (default=0.001)
        The step size for the gradient descent optimization.
        
    lambda_param : float, optional (default=0.01)
        The regularization parameter to prevent overfitting.
        
    n_iters : int, optional (default=1000)
        The number of iterations for the training process.
    """
    
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the SVM classifier.

        Parameters:
        -----------
        X : numpy.ndarray
            Training data of shape (n_samples, n_features).
            
        y : numpy.ndarray
            Target values of shape (n_samples,), where each value is either 1 or -1.
        """
        n_samples, n_features = X.shape
        y_transformed = np.where(y <= 0, -1, 1)
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_transformed[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x_i, y_transformed[idx]))
                    self.bias -= self.learning_rate * y_transformed[idx]

    def predict(self, X):
        """
        Predict the class labels for the input data.

        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (n_samples, n_features).
        
        Returns:
        --------
        numpy.ndarray
            Predicted class labels of shape (n_samples,).
        """
        linear_output = np.dot(X, self.weights) - self.bias
        return np.sign(linear_output)
