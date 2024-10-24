import numpy as np


class SGD():
    """
    Stochastic Gradient Descent (SGD) for linear regression.
    """
    def __init__(self, X, y, gamma=1e-3, n_iter=100_000):
        self._X, self._y = X, y
        self._gamma = gamma
        self._n_iter = n_iter
        self._w = np.random.randn(X.shape[1]) # intialize weights randomly

    
    @property
    def weights(self):
        return self._w


    def _compute_gradient(self, X: np.array, y: np.array) -> np.array:
        """
        Returns the gradient of the mean squared error.

        Parameters:
        -----------
        X : np.array
            Features matrix.
        y : np.array
            Target vector.

        Returns:
        --------
        gradient : np.array
            Gradient of the mean squared error.
        """
        return X.T @ (X @ self._w - y) / len(y)


    def fit(self, batch_size: int=10) -> weights:
        """
        Implement the stochastic gradient descent algorithm to minimize the mean squared error.

        Parameters:
        -----------
        batch_size : int
            Size of the batch used to compute the gradient.

        Returns:
        --------
        None
        """

        for _ in range(self._n_iter):
            # Select random samples
            idx = np.random.randint(low=0, high=self._X.shape[0], size=batch_size)
            X_i = self._X[idx, :]
            y_i = self._y[idx]
            
            # Compute the gradient
            gradient = self._compute_gradient(X_i, y_i)

            # Early stopping if the norm of the gradient is close to zero
            if np.linalg.norm(gradient) < 1e-9:
                break

            # Update weights
            self._w -= self._gamma * gradient
    
        return self._w


    def predict(self, X_eval: np.array) -> np.array:
        """
        Predict the target variable.

        Parameters:
        -----------
        X_eval : np.array
            Features matrix.
        
        Returns:
        --------
        y_pred : np.array
            Predicted target variable.
        """
        return np.sign(X_eval @ self._w)
    

    def score(self, X_eval: np.array, y_eval: np.array) -> float:
        """
        Returns the accuracy of the model.

        Parameters:
        -----------
        X_eval : np.array
            Features matrix.
        y_eval : np.array
            Target vector.

        Returns:
        --------
        accuracy : float
            Accuracy of the model.
        """
        y_pred = self.predict(X_eval)
        return np.mean(y_pred == y_eval)