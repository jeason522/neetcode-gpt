import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        y_stable = y_pred + 1e-7
        n = len(y_true)
        l = -1/n * np.sum(y_true*np.log(y_stable) + (1-y_true)*np.log(1-y_stable))
        return np.round(l, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        y_stable = y_pred + 1e-7
        n_samples = y_true.shape[0]
        n_classes = y_true.shape[1]
        l = 0.0
        for i in range(n_samples):
            for c in range(n_classes):
                l += y_true[i, c] * np.log(y_stable[i, c])
        res = -1/n_samples * l
        return np.round(res, 4)
