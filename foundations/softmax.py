import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z_stable = z - np.max(z)
        res = np.zeros_like(z_stable, dtype=np.float64)
        for i in range(len(z_stable)):
            res[i] = np.exp(z_stable[i])/np.sum(np.exp(z_stable)) 
        return np.round(res, 4)
