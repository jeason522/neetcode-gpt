import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        z1 = np.matmul(W1, x) + b1
        y1 = np.maximum(0, z1) 
        z2 = np.matmul(W2, y1) + b2
        loss = np.round(np.mean((z2 - y_true)**2), 4)

        dy1_dz1 =  (z1 > 0).astype(float)
        dz2_dy1 = W2
        dl_dz2 = 2/len(y_true) * (z2 - y_true)
        dl_dy1 = np.matmul(dl_dz2, dz2_dy1)
        dl_dz1 = dl_dy1 * dy1_dz1

        dW2 = np.round(np.outer(dl_dz2, y1), 4)
        db2 = np.round(dl_dz2, 4)
        dW1 = np.round(np.outer(dl_dz1, x), 4)
        db1 = np.round(dl_dz1, 4)
        
        # remove negative zero
        dW1 = np.where(np.abs(dW1) < 1e-12, 0.0, dW1)
        db1 = np.where(np.abs(db1) < 1e-12, 0.0, db1)
        dW2 = np.where(np.abs(dW2) < 1e-12, 0.0, dW2)
        db2 = np.where(np.abs(db2) < 1e-12, 0.0, db2)
        return {"loss": loss.tolist(), "dW1": dW1.tolist(), "db1": db1.tolist(), "dW2": dW2.tolist(), "db2": db2.tolist()}
