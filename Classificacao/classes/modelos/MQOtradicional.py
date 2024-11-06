import numpy as np
from classes.modelos.modeloBase import BaseModelClass


class MQOT(BaseModelClass):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        super().__init__(x, y)

    def trainModel(self):
        self.matrix_w = np.linalg.inv(self.x.T @ self.x) @ self.x.T @ self.y

    def predict(self, x_new: np.ndarray):
        predicoes =  x_new @ self.matrix_w
        
        return np.argmax(predicoes, axis=1) + 1