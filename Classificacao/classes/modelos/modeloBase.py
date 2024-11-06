import numpy as np

class BaseModelClass():
    """
    Modelo base para as classes que implementam os
    modelos de regressão linear. Ele recebe no seu construtor
    a matriz X de variáveis independentes com o intercepto e 
    a matriz y de variáveis dependentes de x, respectivo a posição.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y
        self.matrix_w:np.ndarray
    # função para ser implementada nos modelos
    def trainModel(self):
        pass

    def predict(self, x_new: np.ndarray):
        pass