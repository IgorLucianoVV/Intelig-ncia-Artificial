import numpy as np
from classes.modelos.modeloBase import BaseModelClass

class GaussianCovModel(BaseModelClass):
    def __init__(self, x: np.ndarray, y: np.ndarray, c: np.ndarray) -> None:
        super().__init__(x, y)
        self.c = c
        self.separatedDataBase()
        self.getStatistcs()

    def getDataByClass(self, classe: int):
        indices_classe = np.where(self.y[0, :] == classe)[0]
        x_classe = self.x[:, indices_classe]
        return x_classe

    def separatedDataBase(self):
        self.separeted_matrix = {}
        for classe in self.c:
            x_c = self.getDataByClass(classe)
            self.separeted_matrix[classe] = {"x": x_c}

    def getStatistcs(self):
        self.cov = np.cov(self.x)
        self.inv_cov = np.linalg.inv(self.cov)  # Inversa da matriz de covariância única
        self.mean_by_class = {classe: np.mean(self.separeted_matrix[classe]["x"], axis=1, keepdims=True) for classe in self.c}

    def discriminante(self, x_new):
        scores = []
        for classe in self.c:
            mean_diff = x_new - self.mean_by_class[classe]
            score = np.sum(mean_diff.T @ self.inv_cov * mean_diff.T, axis=1)
            scores.append(score)
        return np.array(scores)  # Array de scores para cada classe

    def predict(self, x_new: np.ndarray):
        scores = self.discriminante(x_new)
        predictions = np.argmin(scores, axis=0)
        return [self.c[i] for i in predictions]
