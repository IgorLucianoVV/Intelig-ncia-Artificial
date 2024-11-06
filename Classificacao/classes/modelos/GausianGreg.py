import numpy as np
from classes.modelos.modeloBase import BaseModelClass

class GausianGredModel(BaseModelClass):
    def __init__(self, x: np.ndarray, y: np.ndarray, c: np.ndarray) -> None:
        super().__init__(x, y)
        self.c = c
        self.separatedDataBase()
        self.getStatistcs()
        self.cov = self.getCovAgregado()
        self.inv_cov = np.linalg.inv(self.cov)  # Inversa da covariância agregada

    def getDataByClass(self, classe: int):
        indices_classe = np.where(self.y[0, :] == classe)[0]
        x_classe = self.x[:, indices_classe]
        y_classe = self.y[:, indices_classe]
        return x_classe, y_classe

    def separatedDataBase(self):
        self.separeted_matrix = {}
        for classe in self.c:
            x_c, y_c = self.getDataByClass(classe)
            self.separeted_matrix[classe] = {"x": x_c, "y": y_c}

    def getStatistcs(self):
        self.statics = {}
        for classe in self.c:
            x_data = self.separeted_matrix[classe]["x"]
            mean = np.mean(x_data, axis=1, keepdims=True)
            cov = np.cov(x_data)
            prior = x_data.shape[1] / self.x.shape[1]
            self.statics[classe] = {"mean": mean, "cov": cov, "prior": prior}

    def getCovAgregado(self):
        var_agregada = sum(self.statics[classe]["cov"] * self.separeted_matrix[classe]["x"].shape[1] for classe in self.c)
        return var_agregada / self.x.shape[1]

    def discriminante(self, x_new):
        scores = []
        for classe in self.c:
            mean_diff = x_new - self.statics[classe]["mean"]
            # Vetorização: cálculo do termo Mahalanobis para todas as amostras de uma só vez
            score = np.sum(mean_diff.T @ self.inv_cov * mean_diff.T, axis=1)
            scores.append(score)
        return np.array(scores)  # Array de scores para cada classe

    def predict(self, x_new: np.ndarray):
        # Vetorização: calcula discriminantes para todas as classes e amostras de uma só vez
        scores = self.discriminante(x_new)
        # Retorna a classe com menor distância para cada amostra
        predictions = np.argmin(scores, axis=0)
        return [self.c[i] for i in predictions]
