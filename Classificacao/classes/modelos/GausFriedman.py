import numpy as np
from classes.modelos.modeloBase import BaseModelClass

class GausianFriedman(BaseModelClass):
    def __init__(self, x: np.ndarray, y: np.ndarray, c: np.ndarray, lamb: float) -> None:
        super().__init__(x, y)
        self.c = c
        self.lamb = lamb
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
        self.statics = {}
        for classe in self.c:
            mean = np.mean(self.separeted_matrix[classe]["x"], axis=1, keepdims=True)
            cov = np.cov(self.separeted_matrix[classe]["x"])
            prior = self.separeted_matrix[classe]["x"].shape[1] / self.x.shape[1]
            self.statics[classe] = {"mean": mean, "cov": cov, "prior": prior}
        self.setCov()

    def getCovAgregado(self):
        var = sum(self.statics[classe]["cov"] * self.separeted_matrix[classe]["x"].shape[1] for classe in self.c)
        return var / self.x.shape[1]

    def setCov(self):
        gre_cov = self.getCovAgregado()
        N = self.x.shape[1]
        for classe in self.c:
            n = self.separeted_matrix[classe]["x"].shape[1]
            cov = self.statics[classe]["cov"]
            new_cov = ((1 - self.lamb) * n * cov + self.lamb * N * gre_cov) / ((1 - self.lamb) * n + self.lamb * N)
            self.statics[classe]["cov"] = new_cov
            self.statics[classe]["invCov"] = np.linalg.inv(new_cov)
            self.statics[classe]["detCov"] = np.linalg.det(new_cov)

    def discriminante(self, x_new):
        scores = []
        for classe in self.c:
            mean = self.statics[classe]["mean"]
            inv_cov = self.statics[classe]["invCov"]
            log_det_cov = np.log(self.statics[classe]["detCov"])
            diff = x_new - mean
            termo_1 = -0.5 * log_det_cov
            termo_2 = -0.5 * np.sum((diff.T @ inv_cov) * diff.T, axis=1)
            scores.append(termo_1 + termo_2)
        return np.array(scores)

    def predict(self, x_new: np.ndarray):
        scores = self.discriminante(x_new)
        predictions = np.argmax(scores, axis=0)
        return [self.c[i] for i in predictions]
