import numpy as np
from classes.modelos.modeloBase import BaseModelClass

class GausianTraditionalModel(BaseModelClass):
    def __init__(self, x: np.ndarray, y: np.ndarray, c: np.ndarray) -> None:
        super().__init__(x, y)
        self.c = c
        self.separatedDataBase()

    def getDataByClass(self, classe: int):
        indices_classe = np.where(self.y[0, :] == classe)[0]
        x_classe = self.x[:, indices_classe]
        y_classe = self.y[:, indices_classe]
        return x_classe, y_classe

    def separatedDataBase(self):
        self.separeted_matrix = {}
        for classe in self.c:
            x_c, y_c = self.getDataByClass(classe)
            self.separeted_matrix[classe] = {
                "x": x_c,
                "y": y_c
            }

    def getStatistcs(self):
        self.statics = {}
        for classe in self.c:
            x_data = self.separeted_matrix[classe]["x"]
            mean = np.mean(x_data, axis=1, keepdims=True)
            cov = np.cov(x_data)
            inv_cov = np.linalg.pinv(cov)
            det_cov = np.linalg.det(cov)
            prior = x_data.shape[1] / self.x.shape[1]
            self.statics[classe] = {
                "mean": mean,
                "cov": cov,
                "invCov": inv_cov,
                "detCov": det_cov,
                "prior": prior
            }

    def discriminante(self, x_new, classe):
        stat = self.statics[classe]
        diff = x_new - stat["mean"]
        termo_1 = -0.5 * np.log(stat["detCov"])
        termo_2 = -0.5 * (diff.T @ stat["invCov"] @ diff)
        return termo_1 + termo_2

    def predict(self, x_new: np.ndarray):
        predictions = []

        for individuo in x_new.T:
            scores = {classe: self.discriminante(individuo.reshape(-1, 1), classe) for classe in self.c}
            predicted_class = max(scores, key=scores.get)
            predictions.append(predicted_class)
        
        return predictions
