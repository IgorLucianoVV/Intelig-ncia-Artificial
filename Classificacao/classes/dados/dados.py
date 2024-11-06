import numpy as np

class DataHandler:
    def __init__(self, path) -> None:
        # Carregar dados do arquivo e transpor para deixar as amostras em linhas
        self.dados = np.loadtxt(path, delimiter=',').T

    # Preparar os dados em matrizes X e Y
    def setMatrizes(self):
        # Formato dos dados:
        # Primeira coluna: Corrugador do Supercílio.
        # Segunda coluna: Zigomático Maior.
        # Terceira coluna: Categoria (1-Neutro, 2-Sorriso, 3-Sobrancelhas levantadas, 4-Surpreso, 5-Rabugento)

        # Concatenar uma coluna de 1's para incluir o termo de viés
        self.matriz = np.concatenate((np.ones((self.dados.shape[0], 1)), self.dados), axis=1)

        # matrix de variáveis independentes
        self.x = self.matriz[:, :self.dados.shape[1]]
        # matriz de categorias (variável dependente)
        # formar a matriz para ficar 
        self.y = self.matriz[:, self.dados.shape[1]].reshape(-1,1)
        
    def returnYasMatrix(self):
        # Matriz Y preenchida com -1 e com 5 colunas (para as 5 classes)
        matY = np.ones((self.dados.shape[0], 5)) * -1
        # Ajuste para que os valores de y sejam índices de 0 a 4
        matY[np.arange(self.dados.shape[0]), self.y.flatten().astype(int) - 1] = 1
        self.matY = matY

    @staticmethod
    def yAsMatrix(y_colum: np.ndarray):
        # Matriz Y preenchida com -1 e com 5 colunas (para as 5 classes)
        matY = np.ones((y_colum.shape[0], 5)) * -1
        matY[np.arange(y_colum.shape[0]), y_colum.flatten().astype(int) -1] = 1
        return matY
    # para os modelos gausianos bayesianos
    # X ∈ R p×N; Y ∈ R C×N;

    def gausianXY(self):
        # Formato dos dados:
        # Primeira coluna: Corrugador do Supercílio.
        # Segunda coluna: Zigomático Maior.
        # Terceira coluna: Categoria (1-Neutro, 2-Sorriso, 3-Sobrancelhas levantadas, 4-Surpreso, 5-Rabugento)

        
        # matrix de variáveis independentes
        self.x = self.dados[:, :self.dados.shape[1]-1]
        # matriz de categorias (variável dependente)
        # formar a matriz para ficar 
        self.y = self.dados[:, self.dados.shape[1]-1].reshape(-1,1)

        
        self.x, self.y = self.x.T, self.y.T

