import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
# Obtendo o diretório atual do Jupyter Notebook
caminho_script = "\\".join(os.getcwd().split("\\")[:-1])
print("Caminho do diretório atual:", caminho_script)
# Adicionando o diretório ao sys.path se precisar importar módulos
sys.path.append(caminho_script)


from classes.dados.dados import DataHandler as dh
from classes.dados.GaussianData import GausianDataHandler as gdh
from classes.dados.monteCarlo import MonteCarlo as MC
from classes.modelos.GausianCov import GaussianCovModel as GCM
from classes.modelos.GausianTrad import GausianTraditionalModel as GTM
from classes.modelos.GausianGreg import GausianGredModel as GGM
from classes.modelos.GausFriedman import GausianFriedman as GFM
from classes.modelos.MQOtradicional import MQOT
from classes.modelos.GausNaiveBay import GausNaiveBay as GNB




# Armazenar dados 

# lista de lambdas para teste
lmbdas = [0.25, 0.5, 0.75]
title = "Classificador Gaussiano Regularizado (Friedman λ = {lmbda})"


dados_ = {
    "MQO tradicional": [],
    "Classificador Gaussiano Tradicional":[],
    "Classificador Gaussiano (Cov. de todo cj. treino)":[],
    "Classificador Gaussiano (Cov. Agregada)":[],
    "Classificador de Bayes Ingenuo (Naive Bayes Classifier)":[],
}
for i in lmbdas:
    dados_[title.format(lmbda = i)] = []



# mqo tradicional
def MAOtradicional(x_treino, y_treino, x_teste, y_teste):
    y_teste = np.argmax(y_teste.T, axis=1) + 1
    modelo = MQOT(x_treino.T, y_treino.T)
    modelo.trainModel()
    predict = modelo.predict(x_teste.T)

    # avaliar resposta
    aval = predict.flatten() - y_teste.flatten()
    accuracy  = len(aval[aval == 0])/ len(y_teste.flatten())
    return accuracy

# Classificador tradicional Gaussiano
def GaussianoTradicional(x_treino, y_treino, x_test, y_test):
    modelo = GTM(x_treino, y_treino, np.array([1,2,3,4,5]))
    modelo.getStatistcs()
    predict = modelo.predict(x_test)

    # avaliar resposta
    aval = predict- y_test.flatten()
    accuracy  = len(aval[aval == 0])/ len(y_test.flatten())
    return accuracy

def GaussianoCovariancia(x_treino, y_treino, x_test, y_test):
    modelo = GCM(x_treino, y_treino, np.array([1,2,3,4,5]))
    modelo.getStatistcs()
    predict = modelo.predict(x_test)

    # avaliar resposta
    aval = predict- y_test.flatten()
    accuracy  = len(aval[aval == 0])/ len(y_test.flatten())
    return accuracy

def GaussianoAgregado(x_treino, y_treino, x_test, y_test):
    modelo = GGM(x_treino, y_treino, np.array([1,2,3,4,5]))
    modelo.getStatistcs()
    predict = modelo.predict(x_test)

    # avaliar resposta
    aval = predict- y_test.flatten()
    accuracy  = len(aval[aval == 0])/ len(y_test.flatten())
    return accuracy

def GaussianoNaive(x_treino, y_treino, x_test, y_test):
    modelo = GNB(x_treino, y_treino, np.array([1,2,3,4,5]))
    modelo.getStatistcs()
    predict = modelo.predict(x_test)

    # avaliar resposta
    aval = predict- y_test.flatten()
    accuracy  = len(aval[aval == 0])/ len(y_test.flatten())
    return accuracy

def GaussianoRegular(x_treino, y_treino, x_test, y_test, lmbda):
    modelo = GFM(x_treino, y_treino,np.array([1,2,3,4,5]),  lmbda)
    modelo.getStatistcs()
    predict = modelo.predict(x_test)

    # avaliar resposta
    aval = predict- y_test.flatten()
    accuracy  = len(aval[aval == 0])/ len(y_test.flatten())
    return accuracy  



import concurrent.futures
from IPython.display import clear_output
data_handler = gdh(r"EMGsDataset.csv")
x1, y1 = data_handler.x, data_handler.y
data_handler.setMatrizes()
x2, y2 = data_handler.x, data_handler.y
y2 = gdh.yAsMatrix(y2)
x2, y2 = x2.T, y2.T

def run_round(i):
    try:
        # separa dados para treino e teste
        x_treino, y_treino, x_test, y_test = MC.partition2(x1, y1)
        x_treino2, y_treino2, x_test2, y_test2 = MC.partition2(x2, y2)

        resultado = {}
        resultado["MQO tradicional"] = MAOtradicional(x_treino2, y_treino2, x_test2, y_test2)
        resultado["Classificador Gaussiano Tradicional"] = GaussianoTradicional(x_treino, y_treino, x_test, y_test)
        resultado["Classificador Gaussiano (Cov. de todo cj. treino)"] = GaussianoCovariancia(x_treino, y_treino, x_test, y_test)
        resultado["Classificador Gaussiano (Cov. Agregada)"] = GaussianoAgregado(x_treino, y_treino, x_test, y_test)
        resultado["Classificador de Bayes Ingenuo (Naive Bayes Classifier)"] = GaussianoNaive(x_treino, y_treino, x_test, y_test)

        for lamb in lmbdas:
            resultado[title.format(lmbda=lamb)] = GaussianoRegular(x_treino, y_treino, x_test, y_test, lamb)

        return resultado
    except Exception as e:
        return {"error": str(e)}
todos_resultados = []

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(run_round, i) for i in range(20)]
    for i, future in enumerate(concurrent.futures.as_completed(futures)):
        
        resultado = future.result()
        todos_resultados.append(resultado)
# Processa todos_resultados conforme necessário
print(todos_resultados)