import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Função de perturbação para duas variáveis (x1 e x2)
def perturb(x, e):
    x1, x2 = x
    return (np.random.uniform(low=x1 - e, high=x1 + e), 
            np.random.uniform(low=x2 - e, high=x2 + e))

# Função objetivo (Ackley modificada)
def ackley_func(x):
    x1, x2 = x
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2)))
    return term1 + term2 + 20 + np.e

# Função de busca local
def local_search(f, x1_range, x2_range, x_start, e=0.1, max_it=1000, max_viz=100):
    # Eixos x1 e x2 para o gráfico da função
    x1_axis = np.linspace(x1_range[0], x1_range[1], 100)
    x2_axis = np.linspace(x2_range[0], x2_range[1], 100)
    X1, X2 = np.meshgrid(x1_axis, x2_axis)
    Z = f([X1, X2])

    # Criando a figura e o gráfico 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plotando a superfície da função
    ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none', alpha=0.7)

    # Ponto inicial
    x_opt = x_start
    f_opt = f(x_opt)

    # Visualizando o ponto inicial no gráfico 3D
    ax.scatter(*x_opt, f_opt, color='r', marker='x', s=100, label='Ponto Inicial')

    # Inicializando variáveis de controle
    melhoria = True  # Controle para verificar se houve melhoria
    i = 0  # Contador de iterações
    valores = [f_opt]  # Lista para armazenar os valores ótimos

    # Loop de busca local
    while i < max_it and melhoria:
        melhoria = False
        for j in range(max_viz):
            # Gera um candidato perturbado
            x_cand = perturb(x_opt, e)
            f_cand = f(x_cand)
            
            # Queremos minimizar, então aceitamos se f_cand < f_opt
            if f_cand < f_opt:
                x_opt = x_cand
                f_opt = f_cand
                valores.append(f_opt)
                melhoria = True
                
                # Visualizando o novo ponto durante o processo de busca
                ax.scatter(*x_opt, f_opt, color='r', marker='x', s=50)
                break
        i += 1

    # Destacar o ponto final encontrado no gráfico 3D
    ax.scatter(*x_opt, f_opt, color='g', marker='o', s=150, label='Ponto Final')

    # Adicionando rótulos
    ax.set_title('Busca Local na Função Ackley')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f(x_1, x_2)$')
    ax.legend()

    plt.show()

    # Gráfico de convergência
    plt.plot(valores)
    plt.xlabel('Iterações')
    plt.ylabel('f(x1, x2)')
    plt.title('Convergência de f(x1, x2)')
    plt.show()

# Definir o intervalo de busca e o ponto inicial
x1_range = (-8, 8)
x2_range = (-8, 8)
x_start = (0, 0)  # Ponto inicial

# Executar a busca local
local_search(ackley_func, x1_range, x2_range, x_start, e=0.1, max_it=1000, max_viz=100)
