import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Função de perturbação para duas variáveis (x1 e x2)
def perturb(x, e):
    x1, x2 = x
    return (np.random.uniform(low=x1 - e, high=x1 + e), 
            np.random.uniform(low=x2 - e, high=x2 + e))

# Função f(x1, x2) = x1 * cos(x1) / 20 + 2 * e^(-(x1^2) - (x2 - 1)^2) + 0.01 * x1 * x2
def f(x):
    x1, x2 = x
    return (x1 * np.cos(x1) / 20) + 2 * np.exp(-(x1**2) - (x2 - 1)**2) + 0.01 * x1 * x2

# Eixos x1 e x2 para o gráfico da função
x1_axis = np.linspace(-10, 10, 100)
x2_axis = np.linspace(-10, 10, 100)
X1, X2 = np.meshgrid(x1_axis, x2_axis)
Z = f([X1, X2])

# Criando a figura e o gráfico 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plotando a superfície da função
ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none', alpha=0.7)

# Ponto inicial
x_opt = (-10, -10)
f_opt = f(x_opt)

# Visualizando o ponto inicial no gráfico 3D
ax.scatter(*x_opt, f_opt, color='r', marker='x', s=100, label='Ponto Inicial')

# Parâmetros do algoritmo
e = 0.1  # Tamanho da perturbação
max_it = 1000  # Número máximo de iterações
max_viz = 100  # Número máximo de vizinhos testados por iteração
melhoria = True  # Controle para verificar se houve melhoria
i = 0  # Contador de iterações
valores = [f_opt]  # Lista para armazenar os valores ótimos

# Loop de busca local para maximização
while i < max_it and melhoria:
    melhoria = False
    for j in range(max_viz):
        # Gera um candidato perturbado
        x_cand = perturb(x_opt, e)
        f_cand = f(x_cand)
        
        # Queremos maximizar, então aceitamos se f_cand > f_opt
        if f_cand > f_opt:
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
ax.set_title('Busca Local em $f(x_1, x_2)$')
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
