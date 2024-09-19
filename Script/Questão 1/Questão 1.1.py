import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definição da função objetivo f(x1, x2) = x1^2 + x2^2
def funcao_objetivo(x1, x2):
    return x1**2 + x2**2

# Algoritmo Hill Climbing
def hill_climbing(limites, epsilon, max_it):
    xbest = np.array([limite[0] for limite in limites])
    f_best = funcao_objetivo(*xbest)
    
    for _ in range(max_it):
        x_candidato = np.array([
            np.clip(np.random.uniform(low=xbest[0] - epsilon, high=xbest[0] + epsilon), limites[0][0], limites[0][1]),
            np.clip(np.random.uniform(low=xbest[1] - epsilon, high=xbest[1] + epsilon), limites[1][0], limites[1][1])
        ])
        f_candidato = funcao_objetivo(*x_candidato)
        
        if f_candidato < f_best:
            xbest = x_candidato
            f_best = f_candidato

    return xbest, f_best

# Algoritmo Local Random Search
def local_random_search(limites, sigma, max_it):
    xbest = np.random.uniform(low=[limite[0] for limite in limites], high=[limite[1] for limite in limites])
    f_best = funcao_objetivo(*xbest)
    
    for _ in range(max_it):
        x_candidato = np.clip(np.random.normal(loc=xbest, scale=sigma), [limite[0] for limite in limites], [limite[1] for limite in limites])
        f_candidato = funcao_objetivo(*x_candidato)
        
        if f_candidato < f_best:
            xbest = x_candidato
            f_best = f_candidato

    return xbest, f_best

# Algoritmo Global Random Search
def global_random_search(limites, max_it):
    xbest = np.random.uniform(low=[limite[0] for limite in limites], high=[limite[1] for limite in limites])
    f_best = funcao_objetivo(*xbest)
    
    for _ in range(max_it):
        x_candidato = np.random.uniform(low=[limite[0] for limite in limites], high=[limite[1] for limite in limites])
        f_candidato = funcao_objetivo(*x_candidato)
        
        if f_candidato < f_best:
            xbest = x_candidato
            f_best = f_candidato

    return xbest, f_best

# Classe para rodar 100 execuções de um algoritmo de otimização
class TesteRounds:
    def __init__(self, algoritmo, limites, epsilon=None, sigma=None, max_it=1000):
        self.algoritmo = algoritmo
        self.limites = limites
        self.epsilon = epsilon
        self.sigma = sigma
        self.max_it = max_it

    def rounds(self, n_execucoes=100):
        lista_solucoes = []
        lista_ress = []
        
        for i in range(n_execucoes):
            if self.algoritmo == 'hill_climbing':
                solucao, valor = hill_climbing(self.limites, self.epsilon, self.max_it)
            elif self.algoritmo == 'local_random_search':
                solucao, valor = local_random_search(self.limites, self.sigma, self.max_it)
            elif self.algoritmo == 'global_random_search':
                solucao, valor = global_random_search(self.limites, self.max_it)
            else:
                raise ValueError("Algoritmo desconhecido. Escolha entre 'hill_climbing', 'local_random_search' ou 'global_random_search'.")
            
            lista_solucoes.append(solucao)
            lista_ress.append(valor)
        
        print(f"Execução {i + 1} de {n_execucoes}")

        return lista_solucoes, lista_ress

# Parâmetros da execução
limites = [(-100, 100), (-100, 100)]
epsilon = 0.1
sigma = 0.1
max_it = 1000
n_execucoes = 100

# Executar Hill Climbing
hc_test = TesteRounds('hill_climbing', limites, epsilon=epsilon, max_it=max_it)
hc_solucoes, hc_resultados = hc_test.rounds(n_execucoes=n_execucoes)
melhor_hc = hc_solucoes[np.argmin(hc_resultados)]

# Executar Local Random Search
lrs_test = TesteRounds('local_random_search', limites, sigma=sigma, max_it=max_it)
lrs_solucoes, lrs_resultados = lrs_test.rounds(n_execucoes=n_execucoes)
melhor_lrs = lrs_solucoes[np.argmin(lrs_resultados)]

# Executar Global Random Search
grs_test = TesteRounds('global_random_search', limites, max_it=max_it)
grs_solucoes, grs_resultados = grs_test.rounds(n_execucoes=n_execucoes)
melhor_grs = grs_solucoes[np.argmin(grs_resultados)]

# Criar a figura e os gráficos
fig = plt.figure(figsize=(18, 12))

# Gráfico 3D Hill Climbing
ax1 = fig.add_subplot(141, projection='3d')
x1 = np.linspace(-100, 100, 400)
x2 = np.linspace(-100, 100, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = funcao_objetivo(X1, X2)
ax1.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none', alpha=0.7)
ax1.contour(X1, X2, Z, 10, cmap='viridis', offset=0)
ax1.scatter(melhor_hc[0], melhor_hc[1], funcao_objetivo(*melhor_hc), color='r', marker='o', s=100, label='Ponto Mínimo Hill Climbing')
ax1.set_title('Hill Climbing')
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_zlabel('$f(x_1, x_2)$')
ax1.legend()

# Gráfico 3D Local Random Search
ax2 = fig.add_subplot(142, projection='3d')
ax2.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none', alpha=0.7)
ax2.contour(X1, X2, Z, 10, cmap='viridis', offset=0)
ax2.scatter(melhor_lrs[0], melhor_lrs[1], funcao_objetivo(*melhor_lrs), color='g', marker='o', s=100, label='Ponto Mínimo Local Random Search')
ax2.set_title('Local Random Search')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_zlabel('$f(x_1, x_2)$')
ax2.legend()

# Gráfico 3D Global Random Search
ax3 = fig.add_subplot(143, projection='3d')
ax3.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none', alpha=0.7)
ax3.contour(X1, X2, Z, 10, cmap='viridis', offset=0)
ax3.scatter(melhor_grs[0], melhor_grs[1], funcao_objetivo(*melhor_grs), color='b', marker='o', s=100, label='Ponto Mínimo Global Random Search')
ax3.set_title('Global Random Search')
ax3.set_xlabel('$x_1$')
ax3.set_ylabel('$x_2$')
ax3.set_zlabel('$f(x_1, x_2)$')
ax3.legend()

plt.tight_layout()
plt.show()

# Imprimir os resultados
print("Hill Climbing:")
print("Melhor ponto encontrado:", melhor_hc)
print("Valor da função objetivo:", funcao_objetivo(*melhor_hc))
print()
print("Local Random Search:")
print("Melhor ponto encontrado:", melhor_lrs)
print("Valor da função objetivo:", funcao_objetivo(*melhor_lrs))
print()
print("Global Random Search:")
print("Melhor ponto encontrado:", melhor_grs)
print("Valor da função objetivo:", funcao_objetivo(*melhor_grs))
