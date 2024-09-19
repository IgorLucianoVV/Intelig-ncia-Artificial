import numpy as np
import pandas as pd

class Ga_canonico:
    # Implementação do primeiro algoritmo genético (já fornecida anteriormente)
    def __init__(self, bit, pop_size, max_generation, restricoes, crossover_rate=0.9,
                 mutation_rate=0.02, A=10, p=20, n_cross=2, elitismo=True):
        self.A = A
        self.p = p
        self.bit = bit
        self.restricoes = restricoes
        self.pop_size = pop_size
        self.max_generation = max_generation
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_cross = n_cross
        self.elitismo = elitismo  # Habilitar elitismo

    def funcao_objetiva(self, x):
        return self.A * len(x) + np.sum(x**2 - self.A * np.cos(2 * np.pi * x))
    
    def f_apt(self, x):
        return self.funcao_objetiva(x)
    
    def gerar_individuo(self):
        return np.random.randint(0, 2, self.p * self.bit)

    def gerar_populacao(self):
        return np.array([self.gerar_individuo() for _ in range(self.pop_size)])
    
    def phi(self, segment):
        decimal = int("".join(map(str, segment)), 2)
        max_val = 2 ** len(segment) - 1
        return self.restricoes[0] + (self.restricoes[1] - self.restricoes[0]) * decimal / max_val
    
    def convert(self, individual):
        return np.array([self.phi(individual[i*self.bit: (i+1)*self.bit]) for i in range(self.p)])

    def mass_convert(self, populacao):
        return np.array([self.convert(individual) for individual in populacao])
    
    def crossover(self, pai1, pai2):
        pontos_corte = np.sort(np.random.choice(len(pai1), self.n_cross, replace=False))
        f1, f2 = np.copy(pai1), np.copy(pai2)

        for i in range(self.n_cross):
            inicio = pontos_corte[i]
            fim = pontos_corte[i+1] if i+1 < len(pontos_corte) else len(pai1)

            if i % 2 == 0:
                f1[inicio:fim] = pai2[inicio:fim]
                f2[inicio:fim] = pai1[inicio:fim]

        return f1, f2
    
    def roleta(self, populacao, aptidoes):
        aptidoes_invertidas = 1 / (aptidoes + 1e-10)
        soma_aptidoes = np.sum(aptidoes_invertidas)
        probabilidades = aptidoes_invertidas / soma_aptidoes
        
        r = np.random.uniform()
        acumulado = 0.0
        for i, prob in enumerate(probabilidades):
            acumulado += prob
            if acumulado > r:
                return populacao[i]
        return populacao[-1]
    
    def mutacao(self, individuo):
        mutantes = np.random.rand(len(individuo)) < self.mutation_rate
        individuo[mutantes] = 1 - individuo[mutantes]
        return individuo
    
    def new_generation(self, populacao, aptidoes):
        nova_populacao = []
        elite = None

        if self.elitismo:
            elite = populacao[np.argmin(aptidoes)]

        while len(nova_populacao) < self.pop_size:
            pai1 = self.roleta(populacao, aptidoes)
            pai2 = self.roleta(populacao, aptidoes)
            
            if np.random.rand() < self.crossover_rate:
                f1, f2 = self.crossover(pai1, pai2)
            else:
                f1, f2 = np.copy(pai1), np.copy(pai2)
            
            nova_populacao.extend([self.mutacao(f1), self.mutacao(f2)])
        
        if self.elitismo:
            pior_idx = np.argmax([self.f_apt(self.convert(ind)) for ind in nova_populacao])
            nova_populacao[pior_idx] = elite
        
        return np.array(nova_populacao[:self.pop_size])
    
    def verificar_criterio(self, aptidoes):
        return np.min(aptidoes) <= 1e-6

    def executar(self):
        populacao = self.gerar_populacao()
        
        for geracao in range(self.max_generation):
            populacao_dec = self.mass_convert(populacao)
            aptidoes = np.array([self.f_apt(ind) for ind in populacao_dec])
            
            if self.verificar_criterio(aptidoes):
                break
            
            populacao = self.new_generation(populacao, aptidoes)
        
        populacao_dec = self.mass_convert(populacao)
        aptidoes = np.array([self.f_apt(ind) for ind in populacao_dec])
        melhor_individuo = populacao[np.argmin(aptidoes)]
        melhor_aptidao = np.min(aptidoes)
        return melhor_individuo, melhor_aptidao


class Ga_pf:
    def __init__(self, dim, pop_size, max_generation, restricoes, crossover_rate=0.9, mutation_rate=0.02, eta=20, sigma=0.1):
        self.dim = dim
        self.pop_size = pop_size
        self.max_generation = max_generation
        self.restricoes = restricoes
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.eta = eta
        self.sigma = sigma
    
    def funcao_objetiva(self, x):
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    def f_apt(self, x):
        return self.funcao_objetiva(x) + 1
    
    def gerar_individuo(self):
        return np.random.uniform(self.restricoes[0], self.restricoes[1], self.dim)

    def gerar_populacao(self):
        return np.array([self.gerar_individuo() for _ in range(self.pop_size)])
    
    def torneio(self, populacao, aptidoes, k=3):
        participantes = np.random.choice(range(self.pop_size), k, replace=False)
        aptidoes_participantes = aptidoes[participantes]
        vencedor = participantes[np.argmin(aptidoes_participantes)]
        return populacao[vencedor]
    
    def sbx(self, pai1, pai2):
        beta = np.empty(self.dim)
        for i in range(self.dim):
            u = np.random.rand()
            if u <= 0.5:
                beta[i] = (2 * u) ** (1 / (self.eta + 1))
            else:
                beta[i] = (1 / (2 * (1 - u))) ** (1 / (self.eta + 1))
        
        f1 = 0.5 * ((1 + beta) * pai1 + (1 - beta) * pai2)
        f2 = 0.5 * ((1 - beta) * pai1 + (1 + beta) * pai2)
        return f1, f2
    
    def mutacao_gaussiana(self, individuo):
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                individuo[i] += np.random.normal(0, self.sigma)
                individuo[i] = np.clip(individuo[i], self.restricoes[0], self.restricoes[1])
        return individuo
    
    def new_generation(self, populacao, aptidoes):
        nova_populacao = []
        
        while len(nova_populacao) < self.pop_size:
            pai1 = self.torneio(populacao, aptidoes)
            pai2 = self.torneio(populacao, aptidoes)
            
            if np.random.rand() < self.crossover_rate:
                f1, f2 = self.sbx(pai1, pai2)
            else:
                f1, f2 = np.copy(pai1), np.copy(pai2)
            
            nova_populacao.extend([self.mutacao_gaussiana(f1), self.mutacao_gaussiana(f2)])
        
        return np.array(nova_populacao[:self.pop_size])
    
    def verificar_criterio(self, aptidoes, threshold=1e-6):
        return np.min(aptidoes) <= threshold

    def executar(self):
        populacao = self.gerar_populacao()
        
        for geracao in range(self.max_generation):
            aptidoes = np.array([self.f_apt(ind) for ind in populacao])
            
            if self.verificar_criterio(aptidoes):
                break
            
            populacao = self.new_generation(populacao, aptidoes)
        
        aptidoes = np.array([self.f_apt(ind) for ind in populacao])
        melhor_individuo = populacao[np.argmin(aptidoes)]
        melhor_aptidao = np.min(aptidoes)
        return melhor_individuo, melhor_aptidao


def realizar_rodadas(algoritmo_class, num_rodadas=100, **kwargs):
    resultados = []
    for _ in range(num_rodadas):
        alg = algoritmo_class(**kwargs)
        _, melhor_aptidao = alg.executar()
        resultados.append(melhor_aptidao)
    
    return {
        'Menor aptidão': np.min(resultados),
        'Maior aptidão': np.max(resultados),
        'Média de aptidão': np.mean(resultados),
        'Desvio-padrão de aptidão': np.std(resultados)
    }


# Configurações para ambos os algoritmos
config_canonico = {
    'bit': 16,
    'pop_size': 100,
    'max_generation': 500,
    'restricoes': (-10, 10),
    'p': 20
}

config_pf = {
    'dim': 5,
    'pop_size': 100,
    'max_generation': 500,
    'restricoes': (-10, 10)
}

# Executar rodadas
resultados_canonico = realizar_rodadas(Ga_canonico, num_rodadas=100, **config_canonico)
resultados_pf = realizar_rodadas(Ga_pf, num_rodadas=100, **config_pf)

# Exibir resultados comparativos em uma tabela
df = pd.DataFrame({
    'GA Canônico': resultados_canonico,
    'GA Não Canônico': resultados_pf
})

print(df)
