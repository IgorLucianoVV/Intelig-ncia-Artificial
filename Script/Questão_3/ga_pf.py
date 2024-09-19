import numpy as np

class Ga_pf:
    def __init__(self, dim, pop_size, max_generation, restricoes, crossover_rate=0.9, mutation_rate=0.02, eta=20, sigma=0.1):
        self.dim = dim  # Dimensão do problema
        self.pop_size = pop_size
        self.max_generation = max_generation
        self.restricoes = restricoes
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.eta = eta  # Parámetro do Simulated Binary Crossover (SBX)
        self.sigma = sigma  # Desvio padrão da mutação gaussiana
    
    # Função objetiva (exemplo: função de Rastrigin)
    def funcao_objetiva(self, x):
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    # Função de aptidão Ψ(x) = f(x) + 1
    def f_apt(self, x):
        return self.funcao_objetiva(x) + 1
    
    # Geração de um indivíduo (vetor em ponto flutuante)
    def gerar_individuo(self):
        return np.random.uniform(self.restricoes[0], self.restricoes[1], self.dim)

    # Geração da população inicial
    def gerar_populacao(self):
        return np.array([self.gerar_individuo() for _ in range(self.pop_size)])
    
    # Seleção por torneio
    def torneio(self, populacao, aptidoes, k=3):
        participantes = np.random.choice(range(self.pop_size), k, replace=False)
        aptidoes_participantes = aptidoes[participantes]
        vencedor = participantes[np.argmin(aptidoes_participantes)]  # O menor valor vence (problema de minimização)
        return populacao[vencedor]
    
    # Crossover Simulated Binary Crossover (SBX)
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
    
    # Mutação Gaussiana
    def mutacao_gaussiana(self, individuo):
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                individuo[i] += np.random.normal(0, self.sigma)
                # Garantir que o valor esteja dentro das restrições
                individuo[i] = np.clip(individuo[i], self.restricoes[0], self.restricoes[1])
        return individuo
    
    # Criar nova geração
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
    
    # Critério de convergência: aptidão mínima
    def verificar_criterio(self, aptidoes, threshold=1e-6):
        return np.min(aptidoes) <= threshold

    # Execução do algoritmo
    def executar(self):
        populacao = self.gerar_populacao()
        
        for geracao in range(self.max_generation):
            aptidoes = np.array([self.f_apt(ind) for ind in populacao])
            
            if self.verificar_criterio(aptidoes):
                print(f"Convergência alcançada na geração {geracao}")
                break
            
            populacao = self.new_generation(populacao, aptidoes)
        
        # Melhor solução encontrada
        aptidoes = np.array([self.f_apt(ind) for ind in populacao])
        melhor_individuo = populacao[np.argmin(aptidoes)]  # Indivíduo com menor aptidão
        melhor_aptidao = np.min(aptidoes)
        return melhor_individuo, melhor_aptidao


if __name__ == "__main__":
    # Exemplo de uso
    alg_gen = Ga_pf(dim=5, pop_size=100, max_generation=1000, restricoes=(-10, 10), crossover_rate=0.9, mutation_rate=0.02, eta=20, sigma=0.1)
    melhor_individuo, melhor_aptidao = alg_gen.executar()
    print(f"Melhor aptidão (min): {melhor_aptidao}")
    print(f"Melhor indivíduo: {melhor_individuo}")
