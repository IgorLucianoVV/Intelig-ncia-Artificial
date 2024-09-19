import numpy as np
import pandas as pd

class Ga_canonico:
    def __init__(self, bit, pop_size, max_generation, restricoes, crossover_rate=0.85,
                 mutation_rate=0.01, A=10, p=20, n_cross=2, elitismo=True):
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

    # Função objetiva para minimizar (exemplo: função de Rastrigin)
    def funcao_objetiva(self, x):
        return self.A * len(x) + np.sum(x**2 - self.A * np.cos(2 * np.pi * x))
    
    # Função de aptidão (diretamente a função objetivo para minimização)
    def f_apt(self, x):
        return self.funcao_objetiva(x)
    
    # Geração de indivíduo (sequência binária aleatória)
    def gerar_individuo(self):
        return np.random.randint(0, 2, self.p * self.bit)

    # Geração da população inicial
    def gerar_populacao(self):
        return np.array([self.gerar_individuo() for _ in range(self.pop_size)])
    
    # Função para converter de bit para decimal
    def phi(self, segment):
        decimal = int("".join(map(str, segment)), 2)  # Conversão direta de binário para decimal
        max_val = 2 ** len(segment) - 1
        return self.restricoes[0] + (self.restricoes[1] - self.restricoes[0]) * decimal / max_val
    
    # Converter um indivíduo inteiro
    def convert(self, individual):
        return np.array([self.phi(individual[i*self.bit: (i+1)*self.bit]) for i in range(self.p)])

    # Converter toda a população
    def mass_convert(self, populacao):
        return np.array([self.convert(individual) for individual in populacao])
    
    # Crossover com ponto de corte
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
    
    # Seleção por roleta para minimização
    def roleta(self, populacao, aptidoes):
        # Normalizando as aptidões para valores positivos
        aptidoes_invertidas = 1 / (aptidoes + 1e-10)  # Evitar divisão por zero
        soma_aptidoes = np.sum(aptidoes_invertidas)
        probabilidades = aptidoes_invertidas / soma_aptidoes
        
        # Selecionar um indivíduo com base na roleta
        r = np.random.uniform()
        acumulado = 0.0
        for i, prob in enumerate(probabilidades):
            acumulado += prob
            if acumulado > r:
                return populacao[i]
        return populacao[-1]
    
    # Mutação
    def mutacao(self, individuo):
        mutantes = np.random.rand(len(individuo)) < self.mutation_rate
        individuo[mutantes] = 1 - individuo[mutantes]
        return individuo
    
    # Criar nova geração
    def new_generation(self, populacao, aptidoes):
        nova_populacao = []
        elite = None

        # Aplicar elitismo: manter o melhor indivíduo
        if self.elitismo:
            elite = populacao[np.argmin(aptidoes)]  # Menor aptidão = melhor para minimização

        while len(nova_populacao) < self.pop_size:
            pai1 = self.roleta(populacao, aptidoes)
            pai2 = self.roleta(populacao, aptidoes)
            
            if np.random.rand() < self.crossover_rate:
                f1, f2 = self.crossover(pai1, pai2)
            else:
                f1, f2 = np.copy(pai1), np.copy(pai2)
            
            nova_populacao.extend([self.mutacao(f1), self.mutacao(f2)])
        
        # Substituir o pior indivíduo pelo elite, se habilitado
        if self.elitismo:
            pior_idx = np.argmax([self.f_apt(self.convert(ind)) for ind in nova_populacao])
            nova_populacao[pior_idx] = elite
        
        return np.array(nova_populacao[:self.pop_size])
    
    # Verificar critério de convergência (se algum indivíduo satisfaz uma aptidão mínima)
    def verificar_criterio(self, aptidoes):
        return np.min(aptidoes) <= 1e-6  # Critério de convergência para minimização

    def executar(self):
        populacao = self.gerar_populacao()
        
        for geracao in range(self.max_generation):
            populacao_dec = self.mass_convert(populacao)
            aptidoes = np.array([self.f_apt(ind) for ind in populacao_dec])
            
            if self.verificar_criterio(aptidoes):
                print(f"Convergência alcançada na geração {geracao}")
                break
            
            populacao = self.new_generation(populacao, aptidoes)
        
        # Melhor solução encontrada
        populacao_dec = self.mass_convert(populacao)
        aptidoes = np.array([self.f_apt(ind) for ind in populacao_dec])
        melhor_individuo = populacao[np.argmin(aptidoes)]  # Indivíduo com menor aptidão
        melhor_aptidao = np.min(aptidoes)
        return melhor_individuo, melhor_aptidao

# Função para executar 100 rodadas e calcular estatísticas
def rodadas_algoritmo(algoritmo, num_rodadas=100):
    aptidoes = []
    for _ in range(num_rodadas):
        _, melhor_aptidao = algoritmo.executar()
        aptidoes.append(melhor_aptidao)

    # Estatísticas
    aptidoes = np.array(aptidoes)
    menor_valor = np.min(aptidoes)
    maior_valor = np.max(aptidoes)
    media = np.mean(aptidoes)
    desvio_padrao = np.std(aptidoes)

    return menor_valor, maior_valor, media, desvio_padrao

if __name__ == "__main__":
    # Configurações para o primeiro algoritmo genético
    alg_gen1 = Ga_canonico(bit=20, pop_size=100, max_generation=1000, restricoes=(-10, 10), n_cross=1, elitismo=True)

    # Configurações para o segundo algoritmo genético (exemplo com mutação maior)
    alg_gen2 = Ga_canonico(bit=20, pop_size=100, max_generation=1000, restricoes=(-10, 10), n_cross=1, elitismo=False, mutation_rate=0.05)

    # Executar 100 rodadas para ambos os algoritmos
    stats1 = rodadas_algoritmo(alg_gen1)
    stats2 = rodadas_algoritmo(alg_gen2)

    # Tabela de resultados
    df = pd.DataFrame({
        'Algoritmo': ['Algoritmo 1', 'Algoritmo 2'],
        'Menor Aptidão': [stats1[0], stats2[0]],
        'Maior Aptidão': [stats1[1], stats2[1]],
        'Média Aptidão': [stats1[2], stats2[2]],
        'Desvio Padrão': [stats1[3], stats2[3]]
    })

    print(df)
