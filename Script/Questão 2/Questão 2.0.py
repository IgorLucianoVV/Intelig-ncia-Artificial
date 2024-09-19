import random
import math
import time

# Função para calcular o número de conflitos entre as rainhas
def calcular_conflitos(tabuleiro):
    conflitos = 0
    n = len(tabuleiro)
    for i in range(n):
        for j in range(i + 1, n):
            if tabuleiro[i] == tabuleiro[j] or abs(tabuleiro[i] - tabuleiro[j]) == abs(i - j):
                conflitos += 1
    return conflitos

# Função de aptidão f(x) = 28 - h(x)
def funcao_objetivo(tabuleiro):
    max_conflitos = 28  # Número máximo de pares de rainhas
    h_x = calcular_conflitos(tabuleiro)
    return max_conflitos - h_x  # Quanto maior, melhor

# Função para perturbar a solução de forma controlada
def perturbar_controlado(tabuleiro):
    nova_solucao = tabuleiro[:]
    col = random.randint(0, len(tabuleiro) - 1)
    direcao = random.choice([-1, 1])
    nova_linha = nova_solucao[col] + direcao
    if 0 <= nova_linha < len(tabuleiro):
        nova_solucao[col] = nova_linha
    return nova_solucao

# Funções de resfriamento
def resfriamento_geometrico(temperatura, decaimento):
    return decaimento * temperatura

def resfriamento_logaritmico(temperatura, decaimento):
    return temperatura / (1 + decaimento * math.sqrt(temperatura))

def resfriamento_linear(temperatura, delta_t):
    return temperatura - delta_t

# Função de Têmpera Simulada
def tempera_simulada(t_inicial, temperatura_final, iteracoes_por_temperatura, metodo_resfriamento, decaimento, perturbacao):
    n = 8
    solucao_atual = [random.randint(0, n - 1) for _ in range(n)]
    valor_objetivo_atual = funcao_objetivo(solucao_atual)
    temperatura = t_inicial
    iteracao = 0
    maximo_iteracoes = 100000  # Número máximo de iterações
    solucoes_encontradas = set()
    inicio_tempo = time.time()

    while temperatura > temperatura_final and iteracao < maximo_iteracoes:
        for _ in range(iteracoes_por_temperatura):
            nova_solucao = perturbacao(solucao_atual)
            valor_objetivo_novo = funcao_objetivo(nova_solucao)
            delta = valor_objetivo_novo - valor_objetivo_atual

            if delta > 0 or random.random() < math.exp(delta / temperatura):
                solucao_atual = nova_solucao
                valor_objetivo_atual = valor_objetivo_novo

            if valor_objetivo_atual == 28:
                # Normaliza a solução para evitar duplicatas devido a simetrias
                solucao_tuple = tuple(solucao_atual)
                if solucao_tuple not in solucoes_encontradas:
                    solucoes_encontradas.add(solucao_tuple)
                    print(f"Solução {len(solucoes_encontradas)} encontrada: {solucao_atual}")
                    # Verifica se todas as 92 soluções foram encontradas
                    if len(solucoes_encontradas) == 92:
                        fim_tempo = time.time()
                        print(f"Todas as 92 soluções encontradas em {fim_tempo - inicio_tempo:.2f} segundos.")
                        return solucoes_encontradas

        # Atualiza a temperatura
        if metodo_resfriamento == 'geometrico':
            temperatura = resfriamento_geometrico(temperatura, decaimento)
        elif metodo_resfriamento == 'logaritmico':
            temperatura = resfriamento_logaritmico(temperatura, decaimento)
        elif metodo_resfriamento == 'linear':
            temperatura = resfriamento_linear(temperatura, decaimento)

        iteracao += 1

    fim_tempo = time.time()
    print(f"Tempo total: {fim_tempo - inicio_tempo:.2f} segundos.")
    return solucoes_encontradas

# Função para testar os três métodos de resfriamento
def testar_metodos_resfriamento(t_inicial, temperatura_final, iteracoes_por_temperatura):
    metodos_resfriamento = ['linear', 'geometrico', 'logaritmico']
    for metodo in metodos_resfriamento:
        print(f"\nTestando método de resfriamento: {metodo.capitalize()}")
        if metodo == 'linear':
            delta_t = (t_inicial - temperatura_final) / 1000  # nt = 1000 conforme fórmula
            decaimento = delta_t
        elif metodo == 'geometrico':
            decaimento = 0.99  # Decaimento adequado para resfriamento geométrico
        elif metodo == 'logaritmico':
            decaimento = 0.01  # Decaimento adequado para resfriamento logarítmico
        else:
            decaimento = 0.01  # Valor padrão se necessário
        solucoes_encontradas = tempera_simulada(
            t_inicial,
            temperatura_final,
            iteracoes_por_temperatura,
            metodo,
            decaimento,
            perturbar_controlado
        )
        print(f"Total de soluções encontradas: {len(solucoes_encontradas)}\n{'-' * 30}")

# Executar o teste com os três métodos de resfriamento
t_inicial = 1000  # Temperatura inicial
temperatura_final = 0.001  # Temperatura final
iteracoes_por_temperatura = 1000  # Número de iterações por temperatura

testar_metodos_resfriamento(t_inicial, temperatura_final, iteracoes_por_temperatura)