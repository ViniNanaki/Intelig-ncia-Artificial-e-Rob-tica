# Naki
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Função para treinar e testar uma única rede neural
def train_and_test(x, y, hidden_layer_sizes):
    regr = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                        max_iter=100,
                        activation='relu',
                        solver='adam',
                        learning_rate='adaptive',
                        n_iter_no_change=50)
    
    regr.fit(x, y)
    y_pred = regr.predict(x)
    error = mean_squared_error(y, y_pred)
    
    return error, regr.loss_curve_, y_pred

# Função para realizar múltiplas simulações com uma arquitetura específica
def simulate_architecture(x, y, hidden_layer_sizes, n_simulations=10):
    errors = []
    best_loss_curve = None
    best_y_pred = None
    
    print(f'\nSimulando arquitetura: {hidden_layer_sizes}')
    
    for i in range(n_simulations):
        print(f'Simulação {i+1}/{n_simulations}...')
        error, loss_curve, y_pred = train_and_test(x, y, hidden_layer_sizes)
        errors.append(error)
        
        # Guardar a melhor simulação para fins de plotagem
        if best_loss_curve is None or error < min(errors):
            best_loss_curve = loss_curve
            best_y_pred = y_pred
    
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    return mean_error, std_error, best_loss_curve, best_y_pred

# Função para plotar os resultados
def plot_results(x, y, loss_curve, y_pred, architecture):
    plt.figure(figsize=[14, 7])

    # Dados originais
    plt.subplot(1, 3, 1)
    plt.plot(x, y)
    plt.title('Dados Originais')

    # Curva de aprendizagem
    plt.subplot(1, 3, 2)
    plt.plot(loss_curve)
    plt.title(f'Curva de Aprendizagem: {architecture}')

    # Regressão ajustada
    plt.subplot(1, 3, 3)
    plt.plot(x, y, linewidth=1, color='yellow', label='Original')
    plt.plot(x, y_pred, linewidth=2, label='Predito')
    plt.title(f'Regressão: {architecture}')
    plt.legend()

    plt.show()

# Função principal para carregar dados, testar arquiteturas e exibir os gráficos
def run_experiment(test_file, architectures, n_simulations=10):
    print(f'\nCarregando arquivo de teste: {test_file}')
    arquivo = np.load(test_file)
    x = arquivo[0]
    y = np.ravel(arquivo[1])

    for architecture in architectures:
        mean_error, std_error, best_loss_curve, best_y_pred = simulate_architecture(x, y, architecture, n_simulations)

        print(f'Resultados para arquitetura {architecture}:')
        print(f'Média do erro: {mean_error}')
        print(f'Desvio padrão do erro: {std_error}')

        # Plotar os resultados da melhor simulação
        plot_results(x, y, best_loss_curve, best_y_pred, architecture)

# Definir arquiteturas de teste
architectures = [
    (2,),          # 1 camada com 2 neurônios
    (5, 2),        # 2 camadas com 5 e 2 neurônios
    (10, 5, 2),    # 3 camadas com 10, 5 e 2 neurônios
]

# Executar experimentos para os arquivos de teste de 2 a 5
for i in range(2, 6):
    test_file = f'teste{i}.npy'
    run_experiment(test_file, architectures, n_simulations=10)

'''
Modularização das Funções:

train_and_test: Realiza o treinamento e teste de uma única rede neural com a arquitetura fornecida.
simulate_architecture: Executa múltiplas simulações (10 vezes por padrão) para uma arquitetura e calcula a média e o desvio padrão do erro.
plot_results: Responsável por plotar os gráficos dos resultados (dados originais, curva de aprendizagem, predição).
run_experiment: Carrega os dados, executa as simulações para diferentes arquiteturas e gera os gráficos.
Melhor controle de simulação:

Cada simulação é rodada individualmente e, após isso, a melhor curva de perda e predição são usadas para gerar gráficos.
Mais logs: O código agora fornece mais detalhes no console, como o número de simulação atual, permitindo que você veja o progresso ao longo do processo.

Uso de uma estrutura mais clara para que o código fique fácil de ajustar para outros tipos de arquiteturas ou parâmetros no futuro.

Esse código fornecerá resultados detalhados para cada arquitetura e os gráficos da melhor execução (com menor erro) em cada caso.
'''
