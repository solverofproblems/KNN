#Aqui nós temos todas as bibliotecas necessárias para rodar o código.
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

#Aqui temos a base de dados que será usada. Essa base de dados contém os preços de casas. 
base_dados_casa = fetch_california_housing()
 
#Aqui nós temos as variáveis que serão usadas para prever o preço da casa.
X = base_dados_casa.data

#Aqui é a variável algo (target) que o modelo deve tentar prever.
Y = base_dados_casa.target

#Aqui nós temos a separação da base de dados, sendo que 20% será para treino, além disso, padronizei uma randomização de 42, para evitar discrepâncias extremas a cada vez que o código é rodado.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 42) 

#Aqui estamos padronizando os dados para usá-los.
scaler = StandardScaler()

#Aqui o modelo aprende a média e o desvio dos dados de treino.
X_train = scaler.fit_transform(X_train)

#Usa a mesma escala de treino aprendida para ajustar nos dados de teste.
X_test = scaler.transform(X_test)

#Guardamos os valores de erro em uma lista.
erros_gerados = []

#Aqui estamos definindo o vaior de K-Vizinhos a partir da estrutura de repetição "For". Isso é importante, pois, conseguimos verificar um valor médio que o K precisa ter para minimizar o erro de previsão.
for i in range(1, 21):
    knn_reg = KNeighborsRegressor(n_neighbors=i)
    knn_reg.fit(X_train, Y_train)
    previsoes = knn_reg.predict(X_test)

    #Aqui temos a modificação do percentual para valor em dólares. Desse jeito, o gráfico fica mais intuitivo
    erro = mean_absolute_error(Y_test, previsoes)
    erros_gerados.append(erro * 100000)

#Aqui estamos construindo o gráfico que será exibido.
plt.figure(figsize=(12, 8))
plt.plot(range(1, 21), erros_gerados, color='red', linestyle='dashed', marker='o')
plt.title('Erro médio VS Valor de K')
plt.xlabel('K')
plt.ylabel('Erro médio em dólares.')
plt.show()




