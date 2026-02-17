from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

base_dados_casa = fetch_california_housing()
 
X = base_dados_casa.data
Y = base_dados_casa.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 42) 

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

erros_gerados = []

for i in range(1, 21):
    knn_reg = KNeighborsRegressor(n_neighbors=i)
    knn_reg.fit(X_train, Y_train)
    previsoes = knn_reg.predict(X_test)

    erro = mean_absolute_error(Y_test, previsoes)
    erros_gerados.append(erro * 100000)

plt.figure(figsize=(12, 8))
plt.plot(range(1, 21), erros_gerados, color='red', linestyle='dashed', marker='o')
plt.title('Erro médio VS Valor de K')
plt.xlabel('K')
plt.ylabel('Erro médio em dólares.')
plt.show()




