#Vamos definir todas as bibliotecas necessárias para rodar o sistema. 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

#Aqui estamos carregando a base dedados necessária.
base_dados = load_wine() 

#Aqui estamos defindo as variáveis que serão usadas para previsão
X = base_dados.data

#Aqui nós estamos a coluna (característica) que o modelo deve prever. Também é chamado de "Target (Alvo)".
Y = base_dados.target

#Aqui estamos definindo as parcelas de treino e de teste da base de dados. Definimos que 20% de todo o dataset será destinado para testes. Além disso, definimos uma randomização padrão de 42, evitando altas discrepâncias conforme o código é rodado múltiplas vezes.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)

#Aqui estamos normalizando a escala do modelo.
scaler = StandardScaler()

#Aqui o modelo aprende a "média" e o desvio dos dados de treino e já os escala.
X_train = scaler.fit_transform(X_train)
#Usa a mesma escala aprendida no treino para ajustar os dados de teste.
X_test = scaler.transform(X_test)

#Aqui estamos armazenando os erros do modelo em uma lista chamada "Erros". Isso é importante, pois, podemos validar qual é o limiar padrão de K-vizinhos necessários para desenvolver um algoritmo que consegue prever valores sem overfitting ou underfitting.
erros = []
for i in range(1, 21):
    #Veja que o número de K-vizinhos varia de acordo com a estrutura de repetição "For". isso é essencial para reconhecer qual é o intervalo de valores para K que satisfaz nossa análise.
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)
    pred_i = knn.predict(X_test)
    #Usamos a biblioteca Numpy para armazenar o valor em decimal. Entretanto, transformamos em percentual.
    taxa_erro = np.mean(pred_i != Y_test)
    erros.append(taxa_erro * 100)

#Por final, estamos construindo gráfico que seja observado. Esse gráfico mostra o percentual de erro de previsão existente no modelo conforme o valor de K-Vizinhos aumenta. Observamos que no intervalo [12, 18[, nós temos uma estabilidade e o menor percentual de erro possível.
plt.figure(figsize=(12,8))
plt.plot(range(1,21), erros, color='red', linestyle='dashed', marker='o')
plt.title('Taxa de Erros vs Valor de K')
plt.xlabel('K-Vizinhos (Valor)')
plt.ylabel('Erro (em percentual %):')
plt.show()



