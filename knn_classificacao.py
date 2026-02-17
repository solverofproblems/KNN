from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

base_dados = load_wine() 

X = base_dados.data
Y = base_dados.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

erros = []
for i in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)
    pred_i = knn.predict(X_test)
    taxa_erro = np.mean(pred_i != Y_test)
    erros.append(taxa_erro)

plt.figure(figsize=(12,8))
plt.plot(range(1,21), erros, color='blue', linestyle='dashed', marker='o')
plt.title('Taxa de Erros vs Valor de K')
plt.xlabel('K')
plt.ylabel('Erro:')
plt.show()



