# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 19:48:15 2018

@author: USER_
"""

# -----------------------------------------------
#           NAIVE BAYES USANDO SKLEARN             
# -----------------------------------------------

#importar dataset
import pandas as pd
import numpy as np
dataset = pd.read_csv('dataset/iris.data.txt', header = None)
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
y = np.where(y == 'Iris-setosa', 0, np.where(y == 'Iris-versicolor', 1, 2))

#dividir conjunto de treinamento e conjunto de teste
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size = 0.3, random_state = 0)

#preprocessamento: normalizacao de dados
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#fase de treinamento
from sklearn.naive_bayes import GaussianNB                
naive = GaussianNB()
naive.fit(x_train, y_train)

#predicao das classes
y_pred = naive.predict(x_test)
y_prob = naive.predict_proba(x_test)

#matriz de confusao 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#avaliacao do modelo: calculo da acuracia
from sklearn.metrics import accuracy_score
print('Acuracia: ',100* round(accuracy_score(y_test, y_pred), 3), '%')

#plotar fronteira de decisao
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(x, y, classifier, test_idx = None, resolution = 0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'yellow', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    #plotar superficie de decisao
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)                      
    plt.contourf(xx1, xx2, z, alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    #plotar todas as samples
    x_test = x[test_idx, :]
    for idx, classe in enumerate(np.unique(y)):
        plt.scatter(x = x[y == classe, 0], y = x[y == classe, 1], 
                    alpha = 0.8, c = cmap(idx), marker = markers[idx], label = classe)
    #deixar resaltado os test samples
    if test_idx:
        x_test = x[test_idx, :]
        plt.scatter(x_test[:, 0], x_test[:, 1], c = '', alpha = 1.0, 
                    linewidth = 1, marker = 'o', s = 55, label = 'test set')

#EXE -
x_combinado = np.vstack((x_train, x_test))
y_combinado = np.hstack((y_train, y_test))
plot_decision_regions(x = x_combinado, y = y_combinado, classifier = naive,
                      test_idx = range(int(0.7*len(y)), len(y)))
plt.xlabel('petal-length [std]')
plt.ylabel('petal-width [std]')
plt.title('Naive Bayes')
plt.legend(loc = 'upper left')
plt.show()

