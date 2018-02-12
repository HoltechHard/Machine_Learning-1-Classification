# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 19:38:40 2018

@author: USER_
"""

# ---------------------------------------------------
#                  SVM USANDO SKLEARN
# ---------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#               --- Treinamento do Modelo SVM ---

#importar dataset
dataset = pd.read_csv('dataset/iris.data.txt', header = None)
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
y = np.where(y == 'Iris-setosa', 0, np.where(y == 'Iris-versicolor', 1, 2))

#dividir dados de treinamento - dados de teste
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size = 0.3, random_state = 0)

#preprocessamento - normalizacao de dados
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#fase de treinamento 
from sklearn.svm import SVC as SVM
svm = SVM(kernel = 'linear', C = 1.0, random_state = 0)
svm.fit(x_train, y_train)

#predicao das classes
y_pred = svm.predict(x_test)

#criacao da matrix de confusao
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#validacao do modelo: calculo da acuracia
from sklearn.metrics import accuracy_score
print('Acuracia: ', round(100*accuracy_score(y_test, y_pred), 3), '%')


#               --- Visualizacao da superficie de decisao ---

from matplotlib.colors import ListedColormap

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
plot_decision_regions(x = x_combinado, y = y_combinado, classifier = svm,
                      test_idx = range(int(0.7*len(y)), len(y)))
plt.xlabel('petal-length [std]')                      
plt.ylabel('petal-width [std]')
plt.title('SVM - Problema Multiclasse')
plt.legend(loc = 'upper left')
plt.show()


#           --- Kernel SVM - Problemas nao lineares ---

#PROBLEMA 01 : CLASSIFICACAO DO XOR
#dataset do XOR
np.random.seed(0)
x_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(x_xor[:, 0]>0, x_xor[:, 1]>0)
y_xor = np.where(y_xor, -1, 1)

#gráfica exploratoria do comportamento do xor
import matplotlib.pyplot as plt
plt.scatter(x_xor[y_xor==1, 0], x_xor[y_xor==1, 1], c = 'b', marker = 'x', label = '1')
plt.scatter(x_xor[y_xor==-1, 0], x_xor[y_xor==-1, 1], c = 'r', marker = 'o', label = '-1')
plt.ylim(-3.0, 3.0)
plt.title('Função XOR')
plt.legend(loc = 'upper left')
plt.show()

#               --- treinamento Kernel SVM ---

kernel_svm = SVM(kernel = 'rbf', gamma = 0.1, C = 10.0, random_state = 0)
kernel_svm.fit(x_xor, y_xor)
plot_decision_regions(x_xor, y_xor, classifier = kernel_svm)
plt.title('Classificacao do XOR')
plt.legend(loc = 'upper left')
plt.show()

#PROBLEMA 02 : CLASSIFICACAO DO IRIS

#treinamento
k_svm = SVM(kernel = 'rbf', gamma = 0.2, C = 1.0, random_state = 0)
k_svm.fit(x_train, y_train)

#predicao
y_k_pred = k_svm.predict(x_test)

#matriz de confusao
k_cm = confusion_matrix(y_k_pred, y_test)

#avaliacao do modelo: calculo da acuracia
print('Acuracia: ', 100 * round(accuracy_score(y_test, y_k_pred), 3), '%')

plot_decision_regions(x_combinado, y_combinado, classifier = k_svm, 
                      test_idx = range(int(0.7*len(y)), len(y)))
plt.xlabel('petal-length [std]')
plt.ylabel('petal-width [std]')
plt.title('Kernel SVM - Iris')
plt.legend(loc = 'upper left')
plt.show()

                