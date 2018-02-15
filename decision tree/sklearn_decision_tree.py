# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:46:23 2018

@author: USER_
"""

# -------------------------------------------------------------
#               ÁRVORES DE DECISÃO USANDO SKLEARN
# -------------------------------------------------------------

#Critérios de pureza
import numpy as np
import matplotlib.pyplot as plt

def gini_index(p):
    return p*(1-p) + (1-p)*(1-(1-p))

def entropy(p):
    return -(p*np.log2(p) + (1-p)*np.log2(1-p))

def erro(p):
    return 1 - np.max([p, 1-p])

x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p!=0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]
err = [erro(i) for i in x]

#plotar comportamento da impureza de informacao
ax = plt.subplot(111)
for i,lab,ls,c, in zip([ent, sc_ent, gini_index(x), err],
                        ['Entropy', 'Entropy Scaler', 'Gini Index', 'Error'],
                        ['-', '-', '--', '-.'],
                        ['black', 'lightgray', 'red', 'green', 'cyan']):
    line = ax.plot(x, i, label = lab, linestyle = ls, lw = 2, color = c)
ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, 1.1), ncol = 2, 
          fancybox = True, shadow = False)
ax.axhline(y = 0.5, linewidth = 1, color = 'k', linestyle = '--')
ax.axhline(y = 1.0, linewidth = 1, color = 'k', linestyle = '--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Grau de impureza')
plt.show()

#           --- Treinamento do Modelo Árvore de Decisão ---

#importar dataset
import pandas as pd
dataset = pd.read_csv('dataset/iris.data.txt', header = None)
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values                
y = np.where(y == 'Iris-setosa', 0, np.where(y == 'Iris-versicolor', 1, 2))

#dividir conjunto de treinamento - conjunto de teste
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size = 0.3, random_state = 0)                

#fase de treinamento
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, 
                              random_state = 0)
tree.fit(x_train, y_train)

#predicao das classes
y_pred = tree.predict(x_test)
y_prob = tree.predict_proba(x_test)

#criacao da matriz de confusao
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#avaliacao do modelo : calculo da acuracia
from sklearn.metrics import accuracy_score
print('Acuracia: ', round(100*(accuracy_score(y_test, y_pred)), 3), '%')


#               --- Visualização da superficie de decisão --- 

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

#EXE-
x_comb = np.vstack((x_train, x_test))
y_comb = np.hstack((y_train, y_test))
plot_decision_regions(x_comb, y_comb, classifier = tree, 
                      test_idx = range(int(0.7*len(y)), len(y)))
plt.xlabel('petal-length [std]')
plt.ylabel('petal-width [std]')
plt.title('Árvore de Decisão')
plt.legend(loc = 'upper left')
plt.show()

#construcao da árvore de decisao
from sklearn.tree import export_graphviz
export_graphviz(decision_tree = tree, out_file = 'tree.dot', 
                feature_names = ['petal-length', 'petal-width'])

# ----------------------------------------------
#           RANDOM FOREST COM SKLEARN
# ----------------------------------------------

from sklearn.ensemble import RandomForestClassifier

#fase de treinamento
random_forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy',
                                       n_jobs = 2, random_state = 1)
random_forest.fit(x_train, y_train)

#predicao do modelo
y_rand_pred = random_forest.predict(x_test)

#matriz de confusao 
cm_rf = confusion_matrix(y_test, y_rand_pred)

#avaliacao do modelo : cálculo da acuracia
print('Acurácia : ', 100 * round(accuracy_score(y_test, y_rand_pred), 3), '%')

#plotar superficie de decisao
plot_decision_regions(x, y, classifier = random_forest, 
                      test_idx = range(int(0.7*len(y)), len(y)))
plt.xlabel('petal-length [cm]')
plt.ylabel('petal-width [cm]')
plt.title('Random Forest - Iris')
plt.legend(loc = 'upper left')
plt.show()
