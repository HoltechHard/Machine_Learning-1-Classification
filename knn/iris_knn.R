# ----------------------------------------------------------------------------------
#                   APLICACAO DO K-NN AO DATASET IRIS
# ----------------------------------------------------------------------------------

source('data_preprocessing.R')
source('main_knn.R')

#importar dados
dataset = read.csv('dataset/iris.data.txt', header = F)

#preprocessamento
#dataset[, -5] = std.scaler(dataset[, -5], tipo = 'standarization')

#definicao de variáveis
x = dataset[, -5]
y = as.numeric(dataset[, 5])

#dividir conjunto em folds
indexs = kfolds.split(x, k_folds = 10)

#execucao de knn.train
model.train = knn.train(x.train = x, y.train = y, 
                        query = c(5.786,  2.65, 4.95, 1.699), k = 3)

#execucao de knn.test + k-folds cross-validation
results = knn.run(x, y, k = 3, k_folds = 10)

