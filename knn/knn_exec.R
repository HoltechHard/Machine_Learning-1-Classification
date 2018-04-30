# ---------------------------------------------------------------------------------
#                      K - NEAREST NEIGHBORS (K-NN) ALGORITHM
# ---------------------------------------------------------------------------------

##EXEC [IRIS]

source('preprocessing.R')
source('knn_algorithm.R')

#importar dataset
dataset = read.csv('dataset/iris.data.txt', header = F)
x = dataset[, -5]
y = as.numeric(dataset[, 5])

#exec -> knn.train
model = knn.train(x.train = x, y.train = y, 
              query = c(5.786,  2.65, 4.95, 1.699), k = 3)

#resultados do modelo:
# model
#$predict
#[1] 3

#$classes
#[1] 3 2

#$votos
#[1] 2 1

#$prob
#[1] 0.66667 0.33333

#exec -> knn.test
sets = train.test.split(x, y, train.size = 0.7)
x.train = sets$x.train
y.train = sets$y.train
x.test = sets$x.test
y.test = sets$y.test
results = knn.test(x.train, y.train, x.test, y.test,  k = 3)

#resultados de 1 teste particular:
# $y.pred
# [1]  1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 
# [23] 2 2 2 2 2 2 3 3 3 3 3 3 3 2 3 3 3 3 3 3
# [43] 3 3 3

# $acuracy
# [1] 0.97778

#exec -> knn.holdout
holdout = knn.holdout(x, y, k = 3, n_iter = 10)

#resultados de acurácia promedio:
# $acuracys
# [1] 0.97778 0.95556 0.93333 0.97778 0.95556 0.95556 0.95556 0.95556 0.97778 1.00000

# $mean.acuracy
# [1] 0.964447

#exec -> knn.cross_validation
cross.validation = knn.cross_validation(x, y, k = 3, k_folds = 10)

#resultados de acurácia promedio:
#$acuracys
#[1] 1.00000 0.93333 1.00000 0.86667 0.93333 0.93333 1.00000 1.00000 0.93333 1.00000

#$mean.acuracy
#[1] 0.959999

