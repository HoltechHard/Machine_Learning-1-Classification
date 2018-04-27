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

#exec -> knn.test
sets = train.test.split(x, y, train.size = 0.7)
x.train = sets$x.train
y.train = sets$y.train
x.test = sets$x.test
y.test = sets$y.test
results = knn.test(x.train, y.train, x.test, y.test,  k = 3)

#exec -> knn.holdout
holdout = knn.holdout(x, y, k = 3, n_iter = 10)

#exec -> knn.cross_validation
cross.validation <- knn.cross_validation(x, y, k = 3, k_folds = 10)

