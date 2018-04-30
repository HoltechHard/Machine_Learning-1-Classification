# ---------------------------------------------------------------------------------
#                             PERCEPTRON ALGORITHM
# ---------------------------------------------------------------------------------

##EXEC 
source('perceptron_algorithm.R')

#dataset [AND.DAT]

#importar dataset
dataset = read.table('dataset/and.dat', header = F)

#definir variáveis
x = dataset[, -3]
y = as.numeric(dataset[, 3])

#exec -> perceptron.train
model = perceptron.train(x, y, eta = 0.1, threshold = 1e-4)

# resultados do modelo:
# model
# $w
#   V1         V2            
# 0.3511165  0.3535560 -0.1406662 

# $counter
# [1] 5

# $squared_error
# [1] 0

#exec -> perceptron.predict
table = NULL
table = cbind(perceptron.predict(model, c(0, 0)), 
              perceptron.predict(model, c(0, 1)))
table = rbind(table, cbind(perceptron.predict(model, c(1, 0)),  
                           perceptron.predict(model, c(1, 1))))

cat('Tabela AND \n')
print(table)

#       [,1] [,2]
#[1,]    0    0
#[2,]    0    1

# --------------------------------------------------------------------------------#

#dataset [iris.dat]

#importar dataset
dataset = read.csv('dataset/iris.data.txt', header = F)

#definir variáveis
x = dataset[1:100, c('V1', 'V3')]
y = as.numeric(dataset[1:100, 5])

#análise exploratório das classes
plot(x, col = y)

#transformar as classes
ids = which(y == 2)
y[ids] = 0
  
#treinamento do modelo
model = perceptron.train(x, y, eta = 0.01, threshold = 1e-4, max_epochs = 100)

# resultados do modelo:
# model
# $w
#     V1         V3            
# 0.1981208 -0.1672054 -0.1174353 

# $epochs
# [1] 14

# $squared_error
# [1] 0


# predicao de valores

# se x = (5.1, 1.4) ==> y = 1
perceptron.predict(model, c(5.1, 1.4))
# se x = (7, 4.7)  ==> y = 0
perceptron.predict(model, c(7, 4.7))

#exec -> perceptron.test
source('preprocessing.R')
sets = train.test.split(x, y, train.size = 0.7)
x.train = sets$x.train
y.train = sets$y.train
x.test = sets$x.test
y.test = sets$y.test
results = perceptron.test(x.train, y.train, x.test, y.test, eta = 0.01, 
                            threshold = 1e-4, max_epochs = 100)

#resultados de 1 teste particular:
# results
# $model
# $model$w
#     V1          V3             
# 0.06338895 -0.14162592  0.48003408 

# $model$epochs
# [1] 2

# $model$squared_error
# [1] 0

# $y.pred
# [1] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0

# $acuracy
# [1] 1

#exec -> perceptron.holdout
holdout = percepton.holdout(x, y, eta = 0.01, threshold = 1e-4, 
                              max_epochs = 50, n_iter = 10)

#resultados de acurácia promedio: 
# holdout
# $acuracys
# [1] 1.0000 1.0000 1.0000 1.0000 0.9667 1.0000 1.0000 1.0000 1.0000 1.0000

# $mean.acuracy
# [1] 0.996667

#exec -> perceptron.cross_validation
cross.validation = perceptron.cross_validation(x, y, eta = 0.01, threshold = 1e-4, 
                              max_epochs = 50, k_folds = 10)

#resultados de acurácia promedio:
# cross.validation
# $acuracys
# [1] 1.0 0.9 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0

# $mean.acuracy
# [1] 0.99

