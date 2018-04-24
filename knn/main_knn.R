# --------------------------------------------------------------------------------
#                        MAIN K-NEAREST NEIGHBORS (KNN)
# --------------------------------------------------------------------------------

#                       --- fase de preprocessamento ---

source('data_preprocessing.R')

#importar dados
dataset = read.csv('iris.dat.txt', header = F)
dataset[, -5] = std.scaler(dataset[, -5], tipo = 'standarization')

#                       --- fase de classificacao ---

#classificador knn - fase de treinamento
knn.train <- function(x.train, y.train, query, k){
  
  #funcao de distancia
  euclidean = apply(x.train, 1, function(row){
    sqrt(sum((row - query)^2))
  })
  
  #selecionar os k-vizinhos mais próximos
  ids = sort.list(euclidean)[1:k]
  
  #obter as classes 
  classes = y.train[ids]
  lbl_classes = unique(classes)
  
  #sistema de votacao
  votos = rep(0, length(lbl_classes))
  for(i in 1:length(lbl_classes)){
    ids = which(classes == lbl_classes[i])
    votos[i] = length(ids)
  }
  
  #selecionar a classe mais votada
  id = which.max(votos)
  
  #resultados
  ret = list()
  ret$predict = lbl_classes[id]
  ret$classes = lbl_classes
  ret$votos = votos
  ret$probab = votos/sum(votos)
  
  return(ret)
}

#classificador knn - fase de teste
knn.test <- function(x.train, y.train, x.test, y.test, k){
  
  #fase de teste
  y.predict = NULL
  n = nrow(x.test)
  acc = 0
  
  #entrega de cada amostra do conjunto de teste como ponto de consulta
  for(i in 1:n){
    model = knn.train(x.train, y.train, query = x.test[i], k = k)    
    y.predict[i] = model$predict
    
    if(y.predict[i] == y.test)
      accuracy = accuracy + 1
  }
  accuracy = accuracy / n
  
  #resultados
  ret = list()
  ret$model = model
  ret$y.predict = y.predict
  ret$accuracy = accuracy
  
  return(ret)    
}

#                          --- fase de avaliacao ---

#divisao k-folds 
kfolds.split <- funcion(x, k_folds){
  n = nrow(x)
  tam_fold = floor(n / n_folds)
  indexs = matrix(sample(1:n, size = n), nrow = k_folds, ncol = tam_fold)
  
  return(indexs)
}

#classificador knn - fase de validacao [k-folds cross-validation]
knn.run <- function(x, y, k = 3, k_folds = 10){
  
  #divisao de k-folds 
  indexs = kfolds.split(x, k_folds)
  
  #k-folds cross-validation
  accuracy = NULL
  for(i in 1:n_folds){
    train.indexs = as.vector(indexs[-i, ])
    test.indexs = indexs[i, ]
    knn = knn.test(x.train = x[train.indexs, ], y.train = y[train.indexs],
                    x.test = x[test.indexs, ], y.test = y[test.indexs])
    accuracy[i] = knn$accuracy
  }
  
  #resultados 
  ret = list()
  ret$knn = knn
  ret$mean.accuracy = sum(accuracy) / k_folds
  
  return(ret)  
}