# --------------------------------------------------------------------------------
#                     K - NEAREST NEIGHBORS (K-NN) ALGORITHM
# --------------------------------------------------------------------------------

##MAIN

#                         --- processo de classificação ---


#fase de treinamento - classificador knn
knn.train <- function(x.train, y.train, query, k){
  
  #função de distância
  euclidean = apply(x, 1, function(row){
    sqrt(sum((row - query)^2))
  })
  
  #escolher os k-vizinhos mais próximos
  ids = sort.list(euclidean, dec = F)[1:k]
  
  #obter as classes 
  classes = y[ids]
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
  ret$prob = round(votos / sum(votos), 5)
  
  return(ret)
}

#fase de teste - classificador knn
knn.test <- function(x.train, y.train, x.test, y.test, k){
  
  n = nrow(x.test)
  acuracy = 0
  y.pred = NULL
  
  for(i in 1:n){
    xi = x.test[i, ]
    yi = y.test[i]
    
    model = knn.train(x.train, y.train, query = xi, k)
    y.pred[i] = model$predict
    
    if(y.pred[i] == yi){
      acuracy = acuracy + 1
    }
  }
  
  acuracy = round(acuracy/n, 5)
  
  #resultados
  ret = list()
  ret$y.pred = y.pred
  ret$acuracy = acuracy
  
  return(ret)
}


#                         --- processo de avaliação --- 

#técnica holdout subsampling
knn.holdout <- function(x, y, k, n_iter){
  
  acuracys = NULL
  for(i in 1:n_iter){
    sets = train.test.split(x, y, train.size = 0.7)
    model.iter_i = knn.test(x.train = sets$x.train, y.train = sets$y.train, 
                            x.test = sets$x.test, y.test = sets$y.test, k)
    acuracys[i] = model.iter_i$acuracy
  }
  
  #resultados
  ret = list()
  ret$acuracys = acuracys
  ret$mean.acuracy = sum(acuracys)/ length(acuracys)
  
  return(ret)
}

#técnica k-folds cross-validation
knn.cross_validation <- function(x, y, k, k_folds){
  indexs = kfolds.split(x, y, k_folds)
  
  acuracys = NULL
  for(i in 1:k_folds){
    train.indexs = as.vector(indexs[-i, ])
    test.indexs = indexs[i, ]
    
    model.fold_i = knn.test(x.train = x[train.indexs, ], y.train = y[train.indexs],
                      x.test = x[test.indexs, ], y.test = y[test.indexs], k)
    acuracys[i] = model.fold_i$acuracy
  }
  
  #resultados
  ret = list()
  ret$acuracys = acuracys
  ret$mean.acuracy = sum(acuracys) / length(acuracys)
  
  return(ret)
}
