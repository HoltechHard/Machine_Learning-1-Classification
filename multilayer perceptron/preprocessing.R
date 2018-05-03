# ---------------------------------------------------------------------------------
#                               DATA PREPROCESSING
# ---------------------------------------------------------------------------------

##MAIN

#           --- padronização de variáveis a uma mesma escala ---

#classe std.scaler
std.scaler <- function(dataset, tipo){
  m = ncol(dataset)
  
  for(j in 1:m){
    if(tipo == 'normalization'){
      dataset[, j] = std.scaler.normalization(dataset[, j])
    }
    else if(tipo == 'standardization'){
      dataset[, j] = std.scaler.standardization(dataset[, j])
    }
  }
  return(dataset)
}

#técnica: normalização  |  xnorm(i) = (xi - xmin) / (xmax - xmin)
std.scaler.normalization <- function(x){
  n = length(x)
  xmin = min(x)
  xmax = max(x)
  
  xnorm = NULL
  for(i in 1:n){
    xnorm[i] = (x[i] - xmin) / (xmax - xmin)
  }
  
  return(xnorm)
}

#técnica: standarização  |  xstd(i) = (xi - ux) / sigma(x)
# sigma(x) =  sqrt(1/(n-1) * sum(xi - ux)^2)
sigma <- function(x){
  n =length(x)
  ux = sum(x)/n
  soma = sum((x - ux)^2)
  sigma = sqrt(1/(n-1) * soma)
  
  return(sigma)
}

std.scaler.standardization <- function(x){
  n = length(x)
  ux = sum(x)/n
  
  xstd = NULL
  for(i in 1:n){
    xstd[i] = (x[i] - ux) / sigma(x)
  }
  
  return(xstd)
}

#                        --- técnicas de amostragem ---

#técnica: holdout  | divisao - conjunto de treinamento / conjunto teste
train.test.split <- function(x, y, train.size = 0.7){
  n = nrow(x)
  ids = sample(1:n, size = floor(train.size * n))
  
  #conjunto de treinamento
  x.train = x[ids, ]
  
  if(is.null(nrow(y)) == TRUE){
    #problemas de classificação binária
    y.train = y[ids]
  }else{
    #problemas multi-classe
    y.train = y[ids, ]  
  }
  
  #conjunto de teste
  x.test = x[-ids, ]
  
  if(is.null(nrow(y)) == TRUE){
    #problemas de clasificação binária
    y.test = y[-ids]
  }else{
    #problemas multi-classe
    y.test = y[-ids, ]
  }
  
  #resultados
  ret = list()
  ret$x.train = x.train
  ret$y.train = y.train
  ret$x.test = x.test
  ret$y.test = y.test
  
  return(ret)
}

#técnica: k-folds   |  divisão subconjuntos [folds] mutuamente independentes
kfolds.split <- function(x, y, k_folds){
  n = nrow(x)
  tam_folds = floor(n/k_folds)
  
  indexs = matrix(sample(1:n, size = n), nrow = k_folds, ncol = tam_folds)
  
  return(indexs)
}

