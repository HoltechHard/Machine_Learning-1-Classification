# --------------------------------------------------------------------------------
#                              DATA PREPROCESSING
# --------------------------------------------------------------------------------

#               --- padronizacao de variáveis a uma mesma escala ---

#classe std.scaler
std.scaler <- function(dataset, tipo){
  m = ncol(dataset)
  for(j in 1:m){
    if(tipo == 'normalization'){
      dataset[, j] = std.scaler.normalization(dataset[, j])
    }
    else if(tipo == 'standarization'){
      dataset[, j] = std.scaler.standarization(dataset[, j])
    }
  }
  
  return(dataset)
}

#normalizacao: xnorm(i) = (xi - xmin) / (xmax - xmin)
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

#standarizacao: xstd(i) = (xi - ux) / sigma(x)
# sigma(x) = sqrt(1/(n-1) * sum(xi - ux)^2)

sigma <- function(x){
  n = length(x)
  ux = mean(x)
  soma = sum((x - ux)^2)
  sigma = sqrt(1/(n-1) * soma)
  
  return(sigma)
}

std.scaler.standarization <- function(x){
  n = length(x)
  ux = mean(x)
  sigma = sigma(x)
  
  xstd = NULL
  for(i in 1:n){
    xstd[i] = (x[i] - ux) / sigma
  }
  
  return(xstd)
}

#teste para standarizacao
test.iris <- function(){
  dataset = read.csv('iris.dat.txt', header = F)
  dataset[, -5] = std.scaler(dataset[, -5], tipo = 'standarization')
}

#         --- dividir conjunto de treinamento / conjunto teste ---

train.test.split <- function(x, y, train.size = 0.7){
  n = nrow(x)
  ids = sample(1:n, size = floor(train.size * n))
  
  #conjunto de treinamento
  x.train = x[ids, ]
  y.train = y[ids]
  
  #conjunto e teste
  x.test = x[-ids, ]
  y.test = y[-ids]
  
  #resultados
  ret = list()
  ret$x.train = x.train
  ret$y.train = y.train
  ret$x.test = x.test
  ret$y.test = y.test

  return(ret)  
}

#teste para a divisao
test.split.iris <- function(){
  
  #definicao de variáveis
  m = ncol(dataset)
  dataset = read.csv('iris.dat.txt', header = F)  
  x = dataset[, 1:(m-1)]
  y = dataset[, m]
  
  #divisao do conjunto de treinamento/ conjunto de teste
  sets = train.test.split(x, y, train.size = 0.7)
  x.train = sets$x.train
  y.train = sets$y.train
  x.test  = sets$x.test
  y.test  = sets$y.test
  
}


