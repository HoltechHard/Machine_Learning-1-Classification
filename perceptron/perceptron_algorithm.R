# ----------------------------------------------------------------------------------
#                             PERCEPTRON ALGORITHM
# ----------------------------------------------------------------------------------

##MAIN
source('preprocessing.R')

#                         --- processo de classificação ---

#função de ativação => classes: {0, 1}
f <- function(net, delta = 0.5){
  if(net >= delta)
    return(1)
  else
    return(0)
}

#fase de treinamento - classificador perceptron
perceptron.train <- function(x.train, y.train, eta, threshold, max_epochs){
  
  x.train = as.matrix(x.train)
  n = nrow(x.train)
  E = 2*threshold
  counter = 0
  
  #inicializar aleatoriamente pesos
  w = runif(min = -0.5, max = 0.5, n = ncol(x.train) + 1)
  
  while(E >= threshold && counter <= max_epochs){
    E = 0
    
    for(i in 1:n){
      xi = x.train[i, ]
      yi = y.train[i]
      
      #calcular saida estimada
      net_i = c(xi, 1) %*% w
      y.pred = f(net_i)
      
      #calcular erro quadrático
      error = (yi - y.pred)
      E = E + error^2
      
      #calcular gradiente
      dE_dW = -error * c(xi, 1)
      
      #adaptacao dos pesos
      w = w - eta * dE_dW
    }
    
    #número de iterações
    counter = counter + 1
    
    #calculo do erro medio
    E = 1/2* (E/n)
    cat('Epoch[',counter, '] ==> Error: ', E, '... \n')

  }
  
  #resultados
  ret = list()
  ret$w = w
  ret$epochs = counter
  ret$squared_error = E
  
  return(ret)
}

#função de predição de classes
perceptron.predict <- function(model, xi){
  net = c(xi, 1) %*% model$w
  return(f(net))
}

#fase de teste - algoritmo perceptron
perceptron.test <- function(x.train, y.train, x.test, y.test, eta, 
                              threshold, max_epochs){
  x.test = as.matrix(x.test)
  n = nrow(x.test)
  acuracy = 0
  y.pred = NULL
  
  model = perceptron.train(x.train, y.train, eta, threshold, max_epochs)
  
  for(i in 1:n){
    xi = x.test[i, ]
    yi = y.test[i]
    
    y.pred[i] = perceptron.predict(model, xi)
    
    if(y.pred[i] == yi){
      acuracy = acuracy + 1
    }
  }
  
  acuracy = round(acuracy/n, 5)
  
  #resultados
  ret = list()
  ret$model = model
  ret$y.pred = y.pred
  ret$acuracy = acuracy
  
  return(ret)
}

#                           --- processo de avaliação --- 

#técnica holdout subsampling
percepton.holdout <- function(x, y, eta, threshold, max_epochs, n_iter){
  
  acuracys = NULL
  for(i in 1:n_iter){
    sets = train.test.split(x, y, train.size = 0.7)
    model.iter_i = perceptron.test(x.train = sets$x.train, y.train = sets$y.train, 
                        x.test = sets$x.test, y.test = sets$y.test, 
                        eta = 0.01, threshold = 1e-4, max_epochs = 50)
    acuracys[i] = model.iter_i$acuracy
  }
  
  #resultados
  ret = list()
  ret$acuracys = acuracys
  ret$mean.acuracy = sum(acuracys) / length(acuracys)
  
  return(ret)
}

#técnica k-folds cross-validation
perceptron.cross_validation <- function(x, y, eta, threshold, max_epochs, k_folds){
  
  indexs = kfolds.split(x, y, k_folds)
  
  acuracys = NULL
  for(i in 1:k_folds){
    train.indexs = as.vector(indexs[-i, ])
    test.indexs = indexs[i, ]
    
    model.fold_i = 
      perceptron.test(x.train = x[train.indexs, ], y.train = y[train.indexs], 
            x.test = x[test.indexs, ], y.test = y[test.indexs], eta, 
            threshold, max_epochs)
    acuracys[i] = model.fold_i$acuracy
  }
  
  #resultados
  ret = list()
  ret$acuracys = acuracys
  ret$mean.acuracy = sum(acuracys) / length(acuracys)
  
  return(ret)
}

