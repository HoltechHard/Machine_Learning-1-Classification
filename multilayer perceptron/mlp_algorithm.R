# -----------------------------------------------------------------------------------
#                          MULTILAYER PERCEPTRON (MLP)
# -----------------------------------------------------------------------------------

##MAIN
source('preprocessing.R')

#                        --- processso de classifica��o ---

#fun��o de ativa��o
f <- function(net){
  return(1/(1 + exp(-net)))
}

#derivada da fun��o de ativa��o
df <- function(fnet){
  return(fnet * (1 - fnet))
}

#arquitetura - rede neural MLP
mlp.architecture <- function(input.length, hidden.length, output.length, 
                              f_activation = f, df_activation = df){
  
  model = list()
  
  model$input.length = input.length
  model$hidden.length = hidden.length
  model$output.length = output.length
  
  #gera��o aleatoria de pesos -> hidden layer
  model$Wh = matrix(runif(min = -0.5, max = 0.5, 
                    n = hidden.length * (input.length + 1)), 
                    nrow = hidden.length, ncol = input.length + 1)
  
  #gera��o aleatoria de pesos -> output layer
  model$Wo = matrix(runif(min = -0.5, max = 0.5, 
                    n = output.length * (hidden.length + 1)), 
                    nrow = output.length, ncol = hidden.length + 1)
  
  #fun��o de ativa��o
  model$f = f_activation
  
  #derivada da fun��o de ativa��o
  model$df = df_activation
  
  return(model)
}

#fase forward - algoritmo MLP
mlp.forward <- function(model, xn){
  
  ret = list()
  
  #Hidden Layer
  ret$net_h = model$Wh %*% c(xn, 1)
  ret$fnet_h = as.numeric(model$f(ret$net_h))
  
  #Output Layer
  ret$net_o = model$Wo %*% c(ret$fnet_h, 1)
  ret$fnet_o = as.numeric(model$f(ret$net_o))
   
  return(ret)
}

#fase backpropagation - algoritmo MLP
mlp.backpropagation <- function(model, x.train, y.train, eta, threshold, max_epochs){
  
  start.time = Sys.time()
  x.train = as.matrix(x.train)
  y.train = as.matrix(y.train)
  n = nrow(x.train)
  E = 2 * threshold
  counter = 0
  list_errors = NULL
  
  while(E > threshold && counter <= max_epochs){
    E = 0
    
    for(i in 1:n){
      xi = x.train[i, ]
      yi = y.train[i, ]
      
      ann = mlp.forward(model, xi)
      y.pred = ann$fnet_o
      
      #c�lculo do erro
      error = (yi - y.pred)
      E = E + sum(error^2)
      
      #Output Layer
      #dE_dWo = -error * f'(net_o) * f(net_h)
      delta_o = error * as.numeric(model$df(ann$fnet_o))
      dE_dWo = delta_o %*% t(c(ann$fnet_h, 1))
      
      #Hidden Layer
      #dE_dWh = -(f'(net_h) * SUM[error * f'(net_o) * Wo_ji]) * Xk(n)
      Wo_ji = model$Wo[, 1:model$hidden.length]
      delta_h = (as.numeric(model$df(ann$fnet_h))) * (delta_o %*% Wo_ji)
      dE_dWh = t(delta_h) %*% c(xi, 1) ##
      
      #Adapta��o dos pesos -> Output Layer
      #Wo(t+1) = Wo(t) - eta * dE_dWo
      model$Wo = model$Wo + eta * dE_dWo
      
      #Adapta��o dos pesos -> Hidden Layer
      #Wh(t+1) = Wh(t) - eta * dE_dh
      model$Wh = model$Wh + eta * dE_dWh
                  
    }
    
    #n�mero de epochs
    counter = counter + 1
    
    #Erro quadr�tico m�dio
    E = 1/2 * (E/n)
    
    if(counter %% 100 == 0){
      cat('Iter[', counter, '] ==> Error: ', E, ' ... \n')
    }
    
    #capturar os erros acumulados x epoch
    list_errors = rbind(list_errors, c(counter, E))
  }
  
  end.time = Sys.time()
  
  #resultados
  ret = list()
  ret$model = model
  ret$epochs = counter
  ret$square_error = E
  ret$list_errors = list_errors
  ret$time = end.time - start.time
  
  return(ret)
}

#fase de teste - algoritmo MLP
mlp.test <- function(x.train, y.train, x.test, y.test, 
                     input.length, hidden.length, output.length,
                     eta, threshold, max_epochs){
  
  start.time = Sys.time()
  x.test = as.matrix(x.test)
  y.test = as.matrix(y.test)
  n = nrow(x.test)
  acuracy = 0
  y.prob = NULL
  y.pred = NULL
  
  #gera��o da arquitetura da MLP
  model = mlp.architecture(input.length, hidden.length, output.length, 
                           f_activation = f, df_activation = df)
  
  #treinamento da rede MLP
  mlp.trained = mlp.backpropagation(model, x.train, y.train, 
                                    eta, threshold, max_epochs)
  #teste
  for(i in 1:n){
    xi = x.test[i, ]
    yi = y.test[i, ]
    
    fwd = mlp.forward(mlp.trained$model, xi)
    y.prob = rbind(y.prob, as.vector(fwd$fnet_o))
    y.pred = round(y.prob)
    
    if(all.equal(as.vector(y.pred[i, ]), as.vector(yi)) == TRUE){
      acuracy = acuracy + 1
    }
  }
  
  acuracy = round(acuracy/n, 5)
  end.time = Sys.time()
  
  #resultados
  ret = list()
  ret$model = mlp.trained
  ret$y.prob = y.prob
  ret$y.pred = y.pred
  ret$acuracy = acuracy
  ret$time = end.time - start.time
  
  return(ret)
}

#brechmarking: an�lise da converg�ncia -> treinamento (backpropagation)
mlp.convergence.brenchmark <- function(model){
  plot(model$list_errors[, 1], log10(model$list_errors[, 2]), type = 'l', 
       main = 'Brenchmark of MLP Convergence', xlab = '#epochs', 
       ylab = 'log10(square_error)')
}

#brechmarking: an�lise de #neur�nios hidden layer -> teste
mlp.hidden.brechmark <- function(x, y, min.hidden, max.hidden, 
                                 eta, threshold, max_epochs, k_folds){
  
  results = list()
  acuracies = NULL
  
  for(h in min.hidden:max.hidden){
    cat('\n hidden.length = ', h, '\n')
    results[[h]] = 
      mlp.cross_validation(x, y, input.length = ncol(x), hidden.length = h, 
            output.length = ncol(y), eta, threshold, max_epochs, k_folds)
    acuracies = rbind(acuracies, c(h, results[[h]]$mean.acuracy))
  }
  
  #plotar
  plot(acuracies[, 1], acuracies[, 2], type = 'l', 
       main = 'Brenchmark of Hidden Layer', xlab = '#neur�nios hidden',
       ylab = 'acur�cia')
  
  #resultados
  ret = list()
  ret$results = results
  ret$acuracies = acuracies
  
  return(ret)
}


#                           --- processo de avalia��o ---

#t�cnica holdout subsampling
mlp.holdout <- function(x, y, input.length, hidden.length, output.length, 
                        eta, threshold, max_epochs, n_iter){
  
  start.time = Sys.time()
  acuracys = NULL
  for(i in 1:n_iter){
    sets = train.test.split(x, y, train.size = 0.7)
    cat('\n Itera��o ', i, '\n')
    model.iter_i = mlp.test(x.train = sets$x.train, y.train = sets$y.train, 
                        x.test = sets$x.test, y.test = sets$y.test, 
                        input.length, hidden.length, output.length, 
                        eta, threshold, max_epochs)
    acuracys[i] = model.iter_i$acuracy
  }
  
  end.time = Sys.time()
  
  #resultados
  ret = list()
  ret$acuracys = acuracys
  ret$mean.acuracy = sum(acuracys) / length(acuracys)
  ret$time = end.time - start.time
  
  return(ret)
}

#t�cnica k-folds cross-validation
mlp.cross_validation <- function(x, y, input.length, hidden.length, output.length,
                                 eta, threshold, max_epochs, k_folds){
  
  start.time = Sys.time()
  indexs = kfolds.split(x, y, k_folds)
  
  acuracys = NULL
  for(i in 1:k_folds){
    train.indexs = as.vector(indexs[-i, ])
    test.indexs  = indexs[i, ]
    cat('\n Fold ', i, '\n')
    
    if(is.null(nrow(y)) == TRUE){
      model.fold_i = 
        mlp.test(x.train = x[train.indexs, ], y.train = y[train.indexs], 
                 x.test = x[test.indexs, ], y.test = y[test.indexs], 
                 input.length, hidden.length, output.length, 
                 eta, threshold, max_epochs)
      
    }else{
      model.fold_i = 
        mlp.test(x.train = x[train.indexs, ], y.train = y[train.indexs, ], 
                 x.test = x[test.indexs, ], y.test = y[test.indexs, ], 
                 input.length, hidden.length, output.length,
                 eta, threshold, max_epochs)
    }
    
    acuracys[i] = model.fold_i$acuracy
  }
  
  end.time = Sys.time()
  
  #resultados
  ret = list()
  ret$acuracys = acuracys
  ret$mean.acuracy = sum(acuracys) / length(acuracys)
  ret$time = end.time - start.time
  
  return(ret)
}

