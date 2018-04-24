# ------------------------------------------
#                PERCEPTRON
# ------------------------------------------

f <- function(net, delta = 0.5){
  
  if(net >= delta)
    return(1)
  else
    return(0)

}

perceptron.fit <- function(dataset, eta = 0.1, threshold = 1e-3){
  
  #definindo variaveis 
  classId = ncol(dataset)
  x = as.matrix(dataset[, 1:(classId-1)])
  y = dataset[, classId]
  
  #inicializacao aleatoria dos pesos
  weights = runif(min = -0.5, max = 0.5, n = (ncol(x) + 1))
  
  #adaptacao da rede
  total_error = 2 * threshold
  
  while(total_error >= threshold){
    
    total_error = 0
    
    for(i in 1:nrow(dataset)){
      xi = x[i, ]
      yi = y[i]
      
      #cálculo da f_net
      net = c(xi, 1) %*% weights
      y_pred = f(net)
      
      #cálculo do erro
      error = (yi - y_pred)
      total_error = total_error + error^2
      
      #cálculo do gradiente
      dE2_dW = -2 * error * c(xi, 1)
      
      #adaptacao dos pesos
      weights = weights - eta * dE2_dW
    }
    
    total_error = total_error / nrow(dataset)
    
    cat('error: ', total_error, '... \n')
  }
  
  #resultados
  ret = list()
  ret$weights = weights
  ret$error = total_error
  
  return(ret)
}

perceptron.predict <- function(x.test, model){
  net = c(x.test, 1) %*% model$weights
  return(f(net))
}

and.toy <- function(){
  dataset = read.table('dataset/and.dat', header = F)
  model = perceptron.fit(dataset, eta = 0.1, threshold = 1e-4)
  table = NULL
  table = cbind(perceptron.predict(c(0, 0), model), 
                perceptron.predict(c(0, 1), model))
  table = rbind(table, cbind(perceptron.predict(c(1, 0), model), 
                             perceptron.predict(c(1, 1), model)))
  
  cat('Tabela AND \n')
  print(table)
  
  return(model)
}


