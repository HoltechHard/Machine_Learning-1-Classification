# ---------------------------------------------------
#              MULTILAYER PERCEPTRON
# ---------------------------------------------------

#FASE FORWARD: entradas ==> resultado

#funcao de ativacao
f <- function(net){
  return(1/ (1 + exp(-net)))
}

#derivada da funcao de ativacao
df_dnet <- function(fnet){
  return(fnet * (1-fnet))
}

#arquitetura do modelo MLP
mlp.architecture <- function(input.length = 2, hidden.length = 2, output.length = 1,
                    f.activation = f, df.activation = df_dnet){
  model = list()
  model$input.length = input.length
  model$hidden.length = hidden.length
  model$output.length = output.length
  
  #matriz de pesos - hidden layer : Wh
  model$hidden = 
    matrix(runif(min = -0.5, max = 0.5, n = hidden.length * (input.length + 1)), 
           nrow = hidden.length, ncol = input.length + 1)
  
  #matriz de pesos - output layer: Wo
  model$output = 
    matrix(runif(min = -0.5, max = 0.5, n = output.length * (hidden.length + 1)), 
           nrow = output.length, ncol = hidden.length + 1)
  
  model$f = f.activation
  model$df = df.activation
  return(model)
}

#etapa forward
mlp.forward <- function(model, xn){
  ret = list()
  
  #hidden layer: 
  #------------#
  # ui(n) = SUM{k = 1 -> a} [Wik(n) * Xk(n)] + thetai
  ret$u_n = model$hidden %*% c(xn, 1)
  # h(ui(n))
  ret$h_u_n = as.numeric(model$f(ret$u_n))
  
  #output layer
  #------------#
  # vj(n) = SUM{i= 1 -> b}[Wji(n) * h(ui(n))] + thetaj
  ret$v_n = model$output %*% c(ret$h_u_n, 1)
  # f(vj(n))
  ret$f_v_n = as.numeric(model$f(ret$v_n))
  
  return(ret)
}

#FASE BACKWARD: treinamento ==> teste

mlp.backpropagation <- function(model, dataset, eta = 0.1, threshold = 1e-3){
  En = 2 * threshold
  epochs = 0
  
  while(En > threshold){
    En = 0
    
    for(n in 1:nrow(dataset)){
      #vetor de carateristicas X(n)
      xn = as.numeric(dataset[n, 1:model$input.length])
      
      #vetor de classes Y(n)
      yn = as.numeric(dataset[n, (model$input.length+1):ncol(dataset)])
      
      ann = mlp.forward(model, xn)
      
      #saída estimada yj'(n)
      y_pred = ann$f_v_n
      
      #erro:  ej(n) = yj'(n) - yj(n)
      erro = yn - y_pred
      
      #erro quadratico: E(n) = SUM[ej(n)^ 2]
      En = En + sum(erro^2)
      
      #Treinamento da output layer
      # dE_dWo    = - ej(n) * f'(vj(n)) * h(ui(n))
      # delta_Wo  = - eta * dE_dWo
      # Wo(t+1)   = Wo(t) + delta_Wo
      
      #derivada do erro parcial com relacao aos pesos output layer
      de_dWo = as.numeric(erro * model$df(ann$f_v_n))
      
      #Treinamento da hidden layer
      # dE_dWh    = - h'(ui(n)) * (SUM{j=1->c}[ej(n) * f'(vj(n)) * Wo(n)]) * Xk(n)
      # delta_Wh  = - eta * dE_dWh
      # Wh(t+1)   = Wh(t) + delta_Wh
      
      #pesos output layer sem theta
      Wo_ji = model$output[, 1:model$hidden.length]
      
      #derivada do erro parcial com relacao aos pesos hidden layer
      de_dWh = as.numeric(model$df(ann$h_u_n)) * (de_dWo %*% Wo_ji)
      
      #derivada do erro total com relacao a pesos output layer
      dE_dWo = de_dWo %*% as.vector(c(ann$h_u_n, 1))
      
      #derivada do erro total com relacao a pesos hidden layer
      dE_dWh = t(de_dWh) %*% as.vector(c(xn, 1))
      
      #atualizacao pesos output layer
      model$output = model$output + eta * dE_dWo
      
      #atualizacao pesos hidden layer
      model$hidden = model$hidden + eta * dE_dWh
      
    }
    
    # Erro quadrático médio: E(n) = 1/n * SUM[ej(n)^2]
    En = En / nrow(dataset)
    epochs = epochs + 1
    
    if(epochs %% 100 == 0){
      cat('Epoch: ', epochs, ' =>  E(n) = ', En, '\n')
    }
  }
  
  ret = list()
  ret$model = model
  ret$epochs = epochs
  
  return(ret)
}

#-execucao da mlp
dataset = read.table('dataset/xor.dat')
model = mlp.architecture(input.length = 2, hidden.length = 2, output.length = 1)
train = mlp.backpropagation(model = model, dataset = dataset, eta = 0.1)

#testes - problema XOR

# epochs: 28'355

# Wh: pesos hidden layer
# $model$hidden
#         [,1]      [,2]      [,3]
# [1,] 5.600280 -5.721637 -3.070280
# [2,] 5.538668 -5.342318  2.682634

#Wo: pesos output layer
# $model$output
#         [,1]      [,2]     [,3]
# [1,] 8.416483 -8.114552 3.808297

#teste - tabela XOR
m = matrix(c(mlp.forward(train$model, c(0, 0))$f_v_n, 
       mlp.forward(train$model, c(0, 1))$f_v_n, 
       mlp.forward(train$model, c(1, 0))$f_v_n,
       mlp.forward(train$model, c(1, 1))$f_v_n),
       nrow = 2, ncol = 2)

#tabela XOR - teste               |       tabela XOR - esperado
#         [0]         [1]         |           [0]         [1]
#[0]  0.03187294  0.97044282      |    [0]    0.0         1.0
#[1]  0.96369876  0.02813379      |    [1]    1.0         0.0


