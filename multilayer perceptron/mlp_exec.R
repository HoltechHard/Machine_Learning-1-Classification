# -----------------------------------------------------------------------------------
#                          MULTILAYER PERCEPTRON (MLP)
# -----------------------------------------------------------------------------------

#EXEC 
source('mlp_algorithm.R')

#dataset [XOR.DAT]

#importar dataset
dataset = read.table('dataset/xor.dat', header = F)

#definir variáveis
x = dataset[, -3]
y = dataset[, 3]

#exec -> mlp.architecture
model = mlp.architecture(input.length = 2, hidden.length = 2, output.length = 1)

#resultados da arquitetura: 
#model
#$input.length
#[1] 2

#$hidden.length
#[1] 2

#$output.length
#[1] 1

#$Wh
#         [,1]      [,2]        [,3]
#[1,] -0.2557777 0.4869673 0.422445292
#[2,] -0.3073524 0.3531330 0.009170389

#$Wo
#       [,1]       [,2]       [,3]
#[1,] 0.289159 0.08321123 -0.3951658


#exec -> mlp.forward
forward = mlp.forward(model, c(0, 0))

#resultados da etapa forward:
#forward
#$net_h
#         [,1]
#[1,] 0.422445292
#[2,] 0.009170389

#$fnet_h
#[1] 0.6040682 0.5022926

#$net_o
#         [,1]
#[1,] -0.1786977

#$fnet_o
#[1] 0.4554441

#exec -> mlp.backpropagation
results = mlp.backpropagation(model, x, y, eta = 0.1, 
                          threshold = 1e-3, max_epochs = 20000)

#resultados da etapa backpropagation:
#$model
#$model$input.length
#[1] 2

#$model$hidden.length
#[1] 2

#$model$output.length
#[1] 1

#$model$Wh
#         [,1]     [,2]      [,3]
#[1,] -5.199934 5.435381  2.585933
#[2,] -5.998211 5.959766 -3.292964

#$model$Wo
#       [,1]     [,2]     [,3]
#[1,] -8.1042 8.325355 3.821512

#$epochs
#[1] 26845

#$square_error
#[1] 0.0009999402

#$time
#Time difference of 21.82999 secs


#resultados finais - predição de valores

xor.predict <- function(x, y){
  
  v = NULL
  v.pred = NULL
  x = as.matrix(x)
  
  #modelo como pesos aleatórios
  model = mlp.architecture(input.length = 2, hidden.length = 2, output.length = 1, 
                           f_activation = f, df_activation = df)
  
  #treinamento - backpropagation
  results = mlp.backpropagation(model, x.train = x, y.train = y, 
                                eta = 0.1, threshold = 1e-3, max_epochs = 20000)
  
  #predição de valores
  for(i in 1:nrow(x)){
    fwd = mlp.forward(results$model, x[i, ])
    v[i] = fwd$fnet_o
    v.pred[i] = y[i]
  }
  
  table_v = matrix(v, nrow = 2, ncol = 2)
  cat('Valores estimados: \n')
  print(table_v)
  
  table_v.pred = matrix(v.pred, nrow = 2, ncol = 2)
  cat('Valores reais: \n')
  print(table_v.pred)
  
}

#Valores estimados: 
#         [,1]       [,2]
#[1,] 0.03177341 0.96335787
#[2,] 0.97081566 0.02819268

#Valores reais: 
#       [,1] [,2]
#[1,]    0    1
#[2,]    1    0

# --------------------------------------------------------------------------------#

#dataset [IRIS.DAT]

#importar dataset
dataset = read.csv('dataset/iris.data.txt', header = F)
x = dataset[, -5]
y = nnet::class.ind(dataset[, 5])

#exec -> MLP.architecture
model = mlp.architecture(input.length = 4, hidden.length = 4, output.length = 3)

#resultados da arquitetura do modelo:
#model
#$input.length => 4 variáveis na matrix de caraterísticas
#[1] 4

#$hidden.length => 4 hiperplanos separadores 
#[1] 4

#$output.length => 3 classes na saída
#[1] 3

#$Wh
#         [,1]        [,2]         [,3]        [,4]       [,5]
#[1,] 0.4909757  0.15083776  0.006543172 -0.30890971 -0.4933608
#[2,] 0.3086363 -0.04713665 -0.218674030  0.04847281  0.1433256
#[3,] 0.4574309  0.03528334 -0.087460517  0.08134589  0.4659145
#[4,] 0.1128736 -0.43525006 -0.042963191 -0.36518570  0.2849574

#$Wo
#         [,1]       [,2]       [,3]        [,4]       [,5]
#[1,] 0.4605910 -0.3328256 0.04935303  0.07604849  0.3191940
#[2,] 0.3195499  0.2921290 0.22990181 -0.45401033  0.4631732
#[3,] 0.1379429  0.3914864 0.21266740 -0.43299337 -0.2957995

#exec -> mlp.forward
fwd = mlp.forward(model, c(5.9, 3.0, 5.1, 1.8))

#resultados da etapa forward:
#fwd
#$net_h
#         [,1]
#[1,]  2.3332418
#[2,]  0.7948832
#[3,]  2.9709810
#[4,] -1.2312852

#$fnet_h
#[1] 0.9115929 0.6888789 0.9512458 0.2259566

#$net_o
#         [,1]
#[1,] 0.5739194
#[2,] 1.0718206
#[3,] 0.2040962

#$fnet_o
#[1] 0.6396671 0.7449430 0.5508477

#exec -> mlp.backpropagation
results = mlp.backpropagation(model, x.train = x, y.train = y, eta = 0.05, 
                        threshold = 1e-3, max_epochs = 20000)

#resultados da etapa backpropagation:
#results
#$model
#$model$input.length
#[1] 4

#$model$hidden.length
#[1] 4

#$model$output.length
#[1] 3

#$model$Wh
#         [,1]       [,2]       [,3]      [,4]          [,5]
#[1,]  0.9659395   2.211809  -3.619487 -1.965821  -0.103195973
#[2,]  1.7266562 -15.377595   7.380054  8.825596 -13.416765999
#[3,] -0.6220966  -2.234044   3.163246  1.562906  -0.003299461
#[4,] 11.7165132  -3.175508 -12.669022 -9.164343  10.815774213

#$model$Wo
#         [,1]      [,2]      [,3]        [,4]      [,5]
#[1,]   6.805598 -1.729564 -5.576990   0.9398032 -1.471217
#[2,] -13.749754 -9.786874  8.218175  10.5854407 -2.579808
#[3,]  -1.663684 10.227613 -2.422385 -11.2346531 -3.556321

#$epochs
#[1] 50001

#$square_error
#[1] 0.01320827

#$time
#Time difference of 23.39477 mins

#predição de resultados

#dataset[1, ]
ypred_1 = mlp.forward(results$model, c(5.1, 3.5, 1.4, 0.2))$fnet_o
#> ypred_1
#[1] 9.980884e-01     3.278584e-03        7.118807e-08

#yreal_1
yreal_1 = y[1, ]
#    Iris-setosa     Iris-versicolor     Iris-virginica 
#         1               0                   0

#dataset[51, ]
ypred_51 = mlp.forward(results$model, c(7.0, 3.2, 4.7, 1.4))$fnet_o
#> ypred_51
#[1] 2.190769e-03     9.999999e-01        4.544607e-08

#yreal_51
yreal_51 = y[51, ]
#   Iris-setosa       Iris-versicolor       Iris-virginica 
#       0                   1                     0 

#dataset[101, ]
ypred_101 = mlp.forward(results$model, c(6.3, 3.3, 6.0, 2.5))$fnet_o
#> ypred_101
#[1] 0.0001541076       0.0155413928        0.9859211790

#yreal_101
yreal_101 = y[101, ]
#> yreal_101
#     Iris-setosa       Iris-versicolor     Iris-virginica 
#         0                   0                   1 

id = which(ypred_101 == 1)

if(all.equal(as.vector(round(ypred_101)), yreal_101) == TRUE){
  cat('ok')
}else{
  cat('not')
}

#exec -> mlp.test
sets = train.test.split(x, y, train.size = 0.7)
x.train = sets$x.train
y.train = sets$y.train
x.test = sets$x.test
y.test = sets$y.test

iris.test = mlp.test(x.train = sets$x.train, y.train = sets$y.train, 
                     x.test = sets$x.test, y.test = sets$y.test, 
                     input.length = 4, hidden.length = 4, output.length = 3, 
                     eta = 0.1, threshold = 1e-3, max_epochs = 20000)

#resultados de 1 teste
#iris.test
#$model
#$model$model
#$model$model$input.length
#[1] 4

#$model$model$hidden.length
#[1] 4

#$model$model$output.length
#[1] 3

#$model$model$Wh
#[,1]       [,2]      [,3]      [,4]        [,5]
#[1,]  0.4328089  1.9616583 -2.563383 -1.230033 -0.19123584
#[2,] -0.6102117 -2.2993718  3.152946  1.760774  0.06779742
#[3,] -6.2625266  0.4269269  5.411317 10.298223 -4.37269447
#[4,] -1.1869425  8.6746790 -3.419428 -4.004969  3.65524837

#$model$model$Wo
#[,1]       [,2]      [,3]      [,4]       [,5]
#[1,]  2.834547 -6.6267405 -1.864285  1.697583  0.2903609
#[2,] -5.421246  6.6095319 -7.450977  4.354528 -2.6714326
#[3,] -1.183401 -0.4572726  7.348633 -6.367547 -2.7519085

#$model$epochs
#[1] 2110

#$model$square_error
#[1] 0.009998706

#$model$time
#Time difference of 36.47479 secs

# 
# $acuracy
# [1] 0.95556     predição: [43 acertos | 2 erros]
# 
# $time
# Time difference of 36.4908 secs
# 

#exec -> mlp.convergence.brenchmark [#epochs ideal = 2000 aprox]
mlp.convergence.brenchmark(iris.test$model)

#exec -> mlp.hidden.brenchmark [#hidden.length ideal = 6]
iris.hidden.brenchmark = mlp.hidden.brechmark(x, y, min.hidden = 2, max.hidden = 10, 
                              eta = 0.1, threshold = 1e-3, max_epochs = 2000, 
                              k_folds = 10)

#iris.hidden.brenchmark$acuracies
#      #hidden   acuracy
#[1,]    2      0.959999
#[2,]    3      0.953332
#[3,]    4      0.946666
#[4,]    5      0.959998
#[5,]    6      0.959999
#[6,]    7      0.959999
#[7,]    8      0.966666
#[8,]    9      0.960000
#[9,]   10      0.973332

#exec -> mlp.holdout
iris.holdout = 
  mlp.holdout(x, y, input.length = 4, hidden.length = 4, output.length = 3, 
          eta = 0.1, threshold = 1e-2, max_epochs = 10000, n_iter = 10)

#resultados da avaliação usando holdout - random subsampling
#iris.holdout
#$acuracys
#[1] 0.95556 0.97778 0.97778 0.95556 0.97778 0.95556 1.00000 0.95556 0.91111 0.91111

#$mean.acuracy
#[1] 0.95778

#$time
#Time difference of 42.77738 mins

#exec -> mlp.cross_validation
iris.cross_validation = 
  mlp.cross_validation(x, y, input.length = 4, hidden.length = 4, output.length = 3, 
                  eta = 0.1, threshold = 1e-2, max_epochs = 10000, k_folds = 10)

#resultados da avaliação usando k-folds cross-validation
#iris.cross_validation
#$acuracys
#[1] 1.00000 1.00000 1.00000 1.00000 1.00000 0.93333 0.93333 1.00000 1.00000 1.00000

#$mean.acuracy
#[1] 0.986666

#$time
#Time difference of 33.24109 mins

# ----------------------------------------------------------------------------------#

# dataset [WINE.DAT]

#importar dataset
dataset = read.csv('dataset/wine.data.txt', header = F)

#definir variáveis
x = dataset[, -1]
y = nnet::class.ind(dataset[, 1])

#preprocessamento - standardization
x = std.scaler(dataset = x, tipo = 'standardization')

#dividir conjunto de treinamento - conjunto de test
sets = train.test.split(x, y, train.size = 0.7)

#exec -> mlp.test
results = mlp.test(x.train = sets$x.train, y.train = sets$y.train, 
                   x.test = sets$x.test, y.test = sets$y.test, 
                   input.length = 13, hidden.length = 10, output.length = 3,
                   eta = 0.1, threshold = 1e-3, max_epochs = 10000)

#exec -> mlp.convergence.brenchmark [#epochs ideal = 200]
mlp.convergence.brenchmark(model = results$model)

#exec -> mlp.hidden.brenchmark [#hidden ideal = 10]
wine.hidden.brenchmark = mlp.hidden.brechmark(x, y, min.hidden = 2, max.hidden = 15, 
                     eta = 0.1, threshold = 1e-3, max_epochs = 500, k_folds = 10)

#> wine.hidden.brenchmark$acuracies
#       [,1]       [,2]
#[1,]    2      0.976472
#[2,]    3      0.970589
#[3,]    4      0.964706
#[4,]    5      0.982354
#[5,]    6      0.970590
#[6,]    7      0.964707
#[7,]    8      0.970589
#[8,]    9      0.976472
#[9,]   10      0.976472
#[10,]   11     0.988236
#[11,]   12     0.982354
#[12,]   13     0.970589
#[13,]   14     0.970590
#[14,]   15     0.970589

#exec -> mlp.holdout
wine.holdout = 
  mlp.holdout(x, y, input.length = 13, hidden.length = 2, output.length = 3, 
              eta = 0.1, threshold = 1e-3, max_epochs = 500, n_iter = 10)

# wine.holdout
#$acuracys
#[1] 0.96296 0.96296 0.96296 0.98148 0.98148 0.96296 0.94444 0.96296 0.96296 0.98148

#$mean.acuracy
#[1] 0.966664

#exec -> mlp.cross_validation
wine.cross_validation = 
  mlp.cross_validation(x[1:170, ], y[1:170, ], 
                       input.length = 13, hidden.length = 2, output.length = 3,
                       eta = 0.1, threshold = 1e-3, max_epochs = 500, k_folds = 10)

#wine.cross_validation
#$acuracys
#[1] 0.94118 1.00000 1.00000 1.00000 0.94118 1.00000 1.00000 0.94118 1.00000 1.00000

#$mean.acuracy
#[1] 0.982354
