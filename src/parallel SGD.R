rm(list=ls())
library(parallel)
# library(R.matlab)

## Change Path Appropriately

# data <- readMat('/media/rishabh/X/IISc M.tech/2nd-Sem/PDS/Assignment/3/regression_dataset.mat')
input <- read.csv("/media/rishabh/X/IISc M.tech/2nd-Sem/ML/Project/Code/trainBlog.csv", sep=",")
# input <- read.table("/media/rishabh/X/IISc M.tech/2nd-Sem/PDS/Assignment/3/data1.txt", sep=",")

data1 <- data.matrix(input)

d <- ncol(data1)

X = data1[,1:d-1]
Y = data1[,d]

# X = data$train
# Y = data$train.y

nr=nrow(X)
nc=ncol(X)

#s.Normalization
  for(i in 1:nc)
  {
    maxX = max(X[,i])
    minX = min(X[,i])
    X[,i] = (X[,i] - minX)/(maxX - minX)
  }
  # maxY=max(Y)
  # minY=min(Y)
  # Y=(Y-minY)/(maxY-minY)
#e.Normalization

  diff=floor(nc/3)
  w=vector("numeric",length=nc)
  
  w = w+(1/nc)
  star<-Sys.time()
  func=function(k)
  { 
    if(k+2*diff > nc)
      pr=perm[k:nc]
    else
      pr=perm[k:(k+diff-1)]
    
    feature=X[,pr]
    theta=w[pr]
    
    for ( i in 1:nr)
    {
      theta=theta-(0.1/nr)*((theta%*%feature[i,])-Y[i] )*feature[i,]
    }
    return(theta)
  }
#s.Single Core Execution
  #diff=10;
  #handle the last features
  # for (i in 1:100)
  # {
  #   perm = 1:nc#sample(nc)
  # 
  #   initials=seq(1,nc,by=diff)
  #   initials=initials[1:length(initials)-1]
  #   #perm=sample(nc)
  #   
  #   res1 = func(initials[1])
  #   res2 = func(initials[2])
  #   res = c(res1,res2)
  #   w[perm]=res
  # }
  # sum( (Y - t(w%*%t(X)))^2 )/nr
#e.Sngle Core Execution

#s.Multicore Execution
  no_cores <- detectCores() - 1
  cl <- makeCluster(no_cores)
  clusterExport(cl, var=c("X","nc","diff","nr","Y"))
  
  initials=seq(1,nc,by=diff)
  initials=initials[1:length(initials)-1]   #handle the last features
  
  for (i in 1:100)
  {
    perm=sample(nc)
    clusterExport(cl, var=c("perm","w"))
    res=parLapply(cl,initials,func)
  
    w[perm]=unlist(res)
  }
  mean( (Y - t(w%*%t(X)))^2 )     #Mean squared error
  
  stopCluster(cl)
#e.Multicore Execution
  end<-Sys.time()
  print(end - star)
