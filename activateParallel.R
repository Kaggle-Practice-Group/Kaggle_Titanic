activateParallel <- function(ignorecores)
{
  if(.Platform$OS.type=="windows") {
    # Set up parallel processors for Windows.
    library(doParallel)
    cl = makeCluster(detectCores() - ignorecores)
    registerDoParallel(cl)
  }  
  else
  {
    # Set up parallel processors for Mac.
    library('doMC')
    registerDoMC(cores = detectCores() - ignorecores)
  }
}

