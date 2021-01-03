
#' Estimation based on probability generating functions
estmusigPGF <- function(dat, js=1:nrow(dat), allefs, whnew=NULL, ngrid=25, optimizer="rsolnp", startmat=NULL, prior=TRUE, 
                        priormu = c(1,(1/6)^2),priorprec = c(0.5,1/6), cont=NULL,distr ="ln",modus="sameprec"){
  #modus <- "scaleprec"; scaling of precs with one C
  #modus <- "difprec"; dif precs for two different data sources.
  # dat=cbind(datandpars$datbulk,datandpars$datsingle);js=1:5; whnew<-c(rep(TRUE,ncol(datandpars$datbulk)),rep(FALSE,ncol(datandpars$datsingle)));
  # cont <- NULL; allefs=rbind(c(datandpars$nunoise,rep(0,ncol(datandpars$datsingle))),cbind(datandpars$fsbulk,datandpars$fssingle));ngrid=25;distr<-"nb"; optimizer<-"rsolnp";
  # startmat=NULL; prior<-F;priormu = c(1,(1/6)^2);priorprec = c(0.5,1/6)
  # truepars <- cbind(datandpars$muTs,datandpars$precTs);optimizer<-"rsolnp";prior<-F;priormu = c(1,(1/6)^2);priorprec = c(0.5,1/6)
  # modus<- "sameprec";startmat <- cbind(munoise=rep(0,nrow(dat)),initpars[,1:3],precnoise=rep(1,nrow(dat)),initpars[,-(1:3)],c=rep(1,nrow(dat)))
  # startmat <- cbind(munoise=rep(0,nrow(dat)),initpars[,1:3],precnoise=rep(1,nrow(dat)),initpars[,-(1:3)])
  
  #dat = cbind(Tum2,Nor2);js =81:100; startmat =cbind(muTsintemp,muTsin,precTsin,precTsin);distr="nb";allefs = fbulksingle;distr="nb";prior=F;ngrid<-25;optimizer="rsolnp"
 
  
  if(is.null(cont)) cont <- list(outer.iter=10)
  nT <- nrow(allefs)
  nsam <- ncol(dat)
  if(is.null(whnew)){print("Assuming all samples are from the same source, so share parameters"); modus <- "sameprec";whnew<-rep(FALSE,nsam)}
  
  #if(distr =="trnormal") {pdist <- function(x,f,mu,disp){trun0 <- pnorm(0,f*mu, sd=f*disp);return((pnorm(x,f*mu, sd=f*disp)-trun0)/(1-trun0))}; lb <- 0.01; ub <- 10^10}
  if(distr =="normal") {pdist <- function(x,f,mu,disp){return(pnorm(x,f*mu, sd=f*disp))}; lb <- 0.01; ub <- 10^10}
  if(distr =="ln") {pdist <- function(x,f,mu,disp){return(plnorm(x,log(f) + mu, sd=disp))}; lb <- 0.05; ub <- 20}
  if(distr =="nb") {pdist <- function(x,f,mu,disp){return(pnbinom(x/f,mu=exp(mu),size=1/disp))}; lb <- 0.0001; ub <- 100} #because mu is searched on the log-scale

  if(modus=="sameprec") nprecT <- nT
  if(modus=="scaleprec") nprecT <- nT+1
  if(modus=="difprec") nprecT <- 2*(nT-1)+1
  
  params2muprec <- function(params,modus){
   if(modus=="sameprec") return( c(params,params[-(1:(nT+1))]))
   if(modus=="scaleprec") return(c(params[1:nT],params[nT+1], params[(nT+2):(2*nT)], params[(2*nT +1)] * params[(nT+2):(2*nT)]))
   if(modus=="difprec") return(params)
  } 
  
  logpenalty <- function(muprecs, priormu,priorprec,nT,distr){
    mut <- muprecs[1:nT]
    prect <- muprecs[-(1:nT)]
    if(distr=="ln"){
    lp <- sum(dnorm(mut,priormu[1],sd=sqrt(1/priormu[2]),log=T)) +
      sum(dgamma(prect,priorprec[1],rate=priorprec[2],log=T)) 
    } else {
      lp <- sum(dnorm(mut,priormu[1],sd=sqrt(1/priormu[2]),log=T)) +
      sum(dlnorm(prect,mean=0,sd=1,log=T)) 
    }
    return(lp)
  }
  
  resj <- function(j){
    #j<-1
    #initialization for mus and precs when startmat is NULL
    param <- c(rep(2, nT),rep(1,nprecT))
    datj <- dat[j,]
    #j<-1
    whj <- which(js==j)
    print(whj)
    Ydens.loop <- function(i, allefs, muprecs, distr){
      #j=1;efs <- rep(1/nT, nT) ;i<-6;muprecs <- param
      testi <- whnew[i] 
      muTa <- muprecs[1:nT];
      if(!testi) precs <- muprecs[(nT+1):(2*nT)] else precs <- c(muprecs[nT+1],muprecs[(2*nT+1):(3*nT-1)])
      if(distr=="nb") sdTa <- precs else sdTa <- 1/sqrt(precs)
      yi <- as.numeric(datj[i])
      #print(yi)
      
      efsi <- allefs[,i]
      
      whnz <- which(efsi >= 0.001) #
      #lf <- log(efsi[whnz])
      ef <- efsi[whnz]
      muT <- muTa[whnz]
      sdT <- sdTa[whnz]
      nTnz <- length(whnz)
      
      if(yi==0) {a <- 0.5; step <- 1; ngridi <- 0} else {
        if(distr=="nb") { ngridi <- min(ngrid,yi)} else ngridi <- ngrid
        step <- yi/ngridi
        a<- rev(c(0,1:ngridi)+0.5)*step
      }
      
      #t=1
      t<-1
      #convt <- plnorm(a,lf[t] + muT[t],sdT[t]) - plnorm(a-step,muT[t]+lf[t],sdT[t]); 
      convt <- pdist(a,ef[t],muT[t],sdT[t]) - pdist(a-step,ef[t],muT[t],sdT[t]) 
      if(nTnz >1){
        for(t in 2:nTnz){
          #t<-2
          at <- pdist(a,ef[t],muT[t],sdT[t]) - pdist(a-step,ef[t],muT[t],sdT[t]) ; 
          convo <- conv(convt,at) #sorted from high to low powers
          convt <- rev(rev(convo)[1:(ngridi+1)]) #sorted from high to low powers; only powers yi ... 0 are needed
        }
      }
      intpgf<-convt[1]/step
      return(max(10^(-100),intpgf)) 
    }
    

    loglik.loop <- function(params){
      #muprecs <- initpar
      muprecs <- params2muprec(params,modus)
      
      if(!prior) return(-sum(log(sapply(1:nsam, Ydens.loop, muprecs=muprecs, allefs=allefs,distr=distr)))) else {
      minusloglik <- -sum(log(sapply(1:nsam, Ydens.loop, muprecs=muprecs, allefs=allefs,distr=distr)))
      logpenal <- -logpenalty(muprecs,priormu=priormu,priorprec=priorprec,nT=nT,distr=distr)
      return(minusloglik+logpenal) 
      }
    }
    # loglik.loop(param)
    # loglik.loop(truepars[j,])
    #pmt <- proc.time()
    
    # if(is.null(startmat)) {initpar <- param} else 
    #   {
    #   if(is.na(startmat[j,1])) {initpar <- param} else 
    #     {
    #     initpar <- startmat[j,]
    #     }
    #   }
    
    initpar <- param
    if(!is.null(startmat)) 
    {
    whnotNA <- which(!is.na(startmat[j,]))
    if(length(whnotNA)>0) initpar[whnotNA] <- startmat[j,whnotNA] 
    }
    
    print("Starting values:")
    print(initpar) 
    loglik.loop(initpar)
      
    if(optimizer=="rsolnp") optres <- try(solnp(par = initpar, loglik.loop,  
                                                LB = c(rep(-10,nT),rep(lb,nprecT)), UB=c(rep(20,nT),rep(ub,nprecT)),
                                                control=cont )) else {
                                                  optres <- try(optim(par = initpar, loglik.loop, lower =  c(rep(-10,nT),rep(lb,nprecT)), 
                                                                      upper=c(rep(20,nT),rep(ub,nprecT)), method = "L-BFGS-B"))  
                                                  print(optres$value)
                                                  print(optres$par)
                                                }
    if(distr=="normal"){datj <- exp(datj)-1;distrib = "ln"} else distrib <- distr
    loglik <- function(params){
      muprecs <- params2muprec(params,modus)
      return(-sum(log(sapply(1:nsam, Ydens.loop, muprecs=muprecs, allefs=allefs,distr=distrib)))) 
    }
    
    loglikopt <- loglik(optres$pars)
    print(loglikopt)
    optres <- c(optres,likopt=list(loglikopt))
    return(optres)
  } # end function i
  #is <- 1:2
  resall <- lapply(js,resj)
  return(resall)
}



#estimates from single cell data
estmusigsingle <- function(datsingle, fssingle, limma=T, distr="ln"){
  #dat<- datandpars$datsingle; fssingle= datandpars$fssingle
  nT <- nrow(fssingle)
  nsingle <- ncol(fssingle)
  
  if(distr =="ln"){
  allmus <- allprecs <- c()
  
  for(t in 1:nT){
    #t<-1
    whin <- which(fssingle[t,]==1)
   dat <- log(datsingle[,whin])
   mus <- apply(dat,1,mean)
   allmus <- rbind(allmus,mus)
   if(limma){
     design <- cbind(Intercept=rep(1,nsingle/nT))
     fit <- lmFit(dat,design)
     fit <- eBayes(fit)
     varsshrink <- fit$s2.post
     precs <- 1/varsshrink
   } else precs <- 1/apply(dat,1,var)
   allprecs <- rbind(allprecs,precs)
  }
  allpars <- rbind(allmus,allprecs)
  rownames(allpars) <- c(paste("mu",1:nT,sep=""),paste("prec",1:nT,sep=""))
  } #end ln
  if(distr =="nb"){
   datsingle <- round(datsingle,0) #data should be integers
    
   allmus <- alldisps <- c()
   for(t in 1:nT){
     #t<-1
     whin <- which(fssingle[t,]==1)
     dat <- datsingle[,whin]
     mus <- apply(dat,1,mean)
     allmus <- rbind(allmus,mus)
     d <- DGEList(counts = dat) #using edgeR
     d.CR <- estimateCommonDisp(d)
     if(nrow(dat) >= 100) d.CR <- estimateTrendedDisp(d.CR)
     d.CR <- estimateTagwiseDisp(d.CR)
     disps <- d.CR$tagwise.dispersion
     #var2 <- mus + mus^2*disps
     alldisps <- rbind(alldisps,disps)
   }
   allpars <- rbind(sapply(log(allmus),max,y=-5),alldisps)  #log to make it easier for optimization purposes
   rownames(allpars) <- c(paste("logmu",1:nT,sep=""),paste("disp",1:nT,sep=""))
  } #end nb
  return(t(allpars))
}

#init estimate for (mu,sigma) of the remainder component
initmusigremain <- function(datbulk,js,musingle,precsingle,initf=NULL,ngrid=25, optimizer="rsolnp", prior=TRUE,
                            priormu = c(1,(1/6)^2),priorprec = c(0.5,1/6),startmat=NULL,cleverinit=TRUE, distr="ln"){
  #distr<- "ln"; cleverinit<-T; datbulk <- datandpars$datbulk; musingle <- muTs; precsingle<- precTs; initf=rbind(datandpars$nunoise, datandpars$fsbulk);ngrid=25;optimizer="rsolnp"; prior=FALSE; priormu = c(1,(1/6)^2);priorprec = c(0.5,1/6)
  #datbulk=dat;js=1:nrow(dat);initf=allef;musingle=muTs;precsingle=precTs;ngrid=25; optimizer="rsolnp";prior=prior;priormu=priormu;priorprec=priorprec
  #datbulk <- Tum2; js <- 1:2;musingle <- muTsin; precsingle<- precTsin; initf <- NULL;ngrid=25;optimizer="rsolnp";prior=FALSE; priormu = c(1,(1/6)^2);priorprec = c(0.5,1/6)
  
  #FIRST f SHOULD BE REMAINDER COMPONENT!!! 
  nsam <- ncol(datbulk)
  
  #assumes priormeans are known for nT cell types
  nT <- ncol(musingle)+1
  
  #initialize fs; uniform if no info
  if(is.null(initf)){
    allefs <- matrix(rep(rep(1/nT,nT),nsam),ncol=nsam)
  } else allefs <- initf
  
  if(distr =="ln") {pdist <- function(x,f,mu,disp){return(plnorm(x,log(f) + mu, sd=disp))}; lb <- 0.05; ub <- 20}
  if(distr =="nb") {pdist <- function(x,f,mu,disp){return(pnbinom(x/f,mu=exp(mu),size=1/disp))}; lb <- 0.0001; ub <- 100} #because mu is searched on the log-scale

  logpenalty <- function(j, muprecT, priormu,priorprec){
    mut <- muprecT[1]
    prect <- muprecT[2]
    lp <- dnorm(mut,priormu[1],sd=sqrt(1/priormu[2]),log=T) +
      dgamma(prect,priorprec[1],rate=priorprec[2],log=T) 
    return(lp)
  }
  
  resj <- function(j){
    #j<-1
    musj <- musingle[j,]
    precsj <- precsingle[j,]
    datj <- datbulk[j,]
    
    
    allconvo <- list()
    for(i in 1:nsam){
      #i<-1
      yi <- as.numeric(datj[i])
      efsi <- allefs[-1,i]
      
      whnz <- which(efsi >= 0.001) #
      
      ef <- efsi[whnz]
      muT <- musj[whnz]
      if(distri=="nb") sdT <- precsj[whnz] else  sdT <- 1/sqrt(precsj[whnz])
      nTnz <- length(whnz)
      
      
      if(yi==0) {a <- 0.5; step <- 1; ngridi <- 0} else {
        if(distri=="nb") { ngridi <- min(ngrid,yi)} else ngridi <- ngrid
        step <- yi/ngridi
        a<- rev(c(0,1:ngridi)+0.5)*step
      }
      
      #t=1
      t<-1
      convt <- pdist(a,ef[t],muT[t],sdT[t]) - pdist(a-step,ef[t],muT[t],sdT[t]) 
      if(nTnz >1){
        for(t in 2:nTnz){
          #t<-4
          at <- pdist(a,ef[t],muT[t],sdT[t]) - pdist(a-step,ef[t],muT[t],sdT[t]) ; 
          convo <- conv(convt,at) #sorted from high to low powers
          convti <- rev(rev(convo)[1:(ngridi+1)]) #sorted from high to low powers; only powers yi ... 0 are needed
        }} else {convti <- convt}
      allconvo <- c(allconvo,list(convti))
    }
    
    
    Ydens.loop <- function(i, muprecT){
      #i=1;j<-1;muprecT <- c(munaive,precnaive);ngrid<-200
      #i=1;j<-1;muprecT <- c(2.7,0.9);ngrid<-25
      muTe <- muprecT[1]
      if(distri=="nb") sdTe <- muprecT[2] else sdTe <- sqrt(1/muprecT[2])
      yi <- as.numeric(datj[i])
      if(yi==0) {a <- 0.5; step <- 1; ngridi <- 0} else {
        if(distri=="nb") { ngridi <- min(ngrid,yi)} else ngridi <- ngrid
        step <- yi/ngridi
        a<- rev(c(0,1:ngridi)+0.5)*step
      }
      efsi <- allefs[1,i]+10^{-20}
      
      #t=1
      convt <- allconvo[[i]]
      at <- pdist(a, efsi, muTe,sdTe) - pdist(a-step,efsi,muTe,sdTe); 
      convo <- conv(convt,at) #sorted from high to low powers
      convt <- rev(rev(convo)[1:(ngridi+1)]) #sorted from high to low powers; only powers yi ... 0 are needed
      intpgf<-convt[1]/step
      #print(intpgf)
      return(max(10^(-100),intpgf)) 
    }
    
    loglik.loop <- function(muprecT){
      #muprecs <- param
      if(!prior) return(-sum(log(sapply(1:nsam, Ydens.loop, muprecT=muprecT)))) else {
        minusloglik <- -sum(log(sapply(1:nsam, Ydens.loop, muprecT=muprecT)))
        logpenal <- -logpenalty(j, muprecT=muprecT, priormu=priormu,priorprec=priorprec)
        return(minusloglik+logpenal) 
      }
    }
    
    
    # loglik.loop <- function(muprecT){
    #   #muprecT=c(munaive,precnaive)
    #   #muprecT=c(2.7,0.9)
    #   minusloglik <- -sum(log(sapply(1:nsam, Ydens.loop,  muprecT=muprecT)))
    #   return(minusloglik) 
    # }
    #loglik.loop(initpar)
    #loglik.loop(c(12,1/13))
    # loglik.loop(truepars[j,])
    #pmt <- proc.time()
    if(is.null(startmat)){
    if(cleverinit){
      if(distr=="ln"){
      muinit <- mean(log(as.numeric(datj)))
      precinit <- 1/(sd(log(as.numeric(datj)))^2)
      initpar <- c(muinit,precinit)
    } 
    if(distr=="nb"){
      mu0 <- mean(as.numeric(datj))
      muinit <- log(mu0)
      dispinit <- max(0.1,(var(as.numeric(datj))-mu0)/mu0^2)
      initpar <- c(muinit,dispinit)
      #initpar <- c(1,1)
    }
    } else {initpar <- c(1,1)}} 
    else {initpar <- startmat[j,]}
    cat("Initial:",round(loglik.loop(initpar),4),"Pars:",round(initpar,5))
    
    
    if(optimizer=="rsolnp") optres <- try(solnp(par = initpar, loglik.loop,  
                                                LB =  c(-1,lb), UB=c(20,ub),
                                                control=list(outer.iter=10) )) else {
                                                  optres <- try(optim(par = initpar, loglik.loop, lower =  c(-0.01,lb), upper=c(20,ub), 
                                                                      method = "L-BFGS-B"))  
                                                  # print(optres$value)
                                                  print(optres$par)
                                                }
    return(optres)
  } # end function j
  #resj(3)
  return(lapply(js,resj))
}

