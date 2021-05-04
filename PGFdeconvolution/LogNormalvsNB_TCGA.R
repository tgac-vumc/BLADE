#' # Deconvolution using PGFs: Log-Normal vs Negative Binomial (TCGA)
#' 
#' Script: Mark van de Wiel, mark.vdwiel@amsterdamumc.nl
#' 
#' Goal: compare the fit of a log-normal convolution with that of a negative 
#' binomial convolution, assuming known cell-type fractions for TCGA data. 
#' The fractions of 8 cell types have been estimated using epic.
#' 
#' Step 1: use epic with signature genes to estimate fractions f. 
#' Step 2: estimate the parameters of the convolution components given f 
#' using maximum likelihood, where the convolution is efficiently approximated 
#' by a discrete probability generating function. 
#' Step 3: Compare likelihoods of the Log-normal and Negative Binomial 
#' convolutions (same nr of parameters) for many genes
#' 
#' ## Estimating fractions f by EPIC using immunedeconv package
library(immunedeconv)

exprdata <- immunedeconv::dataset_racle$expr_mat
dim(exprdata)
res = deconvolute(exprdata, "epic")
res$cell_type

#' From epic package; Reference profiles obtained from single cell data of 
#' tumor infiltrating cells. 7 classes (+1 class for unknown, 
#' supposedly malignant). Counts are represented as Total per million counts

datprior <- TRef$refProfiles
rnprior <- rownames(datprior)
dim(datprior)

#' loads gene expression data
load("TCGA-MESO.RData")

datgene <- t(X[[2]])
rndat <- rownames(datgene)
dim(datgene)

#' Mapping: where do the prior genes occur in the data set?
matchNames <- match(rnprior,rndat) 
whnotNA <- which(!is.na(matchNames))
length(whnotNA)/nrow(datprior) #~75% matches

#' rescaling to CPM
datgene_match <- datgene[matchNames[whnotNA],]
datprior_match <- datprior[whnotNA,]
cs <- colSums(datgene_match)
cpm <- t(t(datgene_match)/(cs/10^6))  #normalizing to cpm
csprior <- colSums(datprior_match)
cpm_prior <- t(t(datprior_match)/(csprior/10^6)) #renormalizing to cpm; required because of matching


#' Estimating fractions with epic 
dc <- deconvolute_epic(datgene,tumor=TRUE,scale_mrna=FALSE,scaleExprs=TRUE) #indeed render very similar results

#' Results for first 5 samples
dc[,1:5]

#' ## Deconvolution: estimating components using PGFs

library(pracma) #for convolution
library(Rsolnp) # for optimization with lagrange mult (eq constraint)
library(MASS) #for estimating initial values

#' Source function for max lik estimation for convolutions
source('source_estmusigmadist2.R')

#' Auxiliary function for max lik estimation NB parameters (needed
#' for initialization)
initnb2 <- function(datc,fracs){
  #datc <- cpmc[gene10[10],];fracs <- t(dc)
  datag <- data.frame(y=datc,fracs)
  fitnb <- try(glm.nb(y ~ 1,data=datag))
  nf <- ncol(fracs)
  if(!class(fitnb)[1]=="try-error"){
    sizeest <- max(0.1,1/fitnb$theta)
    cf <- sapply(fitnb$coef,function(x) max(-5,min(5,x)))
    parin <- c(rep(cf,nf),rep(sizeest,nf))
  } else {
    fit <- glm(y ~ 1,family="poisson",data=datag)
    sizeest <- 0.1
    cf <- sapply(fit$coef,function(x) max(-5,min(5,x)))
    parin <- c(rep(cf,nf),rep(sizeest,nf))
  }
  return(parin)
}

#' Auxiliary function for max lik estimation log-normal parameters (needed
#' for initialization)
initloggauss2 <- function(datc,fracs){
  #datc <- cpmc[gene10[10],];fracs <- t(dc)
  mn <- mean(log(datc+0.01))
  #vars <- var(log(datc+0.01)) + 10^{-7}
  nf <- ncol(fracs)
  parin <- c(rep(mn,nf),rep(1,nf))
  return(parin)
}



#' Filter out genes with row mean > = 5
rnmean <- rowMeans(datgene_match)
whin <- which(rnmean>=5) 
length(whin)/nrow(datgene_match) #82%

#' Sample 200 genes with row means exceeding 5
set.seed(4664352)
gene200 <- sort(sample(whin,200)) 
cpmc <- round(cpm) #count scale
cpmc200 <- cpmc[gene200,]

#' Starting values for log-normal 
smloggauss <- t(apply(cpmc200,1,initloggauss2,fracs=t(dc)))

#' Starting values for negative binomial
smnb <- suppressWarnings(t(apply(cpmc200,1,initnb2,fracs=t(dc))))

#' Estimating the 8 log-normal components. Takes considerable time (e.g. approx. 
#' 1hr for 200 genes, m=84). allefs contains the 8 fractions for all samples 
#' (8 x n matrix). ngrid defines the granularity of the discrete pgf approximation
#' Here, we assume the same precision parameter for all 8 cell types. 
#' distr defines the distribution of the components, here log-normal (ln)

pmt <- proc.time()
rms <- estmusigPGF(dat=cpmc200,js=1:200, allefs=dc, ngrid=25, optimizer="rsolnp", startmat=smloggauss, 
                    cont = list(outer.iter=3,tol=10^(-5)), distr="ln", modus="sameprec", prior=FALSE)
ct <- proc.time()-pmt

#' Estimating the 8 negative binomial components. distr defines the distribution
#' of the components, here negative binomial (nb)

pmt <- proc.time()
rmsnb <- estmusigPGF(dat=cpmc200,js=1:200,allefs=dc, ngrid=25, startmat=smnb, cont = list(outer.iter=3,tol=10^(-5)), distr="nb",
                      modus="sameprec",optimizer="rsolnp",prior=FALSE)
ctnb <- proc.time() -pmt

#' Results are stored in an .Rdata file
#load("res_tcga_meso_200genes.Rdata")

#' Compute likelihoods
liks <- function(est) return(sapply(1:length(est),function(i) -est[[i]]$likopt))
liksg <- liks(rms)
liksnb <- liks(rmsnb)

#' Plot likelihoods
plot(liksnb, liksg,ylim = c(-700,-100),xlim=c(-700,-100),ylab = "max lik LN", xlab = "max lik NB")
abline(a=0,b=1)

#' Save results
save(cpmc,gene200,smloggauss,smnb,rms,rmsnb,ct,ctnb,liksg,liksnb,file="res_tcga_meso_200genes.Rdata")


