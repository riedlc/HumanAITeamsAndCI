# Change workikng directory
# Run this scripts with both the egoLoopFalse and egoLoopTrue directories
setwd("YOUR DIRECTORY WITH SEPARATE GRID SEARCH CSVS")

# Load libraries
require(data.table)
require(entropy)

# Load data
files <- list.files(".", full.names = TRUE)
# Combine files
dat <- lapply(files, fread)
dat <- rbindlist(dat)
# Reround values in case issues saving floats in Python
dat$Alpha <- round(dat$Alpha, 2)  
dat$MY <- round(dat$MY, 2)
dat$Y <- round(dat$Y, 2)
dat$N <- round(dat$N, 2)
dat$MN <- round(dat$MN, 2)
dat$Count <- 1
dat$IndivID <- paste0(dat$TeamID, sep="-", dat$SubjectID)
dat$HumanAns[dat$HumanAns==-1] <- NA   
dat <- dat[!is.na(dat$HumanAns),]


## Compute Observation-Level Likelihood based on which answer the human gave
dat$Lik <- apply(dat[,c("HumanAns", "TomPost0", "TomPost1", "TomPost2", "TomPost3", "TomPost4")], 1, function(x) {
  x[2+x["HumanAns"]]
})
dat$LogLik <- log(dat$Lik)

head(dat[,c("HumanAns", "TomPost0", "TomPost1", "TomPost2", "TomPost3", "TomPost4", "Lik", "LogLik" )])
## Compute Observation-Level Likelihood based on which answer the human gave, using only the prior
dat$LikPriorOnly <- apply(dat[,c("HumanAns", "Prior0", "Prior1", "Prior2", "Prior3", "Prior4")], 1, function(x) {
  x[2+x["HumanAns"]]
})

## Compute Observation-Level Likelihood based on which answer the human gave, using only the prior
dat$PriorOnlyAns <- apply(dat[,c("Prior0", "Prior1", "Prior2", "Prior3", "Prior4")], 1, function(x) {
  nnet::which.is.max(x) - 1  # In case of tie, we give a random answer
})

dat$PriorCor      <- (dat$PriorOnlyAns == 2) + 0
dat$HumPriorMatch <- (dat$HumanAns == dat$PriorOnlyAns) + 0


## Compute KL from uniform
dat$KL <- apply(dat[,c("TomPost0", "TomPost1", "TomPost2", "TomPost3", "TomPost4")], 1, function(x) {
  KL.empirical(x, c(1/5, 1/5, 1/5, 1/5, 1/5) )
})


#################################################
## Aggregate the Individual-Level data to the "dataset level": 
## how likeliy is this dataset of 145 observations under each model parameter
## Remember log/product rule: log( prod(x) ) ==> sum( log(x) )
## Log Likelihood: sum(log(x))
## (Raw) Likelihood prod((x))
#################################################
setLevel <- dat[,.( NumObs=sum(Count),
                    HumanPerf=mean(HumCor),
                    ToMPerf=mean(TomCor),
                    Agreement=mean(HumTomMatch),
                    Agreement=mean(HumPriorMatch),
                    Lik=prod(Lik),
                    LogLik=sum(LogLik),
                    LogLikPrioOnly=sum(log(LikPriorOnly) ),
                    KL=mean(KL)
                  ), by=list(Alpha, N, MN, MY, Y)]
setLevel <- setLevel[order(setLevel$LogLik, decreasing=TRUE),]


#########################
## Write output of data-set level and obs-level for specific points of interest
#########################
write.csv(setLevel, "YOUR OUTPUT DIRECTORY/setLevel-2022-08-11-SUMNORM-egoLoopFalse.csv", row.names=FALSE)


mle <- dat[dat$Alpha==1   & dat$N==0.05 & dat$MN==0.05 & dat$MY==2.00 & dat$Y==2, ]
max <- dat[dat$Alpha==0.8 & dat$N==0.25 & dat$MN==0.90 & dat$MY==1.85 & dat$Y==2, ]
agr <- dat[dat$Alpha==0.3 & dat$N==0.10 & dat$MN==0.70 & dat$MY==1.30 & dat$Y==1.9, ]

