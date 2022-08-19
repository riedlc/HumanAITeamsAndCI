#################################################
##    Human-Agent Teams -- Hidden Profile
##
## Main analysis
#################################################

# Change workikng directory
setwd("YOUR WORKING DIRECTORY")

# Load libraries
require(data.table)
require(ggplot2)
require(texreg)

# Configuration
cols <- c('#909090', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7')
# Alignment is black (0)
# Human is orange (1)
# Agent is blue (2)



# This is the file generated at the end of LikAnalysis_discovery.R
setLevel <- read.csv("YOUR OUTPUT DIRECTORY/setLevel-2022-08-11-SUMNORM-egoLoopTrue.csv", stringsAsFactors = FALSE)
# setLevel <- read.csv("DATA/setLevel-2022-08-11-SUMNORM-egoLoopFalse.csv", stringsAsFactors = FALSE)

setLevel <- setLevel[order(setLevel$LogLik, decreasing=TRUE),]

# Max agreement
setLevel[which.max(setLevel$Agreement),]

## Maximum performance
setLevel[which.max(setLevel$ToMPerf),]

## MLE - Maximum Likelihood: What are the most likely model parameters alpha, N, MN, Y, MY?
setLevel[which.max(setLevel$LogLik),]

## Self-actualization only
head(setLevel[setLevel$Alpha==0,], 5)

# Maximum likelihood for PriorOnly
setLevel[which.max(setLevel$LogLikPrioOnly),]

# mle for partnerLoopFalse
# ONLY applies for egoLoopFalse data - no self-actualization
# head(setLevel[setLevel$Alpha==1,], 5)



sub <- dat[dat$Alpha==0.3 & dat$N==0.1 & dat$MN==0.4 & dat$MY==2 & dat$Y==2,]
dim(sub)
mean(sub$PriorCor)
mean(sub$HumPriorMatch)


###########################################
## Compare 4 models
## (a) random 
## (b) just clues (only prior)
## (c) self-loop only (alpha=0)
## (d) ToM (MLE)
##
###########################################

# Random
llRandom <- log(1/5) * 144
dfRandom <- 0

# Prior Only
llPrior <- -215.9055
dfPrior <- 3

# Self-Loop only (prior + outgoing) -- alpha=0, other paras at maxLik
llSelfOnly <- -187.5768
dfSelfOnly <- 4        

# ToM MLE -- alpha at MaxLik
llToMMLE <- -106.6397  
dfToMMLE <- 5          

# ToM Max performance
llToMMaxPerf <- -109.8264
dfToMMaxPerf <- 5        

# Comparisons
qSelfVsToM <- -2 * (llSelfOnly - llToM)
pchisq(qSelfVsToM, dfToM-dfSelfOnly, lower.tail = FALSE)

# ToM MLE vs. ToM MaxPerf
q <- -2 * (llToMMaxPerf - llToMMLE)
pchisq(q, 1, lower.tail = FALSE)

# Random vs. Prior
q <- -2 * (llRandom - llPrior)
pchisq(q, dfPrior-dfRandom, lower.tail = FALSE)





#################################################
#################################################
## ToM Analysis
## 
## Is ToM predictive of individual and team performance?
#################################################
#################################################
netStats <- read.csv("..data/network_measures.csv", stringsAsFactors = FALSE); netStats$X <- NULL
netStats$Degree <- NULL # THis is already part of the data

## Checked: this works better with maxLik rather than individual max alpha across all parameters
allAlphaAtMaxLik <- dat[dat$N==0.1 & dat$MN==1 & dat$MY==1.45 & dat$Y==2, ]
dim(allAlphaAtMaxLik)
table(allAlphaAtMaxLik$Alpha)
write.csv(allAlphaAtMaxLik, "../data/allAlphaAtMaxLikegoLoopTrue-SUMNORM.csv", row.names = FALSE)

allAlphaAtMaxLik <- read.csv("../data/allAlphaAtMaxLikegoLoopTrue-SUMNORM.csv", stringsAsFactors = FALSE)
dim(allAlphaAtMaxLik)

indivAlpha <- lapply(unique(allAlphaAtMaxLik$IndivID), function(i) {
  sub <- allAlphaAtMaxLik[allAlphaAtMaxLik$IndivID==i,]
  sub[which.max(sub$Lik),]
  # sub[nnet::which.is.max(sub$Lik),]  # breaks ties at random
})
indivAlpha <- rbindlist(indivAlpha)
indivAlpha <- merge(indivAlpha, weights[,c("TeamID", "SubjectID", "HumanCorrectWeighted")])
indivAlpha <- merge(indivAlpha, netStats)
indivAlpha$BetweennessCentralityZ <- scale(indivAlpha$BetweennessCentrality)
indivAlpha$AlphaZBetweennessZ <- indivAlpha$AlphaZ * indivAlpha$BetweennessCentralityZ

########################################
## Team-Level Analysis
########################################
teamLevel <- indivAlpha[,.( TeamPerf=mean(HumCor),
                            TeamPerfMajVote=names(which.max(table(HumanAns)))==2,
                            ToMMean=mean(Alpha),
                            BetweennessMean=mean(BetweennessCentrality),
                            IndivLvlToMDegreeMean=mean(AlphaZ*DegreeZ),
                            IndivLvlToMBetweennessMean=mean(AlphaZ*BetweennessCentralityZ)
                          ), by=.(TeamID)]

write.csv(teamLevel, file="../data/TeamLevelAgg.csv", row.names = FALSE)

####################################
## Table: Predict Indiv and team performance from ToM
####################################
models <- list(
  lm(  HumCor ~ Alpha, indivAlpha),
  lm(  HumCor ~ Alpha * BetweennessCentralityZ, indivAlpha),
  lm(TeamPerf ~ ToMMean, teamLevel),
  lm(TeamPerf ~ ToMMean  + BetweennessMean + ToMBetweennessMean, teamLevel),
  lm(TeamPerfMajVote ~ ToMMean + BetweennessMean + ToMBetweennessMean, teamLevel)
)
screenreg(models, stars=c(.001, .01, .05, .1))


####################################
## Figure: Team-level ToM predicts performance
####################################
ggplot(teamLevel, aes(ToMMean, TeamPerf) ) +
  geom_jitter() +
  theme_classic() +
  geom_smooth(method="lm", color=cols[2], fill="#B0B0B0") +
  scale_y_continuous(labels = scales::percent) +
  labs(x=expression( paste("Average ToM Ability ", alpha)), y="Team Performance (% Correct)") + # TODO add alpha
  annotate("text", x=0.8, y=0.99, label = paste0("italic(R) ^ 2 == ", round( summary(models[[5]])$r.squared, 2) ), parse = TRUE)
# ggsave("FIGURES/Rplot-TeamLevel-ToM-Performance.pdf", width=5, height=3.333333)

