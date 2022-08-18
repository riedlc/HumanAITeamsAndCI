#################################################
##    Human-Agent Teams -- Hidden Profile
##
## Main analysis file by CR. 
## Analyses for CI'23 and AAAI'23
##

## Documentation on how to compute LR test
## * Simulate-Posterior-Distribution.pdf
## * http://modernstatisticalworkflow.blogspot.com/2017/05/model-checking-with-log-posterior.html
## * LR with more than two models: https://www.aafp.org/dam/AAFP/documents/journals/afp/Likelihood_Ratios.pdf
## * Good documentation of BDA but no LR https://vioshyvo.github.io/Bayesian_inference/chap-multi.html#marginal-posterior-distribution
## * the max of the LL is the same as the max of the L: https://www.statisticshowto.com/log-likelihood-function/
#################################################

# Change workikng directory
setwd("~/Dropbox/Human-Agent-Teams-Hidden-Profile")

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
# Alpha   N  MN  MY   Y NumObs HumanPerf   ToMPerf Agreement          Lik   LogLik LogLikPrioOnly
#  0.45 0.05 0.75 1.25 1.95    144 0.6666667 0.7361111 0.8263889   0.4791667 3.563747e-52 -118.4636      -195.9904 0.4032881

## Maximum performance
setLevel[which.max(setLevel$ToMPerf),]
# Alpha    N   MN   MY Y NumObs HumanPerf   ToMPerf Agreement          Lik    LogLik LogLikPrioOnly
#  0.95 0.35 0.85 1.95 2    144 0.6666667 0.7986111 0.7291667   0.4513889 2.009154e-48 -109.8264       -187.844 0.7127229

## MLE - Maximum Likelihood: What are the most likely model parameters alpha, N, MN, Y, MY?
setLevel[which.max(setLevel$LogLik),]
#   Alpha    N   MN MY Y NumObs HumanPerf   ToMPerf Agreement          Lik    LogLik LogLikPrioOnly
# 1  0.95 0.1 1.00 1.45 2    144 0.6666667 0.7430556 0.7569444   0.4861111 4.863937e-47 -106.6397      -189.7005 0.7014729

## Self-actualization only
head(setLevel[setLevel$Alpha==0,], 5)
# Alpha    N   MN MY Y NumObs HumanPerf   ToMPerf Agreement          Lik    LogLik LogLikPrioOnly
# 695992     0 0.05 0.05 1.50 2    144 0.6666667 0.6388889 0.7291667   0.4930556 2.701369e-64 -146.3717      -190.2662 0.2211926

# Maximum likelihood for PriorOnly
setLevel[which.max(setLevel$LogLikPrioOnly),]
# Alpha    N   MN MY Y NumObs HumanPerf ToMPerf Agreement          Lik    LogLik LogLikPrioOnly
# 1     1 0.05 0.05  2 2    144 0.6666667    0.75 0.7291667 2.630474e-62 -141.7931      -215.9055

# mle for partnerLoopFalse
# ONLY applies for egoLoopFalse data - no self-actualization
# head(setLevel[setLevel$Alpha==1,], 5)
#   Alpha    N MN   MY Y NumObs HumanPerf   ToMPerf Agreement Agreement.1          Lik    LogLik LogLikPrioOnly        KL
# 1     1 0.15  1 1.55 2    144 0.6666667 0.7638889 0.6666667   0.4166667 9.017002e-59 -133.6534      -188.0829 0.5772996



sub <- dat[dat$Alpha==0.3 & dat$N==0.1 & dat$MN==0.4 & dat$MY==2 & dat$Y==2,]
dim(sub)
mean(sub$PriorCor)
mean(sub$HumPriorMatch)


###########################################
##  1. find largest Lik for some para theta in the nulll model; find maximum lik estimator of theta
##  2. do same for alternative 
##
## Compare 4 models
## (a) random 
## (b) just clues (only prior)
## (c) self-loop only (alpha=0)
## (d) ToM (alpha=0.6)
##
## \gamma_LR = -2*log( likNull / likFull) OR   --- this is documented nicely on 
## \gamma_LR = -2 * (llNull - llFull)
## Then check if \gamma_LR > chi^2 -- definitely GREATER: if you google chi square critical value tables, it says "larger than"
##    so it needs to be pchisq(..., lower.tail=FALSE)
###########################################

# Random
llRandom <- log(1/5) * 144
dfRandom <- 0

# Prior Only
llPrior <- -215.9055
dfPrior <- 3            # Y, MY, N, MN: technically these should all be be +1 "use prior" should be a parameter
                        # can't do LR test with DF=0

# Self-Loop only (prior + outgoing) -- alpha=0, other paras at maxLik
llSelfOnly <- -187.5768 # alpha=0, otherwise maxLik
dfSelfOnly <- 4         # alpha=0 so one DF less, Y, MY, N, MN

# ToM MLE -- alpha=0.6 (at MaxLik)
llToMMLE <- -106.6397      # best with .6   highest posterior odds
dfToMMLE <- 5              # alpha, Y, MY, N, MN

# ToM Max performance
llToMMaxPerf <- -109.8264      # best with .6   highest posterior odds
dfToMMaxPerf <- 5              # alpha, Y, MY, N, MN

# Comparisons
# q = null model / alternative <= FOR raw likelihood (not logLik)
# p-value is -2 * q
# but with LL it should be chi = -2 * (llNull - llAlternative)
# difference in size of the parameter space between null and alternative models.
qSelfVsToM <- -2 * (llSelfOnly - llToM)
pchisq(qSelfVsToM, dfToM-dfSelfOnly, lower.tail = FALSE)

# ToM MLE vs. ToM MaxPerf
q <- -2 * (llToMMaxPerf - llToMMLE)
pchisq(q, 1, lower.tail = FALSE)

# Random vs. Prior
q <- -2 * (llRandom - llPrior)
pchisq(q, dfPrior-dfRandom, lower.tail = FALSE)


pchisq(qSelfVsToM, dfPrior-dfRandom, lower.tail = FALSE)

curve(dchisq(x, df = 1), from = 0, to = 40)


##################################################
## Plot marginal posteriors (integrating out other nuisance parameters)
## Documentation: https://vioshyvo.github.io/Bayesian_inference/chap-multi.html#marginal-posterior-distribution
##################################################
sub0 <- setLevel[setLevel$Alpha==0,]
dim(sub0)
m0 <- aggregate( LogLik ~ N, sub0, mean); m0$Alpha <- 0

sub1 <- setLevel[setLevel$Alpha==1,]
dim(sub1)
m1 <- aggregate( LogLik ~ N, sub1, mean); m1$Alpha <- 1

melt <- rbind(m0, m1)
melt$Alpha <- factor(melt$Alpha, labels=c("Ego Only", "ToM"))

ggplot(melt, aes(x=N, y=LogLik, color=Alpha, group=Alpha)) +
  geom_line() +
  theme_classic() +
  scale_color_discrete(NULL) +
  theme(legend.position = "bottom")
ggsave("FIGURES DIRECTORY/Rplot-marginal-posterior-N.pdf", width = 5, height=4.5)



subMaxLikSelfOnly <- dat[dat$Alpha==0   & dat$N==0.2 & dat$MN==0.1 & dat$MY==2 & dat$Y==2, ]
subMaxLikSelfOnly$Model <- "Ego Only"
dim(subMaxLikSelfOnly)
hist(subMaxLikSelfOnly$Lik)
subMaxLikToM      <- dat[dat$Alpha==0.6 & dat$N==0.5 & dat$MN==1.0 & dat$MY==2 & dat$Y==2, ]
subMaxLikToM$Model <- "ToM"
dim(subMaxLikToM)
hist(subMaxLikToM$Lik)

melt <- rbind(subMaxLikSelfOnly, subMaxLikToM)
ggplot(melt, aes(Lik, fill=Model, color=Model)) +
  geom_density(alpha=.4) +
  theme_classic()
ggsave("FIGURES DIRECTORY/Rplot-density-of-likelihoods.pdf", width = 5, height=4.5)


###########################################
## Marginals across alpha
###########################################
agg <- aggregate( LogLik ~ Alpha + MY, dat, mean)
ggplot(agg, aes(x=Alpha, y=LogLik, color=MY, group=MY) ) +
  geom_line()
ggsave("FIGURES DIRECTORY/Rplot-marginals-alpha-MY.pdf")

# Alpha works differently for N vs. (MY, Y)
#   N: high weight + high alpha are good
#   Y: little change but maybe: low=> high alpha; high => medium
#   MY: low=> high alpha; high => medium
#   NM: no difference across NM; alpha=0.75 is always best

# Integrating out alpha, this is the best fit:
agg[which.max(agg$LogLik),]
#        N MN Y MY     LogLik
# 6594 0.4  1 2  2 -0.7797316
#############################################



# Compute overlap with random model -- that's not in dat because we didn't actually run a random model
rnd <- subMaxLikSelfOnly[,c("HumanAns")]
# Human correct: 
96 / 144
rnd <- rnd[rep(1:144, each=100),]
dim(rnd)
rnd$RandModel <- sample(0:4, nrow(rnd), replace=TRUE)
rnd$Agreement <- rnd$HumanAns == rnd$RandModel
mean(rnd$Agreement)




#################################################
#################################################
## ToM Analysis
## 
## Is ToM predictive of individual and team performance?
#################################################
#################################################
weights <- read.csv("../data/responses.csv", stringsAsFactors = FALSE); weights$X <- NULL
names(weights)[names(weights)=="GAMEID"] <- "TeamID"
weights$SubjectID <- weights$SubjectID - 1
weights <- weights[weights$Cul_Conf!=0,]
dim(weights)
weights$CulpritWeight <- (weights$Cul_Conf -1 ) / 4
table(weights$CulpritWeight)
weights$HumanCorrectWeighted <- ifelse( weights$Culprit == 3, 1 + weights$CulpritWeight, 0 - weights$CulpritWeight)

netStats <- read.csv("..data/network_measures.csv", stringsAsFactors = FALSE); netStats$X <- NULL
netStats$Degree <- NULL # THis is already part of the data

## Checked: this works better with maxLik rather than individual max alpha across all parameters
allAlphaAtMaxLik <- dat[dat$N==0.1 & dat$MN==1 & dat$MY==1.45 & dat$Y==2, ]
dim(allAlphaAtMaxLik)
table(allAlphaAtMaxLik$Alpha)
write.csv(allAlphaAtMaxLik, "../data/allAlphaAtMaxLikegoLoopTrue-SUMNORM.csv", row.names = FALSE)

# allAlphaAtMaxLik <- read.csv("DATA/allAlphaAtMaxLikegoLoopTrue.csv", stringsAsFactors = FALSE)
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
indivAlpha$AlphaZ <- scale(indivAlpha$Alpha)
indivAlpha$DegreeZ <- scale(indivAlpha$Degree)
indivAlpha$BetweennessCentralityZ <- scale(indivAlpha$BetweennessCentrality)
indivAlpha$AlphaZDegreeZ      <- indivAlpha$AlphaZ * indivAlpha$DegreeZ
indivAlpha$AlphaZBetweennessZ <- indivAlpha$AlphaZ * indivAlpha$BetweennessCentralityZ


## Some simple stats
dim(indivAlpha)
hist(indivAlpha$Alpha, breaks=30)
table(indivAlpha$Alpha)

cor.test(indivAlpha$Alpha, indivAlpha$HumCor)  # 0.5391763 p < 0.001
cor.test(indivAlpha$Degree, indivAlpha$HumCor) # NS -- makes sense because network is assigned randoomly


#########################
## Figure: Individual scatter plot
#########################
ggplot(indivAlpha, aes(Alpha, HumCor) ) +
  geom_jitter() +
  theme_classic() +
  geom_smooth(method="lm")


########################################
## Team-Level Analysis
########################################
teamLevel <- indivAlpha[,.( TeamPerf=mean(HumCor),
                            TeamPerfW=mean(HumanCorrectWeighted),
                            TeamPerfMajVote=names(which.max(table(HumanAns)))==2,
                            ToMMean=mean(Alpha),
                            ToMMin=min(Alpha),
                            ToMSD=sd(Alpha),
                            DegreeMean=mean(Degree),
                            BetweennessMean=mean(BetweennessCentrality),
                            IndivLvlToMDegreeMean=mean(AlphaZ*DegreeZ),
                            IndivLvlToMBetweennessMean=mean(AlphaZ*BetweennessCentralityZ)
                          ), by=.(TeamID)]

write.csv(teamLevel, file="../data/TeamLevelAgg.csv", row.names = FALSE)

models <- list(
  # lm(HumCor ~ Alpha, indivAlpha),
  lm(TeamPerf ~ ToMMean, teamLevel),
  lm(TeamPerfMajVote ~ ToMMean, teamLevel),
  lm(TeamPerf ~ ToMMean + ToMMin, teamLevel),
  lm(TeamPerf ~ ToMMean + ToMSD, teamLevel),
  lm(TeamPerf ~ ToMMean + DegreeMean + IndivLvlToMDegreeMean, teamLevel),
  lm(TeamPerf ~ ToMMean + BetweennessMean + IndivLvlToMBetweennessMean, teamLevel),
  # lm(TeamPerf ~ DegreeMean, teamLevel),
  # lm(TeamPerf ~ ToMDegreeMean, teamLevel),
  # lm(TeamPerf ~ ToMMean + ToMDegreeMean, teamLevel),
  lm(TeamPerf ~ ToMMean + DegreeMean + IndivLvlToMDegreeMean, teamLevel),
  lm(TeamPerf ~ ToMMean + DegreeMean + IndivLvlToMBetweennessMean, teamLevel),
  lm(       TeamPerf ~ ToMMean + BetweennessMean + IndivLvlToMBetweennessMean, teamLevel),
  lm(       TeamPerfW ~ ToMMean + BetweennessMean + IndivLvlToMBetweennessMean, teamLevel),
  lm(TeamPerfMajVote ~ ToMMean + BetweennessMean + IndivLvlToMBetweennessMean, teamLevel)
)
screenreg(models, stars=c(.001, .01, .05, .1))


####################################
## Table: Predict Indiv and team performance from ToM
##    Combined table for CI / AAAI paper
####################################
models <- list(
  lm(  HumCor ~ Alpha, indivAlpha),
  lm(  HumCor ~ Alpha + Degree, indivAlpha),
  lm(  HumCor ~ Alpha * Degree, indivAlpha),
  lm(  HumanCorrectWeighted ~ Alpha, indivAlpha),
  lm(  HumCor ~ Alpha * BetweennessCentralityZ, indivAlpha),
  lm(  HumanCorrectWeighted ~ Alpha * BetweennessCentralityZ, indivAlpha),
  lm(TeamPerf ~ ToMMean, teamLevel),
  lm(TeamPerf ~ ToMMean  + BetweennessMean + ToMBetweennessMean, teamLevel),
  lm(TeamPerfW ~ ToMMean  + BetweennessMean + ToMBetweennessMean, teamLevel),
  lm(TeamPerfMajVote ~ ToMMean + BetweennessMean + ToMBetweennessMean, teamLevel)
)
screenreg(models, stars=c(.001, .01, .05, .1))
# We find that the effect of ToM ability is moderated by betweeness centrality ($\beta = 0.15; p = 0.090$) suggesting that high-degree network positions require a higher degree of ToM ability in order to perform well. 


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

