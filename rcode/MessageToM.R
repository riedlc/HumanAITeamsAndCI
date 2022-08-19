#################################################
##    Human-Agent Teams -- Hidden Profile
##
## Analysis of outgoing message ToM
#################################################

# Change workikng directory
setwd("YOUR WORKING DIRECTORY")

# Load libraries
require(data.table)
require(ggplot2)
require(texreg)
source("PATH TO qcut/R-global/qcut.R")

# Configuration
cols <- c('#909090', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7')

# Load data
msg <- fread("../data/ego_vs_alter.csv", stringsAsFactors = FALSE); msg$V1 <- NULL
msg$IndivID <- paste(msg$TeamID, msg$SubjectIDSender, sep="-")
msg$Count <- 1

## Aggregate messages to individual-level
indivLevel <- msg[,.( NumMsg = sum(Count),
               NumY   = sum(MessageType=="Y"),
               MeanEgoSurprise = mean(SurpriseAccordingToSendersEgoModel),
               MeanEgoSurpriseTypeY = mean(SurpriseAccordingToSendersEgoModel[MessageType=="Y"]),
               MeanAlterSurprise = mean(SurpriseAccordingToSendersAlterModelOfReceiver)
              ), by=.(IndivID)]
indivLevel$MeanEgoSurpriseTypeY[is.na(indivLevel$MeanEgoSurpriseTypeY)] <- 0
indivLevel$TeamID <- as.numeric( substr(indivLevel$IndivID, 1, 6) )
summary(indivLevel)

# Aggregate individual to team-level
teamLevel <- indivLevel[,.( NumMsg = sum(NumMsg),
                     NumY   = sum(NumY),
                     MeanEgoSurprise = mean(MeanEgoSurprise),
                     MeanEgoSurpriseTypeY = mean(MeanEgoSurpriseTypeY),
                     MeanAlterSurprise = mean(MeanAlterSurprise)
                    ), by=.(TeamID)]

# Merge team-level performance data
# This csv is generated in LikAnalysis.R
teamLevelPerf <- read.csv("../data/TeamLevelAgg.csv", stringsAsFactors = FALSE)

teamLevel <- merge(teamLevel, teamLevelPerf)


####################################
## Figure: Team-level ToM predicts performance
####################################
ggplot(teamLevel, aes(MeanAlterSurprise, TeamPerf) ) +
  geom_jitter() +
  theme_classic() +
  geom_smooth(method="lm", color=cols[2], fill="#B0B0B0") +
  scale_y_continuous(labels = scales::percent) +
  labs(x=expression( paste("Average ToM Communication Ability ", alpha[C] )), y="Team Performance (% Correct)") + 
  annotate("text", x=0.63, y=0.57, label = paste0("italic(R) ^ 2 == ", round( summary(models[[3]])$r.squared, 2) ), parse = TRUE)
# ggsave("FIGURES/Rplot-TeamLevel-ToM-Performance-Communication.pdf", width=5, height=3.333333)



####################################
## Figure: Dynamic CI 
####################################
msg <- msg[order(msg$TeamID, msg$Time),]
msg$WithinTeamTime <- unlist( lapply( unique(msg$TeamID), function(tid) {
  t <- msg$Time[msg$TeamID==tid]
  ecdf(t)(t)
}))

msg$TimeWindow <- qcut(msg$WithinTeamTime, 15)
table(msg$TimeWindow)

timeAgg <- aggregate(SurpriseAccordingToSendersAlterModelOfReceiver ~ TimeWindow + TeamID, msg, mean)
timeAgg <- merge(timeAgg, teamLevelPerf)

timeAgg$Group <- "neutral"
timeAgg$Group[timeAgg$TeamPerf >.6] <- "high"
timeAgg$Group[timeAgg$TeamPerf <=.6] <- "low"
timeAgg$TimeWindowP <- timeAgg$TimeWindow <- (timeAgg$TimeWindow - min(timeAgg$TimeWindow)) / 
                      (max(timeAgg$TimeWindow) - min(timeAgg$TimeWindow))


ggplot(timeAgg, aes(x=TimeWindowP, y=SurpriseAccordingToSendersAlterModelOfReceiver, color=Group, group=Group)) +
  geom_jitter(alpha=.5) +
  geom_smooth() + 
  # geom_smooth(method="gam") +  # This one works also really nicely
  theme_classic() +
  scale_color_manual(NULL, values=c("#2ca25f", "#e34a33")) +
  scale_x_continuous(labels = scales::percent) +
  labs(x="Time Elapsed", y="Surprise of Messages") +
  theme(legend.justification = c(1, 0), legend.position = c(1, 0.01)) +
  guides(color=guide_legend(nrow=1, byrow=TRUE))
# ggsave("FIGURES/Rplot-Surprise-Pattern.pdf", width=5, height=3.333333)


#####################################################
## Figure: Include only first X% of messages, then see when we can make significant predictions about final team performance
#####################################################
out <- lapply( 1:max(timeAgg$TimeWindow), function(t) {
  sub <- timeAgg[timeAgg$TimeWindow <= t,]
  fit <- cor.test( sub$TeamPerf, sub$SurpriseAccordingToSendersAlterModelOfReceiver)
  data.frame( T=t, CILower=fit$conf.int[1], Estimate=fit$estimate, CIUpper=fit$conf.int[2])
})

melt <- rbindlist(out)
melt$T <- (melt$T - min(melt$T)) / (max(melt$T) - min(melt$T))
ggplot(melt, aes(x=T, y=Estimate)) +
  geom_ribbon(aes(ymin=CILower, ymax=CIUpper), fill="#B0B0B0", alpha=0.5) +
  geom_line(color=cols[2]) +
  geom_hline(yintercept=0, linetype="dashed", color="grey") +
  geom_vline(xintercept=.22, linetype="dashed", color="grey") +
  theme_classic() + 
  scale_x_continuous(labels = scales::percent) +
  labs(x="Percent of Team Messages Included", y=expression( atop( paste("ToM Communication Ability ", alpha[C], " and"), 
                                                                  paste("Team Performance Correlation (", rho, ")")) ) ) +
  annotate("text", x=0.55, y=0.05, label = "Region of significant prediction")
# ggsave("FIGURES/Rplot-TeamLevel-ToM-Performance-Communication-Over-Time.pdf", width=5, height=3.333333)
