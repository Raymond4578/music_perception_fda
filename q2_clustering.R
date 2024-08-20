rm(list = ls())
dat <- read.csv('./output/dataframe/clust_pred_results.csv')
dat$ID <- factor(dat$ID)
dat$Emotion <- factor(dat$Emotion, levels = c('anger', 'happiness', 'sadness', 'tenderness'))
dat$Piece <- factor(dat$Piece)
dat$amp_pred <- factor(dat$amp_pred, levels = c(0, 1))
dat$pha_pred <- factor(dat$pha_pred, levels = c(0, 1))

model <- lme4::glmer(
  amp_pred ~ pha_pred + (1 | ID),
  data = dat,
  family = binomial(link = 'logit')
)
summary(model)