rm(list = ls())
dat <- read.csv('./output/dataframe/amp_pha_dist_reg.csv')
dat$Emotion <- factor(dat$Emotion, levels = c('anger', 'happiness', 'sadness', 'tenderness'))
dat$Piece <- factor(dat$Piece)
dat$ID <- factor(dat$ID)

amp_m <- lmerTest::lmer(
  Amp_dist ~ Emotion + (1 | ID),
  data = dat
)
summary(amp_m)

pha_m <- lmerTest::lmer(
  Pha_dist ~ Emotion + (1 | ID),
  data = dat
)
summary(pha_m)