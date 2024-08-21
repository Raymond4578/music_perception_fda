rm(list = ls())
# Load the dataset
dat <- read.csv('./output/dataframe/amp_pha_dist_reg.csv')
dat$Emotion <- factor(dat$Emotion, levels = c('anger', 'happiness', 'sadness', 'tenderness'))
dat$Piece <- factor(dat$Piece)
dat$ID <- factor(dat$ID)

# Construct a mixture effect model: amplitude distance ~ Emotion
amp_m <- lmerTest::lmer(
  Amp_dist ~ Emotion + (1 | ID),
  data = dat
)
summary(amp_m)

# Construct a mixture effect model: phase distance ~ Emotion
pha_m <- lmerTest::lmer(
  Pha_dist ~ Emotion + (1 | ID),
  data = dat
)
summary(pha_m)