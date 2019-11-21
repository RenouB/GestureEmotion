library(tidyverse)
df0 <- read.csv("../data/df0.csv")
df1 <- read.csv("../data/df1.csv")
df2 <- read.csv("../data/df2.csv")
df3 <- read.csv("../data/df3.csv")
df4 <- read.csv("../data/df4.csv")
df5 <- read.csv("../data/df2.csv")

# aggregate all dfs together into a mean df
mean_df <- (df0+df1+df2+df3+df4) / 5

# seperate emotions and affect into different dfs
emotions <- data.frame(mean_df$Anger, mean_df$Happiness, mean_df$Sadness, mean_df$Surprise)
colnames(emotions) <- c("Anger", "Happiness", "Sadness", "Surprise")
affect <- data.frame(mean_df$Activation, mean_df$Anticipation, mean_df$Valence, mean_df$Authenticity)
colnames(affect) <- c("Activation", "Anticipation", "Power", "Valence")

# transpose emotions and convert back into df
emotions_t = t(emotions)
emotions <- data.frame(emotions_t)
# get standard deviation, mean for each emotion. count number of times this emotion was > 1
emotions <- transform(emotions_t, SD=apply(emotions_t, 1, sd), MEAN=apply(emotions_t,1,mean), 
                      COUNT=rowSums(emotions > 0))
# normalize count by number of observations
emotions$NORM_COUNT <- emotions$COUNT / dim(mean_df)[1]

# transpose emotions and convert back into df
affect_t = t(affect)
affect <- data.frame(affect_t)
# get standard deviation, mean for each affect. count number of times this emotion was > 1
affect <- transform(affect_t, SD=apply(affect_t, 1, sd), MEAN=apply(affect_t,1,mean),
                    COUNT=rowSums(emotions > 0))
# normalize count by number of observations
affect$NORM_COUNT <- affect$COUNT / dim(mean_df)[1]


# plot emotion mean intensity
p <- ggplot(data=emotions, aes(x= reorder(row.names(emotions), -emotions$MEAN),  y=emotions$MEAN))
p + geom_bar(stat = "identity") + labs(x = "Emotion", y = "Mean Intensity") + ylim(-.2,.6) + geom_errorbar(aes(ymin=emotions$MEAN-emotions$SD, ymax=emotions$MEAN+emotions$SD))
  
# plot affect mean value
p <- ggplot(data=affect, aes(x= reorder(row.names(affect), -affect$MEAN),  y=affect$MEAN))
p + geom_bar(stat = "identity") + labs(x = "Affect", y = "Mean Value") + geom_errorbar(aes(ymin=affect$MEAN-affect$SD, ymax=affect$MEAN+affect$SD))
# plot normalized emotion count
p <- ggplot(data=emotions, aes(x= reorder(row.names(emotions), -emotions$NORM_COUNT),  y=emotions$NORM_COUNT))
p + geom_bar(stat = "identity") + labs(x = "Emotion", y = "Percentage of timepoints") + ylim(0,1)
